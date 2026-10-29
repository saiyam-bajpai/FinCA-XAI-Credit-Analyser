import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.core.security import get_current_user, get_user_from_token_or_query
from backend.core.underwriting import create_application_for_user, get_xai_engine
from backend.core.uploads import delete_stored_file, resolve_stored_path, save_upload
from backend.db.database import get_db
from backend.db.models import AuditLog, DocumentRequest, LoanApplication, User
from backend.schemas.application import (
    DocumentRequestResponse,
    DocumentRequestStatusUpdate,
    LoanApplicationCreate,
    LoanApplicationResponse,
)

router = APIRouter(prefix="/api/applications", tags=["Applications"])


@router.post("/apply", response_model=LoanApplicationResponse, status_code=status.HTTP_201_CREATED)
def apply_for_loan(
    app_in: LoanApplicationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Ensures the XAI artifacts are available before we touch the database,
        # so a scoring failure never leaves a half-written application behind.
        get_xai_engine()
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI scoring engine artifact unavailable: {err}"
        )

    return create_application_for_user(db, current_user, app_in)


@router.get("/my-applications", response_model=List[LoanApplicationResponse])
def get_my_applications(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    applications = (
        db.query(LoanApplication)
        .filter(LoanApplication.applicant_id == current_user.id)
        .order_by(LoanApplication.created_at.desc())
        .all()
    )
    return applications


@router.post("/documents/{document_id}/upload", response_model=DocumentRequestResponse)
def upload_applicant_document(
    document_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Applicant uploads a real document file (PDF or Image, max 10 MB)."""
    document = (
        db.query(DocumentRequest)
        .join(LoanApplication, DocumentRequest.application_id == LoanApplication.id)
        .filter(
            DocumentRequest.id == document_id,
            LoanApplication.applicant_id == current_user.id,
        )
        .first()
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document request with ID {document_id} not found."
        )

    # Persist and validate file type (PDF/Images) and size (10 MB limit)
    stored = save_upload(file)

    # Remove previously stored file if applicant is re-uploading
    if document.stored_path:
        delete_stored_file(document.stored_path)

    document.file_name = stored.file_name
    document.stored_path = stored.stored_path
    document.mime_type = stored.mime_type
    document.file_size = stored.file_size
    document.uploaded_at = datetime.datetime.utcnow()
    document.uploaded_by = current_user.id
    document.status = "UPLOADED"

    db.add(
        AuditLog(
            application_id=document.application_id,
            officer_id=None,
            action="DOCUMENT_UPLOADED",
            notes=f"Applicant uploaded file '{stored.file_name}' ({stored.mime_type}, {round(stored.file_size / 1024, 1)} KB) for '{document.doc_name}'.",
        )
    )

    db.commit()
    db.refresh(document)
    return document


@router.get("/documents/{document_id}/file")
def get_applicant_document_file(
    document_id: int,
    download: bool = Query(False, description="Set to true to force download instead of inline view"),
    current_user: User = Depends(get_user_from_token_or_query),
    db: Session = Depends(get_db)
):
    """View or download the uploaded document file. Only accessible to applicant owner or loan officers."""
    document = (
        db.query(DocumentRequest)
        .join(LoanApplication, DocumentRequest.application_id == LoanApplication.id)
        .filter(DocumentRequest.id == document_id)
        .first()
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document request with ID {document_id} not found."
        )

    # Check permission: applicant must own the application, or user must be a LOAN_OFFICER
    is_owner = document.application.applicant_id == current_user.id
    is_officer = current_user.role == "LOAN_OFFICER"
    if not (is_owner or is_officer):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to view this document."
        )

    if not document.stored_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No file has been uploaded for this document request."
        )

    file_path = resolve_stored_path(document.stored_path)
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored document file could not be found on server."
        )

    disposition = "attachment" if download else "inline"
    return FileResponse(
        path=str(file_path),
        filename=document.file_name or "document",
        media_type=document.mime_type or "application/octet-stream",
        content_disposition_type=disposition
    )


@router.patch("/documents/{document_id}", response_model=DocumentRequestResponse)
def upload_requested_document(
    document_id: int,
    update_in: DocumentRequestStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Applicant marks a requested document as uploaded (metadata update fallback)."""
    document = (
        db.query(DocumentRequest)
        .join(LoanApplication, DocumentRequest.application_id == LoanApplication.id)
        .filter(
            DocumentRequest.id == document_id,
            LoanApplication.applicant_id == current_user.id,
        )
        .first()
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document request with ID {document_id} not found."
        )

    if update_in.status.upper() != "UPLOADED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Applicants may only mark a document as UPLOADED."
        )

    document.status = "UPLOADED"

    db.add(
        AuditLog(
            application_id=document.application_id,
            officer_id=None,
            action="DOCUMENT_UPLOADED",
            notes=f"Applicant marked document as uploaded: {document.doc_name}",
        )
    )

    db.commit()
    db.refresh(document)
    return document
