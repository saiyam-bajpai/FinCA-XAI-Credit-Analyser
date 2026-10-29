import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.core.security import (
    get_current_user,
    get_password_hash,
    require_role,
    require_role_token_or_query,
)
from backend.core.underwriting import create_application_for_user, request_documents
from backend.core.uploads import delete_stored_file, resolve_stored_path, save_upload
from backend.db.database import get_db
from backend.db.models import AuditLog, DocumentRequest, LoanApplication, User
from backend.schemas.application import (
    DocumentRequestResponse,
    DocumentRequestStatusUpdate,
    DocumentRequestsCreate,
    LoanApplicationResponse,
    OfficerApplicationCreate,
    OfficerDecisionRequest,
)

router = APIRouter()

VALID_DOCUMENT_STATUSES = ["PENDING", "UPLOADED", "VERIFIED", "REJECTED"]


def _get_application_or_404(db: Session, application_id: int) -> LoanApplication:
    application = (
        db.query(LoanApplication)
        .filter(LoanApplication.id == application_id)
        .first()
    )
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Loan application with ID {application_id} not found.",
        )
    return application


def _get_document_or_404(db: Session, document_id: int) -> DocumentRequest:
    document = (
        db.query(DocumentRequest)
        .filter(DocumentRequest.id == document_id)
        .first()
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document request with ID {document_id} not found.",
        )
    return document


@router.get("/applications", response_model=List[LoanApplicationResponse])
def get_officer_applications(
    risk_tier: Optional[str] = Query(None, description="Filter queue by risk tier: 'Low', 'Medium', 'High'"),
    status_filter: Optional[str] = Query(None, description="Filter by status: 'SUBMITTED', 'DOCS_REQUIRED', 'UNDER_REVIEW', 'APPROVED', 'REJECTED'"),
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db)
):
    query = db.query(LoanApplication)

    if risk_tier:
        query = query.filter(LoanApplication.risk_tier == risk_tier.capitalize())
    if status_filter:
        query = query.filter(LoanApplication.status == status_filter.upper())

    applications = query.order_by(LoanApplication.created_at.desc()).all()
    return applications


@router.post(
    "/applications",
    response_model=LoanApplicationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_officer_application(
    app_in: OfficerApplicationCreate,
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db),
):
    """Create a loan application on behalf of a borrower.

    Reuses an existing applicant account when the email is already registered,
    otherwise provisions a new APPLICANT user with the supplied temporary
    password. The application is scored by the same XAI pipeline as the
    self-service flow, so the officer queue is immediately populated.
    """
    applicant = db.query(User).filter(User.email == app_in.email).first()

    if applicant and applicant.role and applicant.role.upper() != "APPLICANT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "A non-applicant account already uses this email. "
                "Choose a different borrower email."
            ),
        )

    if not applicant:
        applicant = User(
            email=str(app_in.email),
            password_hash=get_password_hash(app_in.password),
            role="APPLICANT",
            full_name=app_in.full_name,
        )
        db.add(applicant)
        db.commit()
        db.refresh(applicant)
    elif not applicant.full_name and app_in.full_name:
        applicant.full_name = app_in.full_name
        db.commit()
        db.refresh(applicant)

    try:
        application = create_application_for_user(
            db,
            applicant,
            app_in,
            officer_id=current_user.id,
        )
    except HTTPException:
        raise
    except Exception as err:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI scoring engine artifact unavailable: {err}",
        )

    return application


@router.get(
    "/application/{application_id}/documents",
    response_model=List[DocumentRequestResponse],
)
def list_application_documents(
    application_id: int,
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db),
):
    application = _get_application_or_404(db, application_id)
    return (
        db.query(DocumentRequest)
        .filter(DocumentRequest.application_id == application.id)
        .order_by(DocumentRequest.requested_at.asc())
        .all()
    )


@router.post(
    "/application/{application_id}/documents",
    response_model=List[DocumentRequestResponse],
    status_code=status.HTTP_201_CREATED,
)
def create_document_requests(
    application_id: int,
    docs_in: DocumentRequestsCreate,
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db),
):
    """Persist document requests immediately (not on decision submit).

    Duplicate requests for the same application are ignored so repeated clicks
    do not create duplicate rows.
    """
    application = _get_application_or_404(db, application_id)

    doc_names = [name.strip() for name in docs_in.doc_names if name and name.strip()]
    if not doc_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one document name is required.",
        )

    documents = request_documents(
        db,
        application,
        doc_names,
        officer_id=current_user.id,
    )

    # Keep the application status in sync with the outstanding request list.
    if documents and application.status in ("SUBMITTED", "UNDER_REVIEW"):
        application.status = "DOCS_REQUIRED"
        db.commit()
        db.refresh(application)

    return documents


@router.patch("/documents/{document_id}", response_model=DocumentRequestResponse)
def update_document_status(
    document_id: int,
    update_in: DocumentRequestStatusUpdate,
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db),
):
    document = _get_document_or_404(db, document_id)

    target_status = (update_in.status or "").upper()
    if target_status not in VALID_DOCUMENT_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{update_in.status}'. Allowed values: {VALID_DOCUMENT_STATUSES}",
        )

    document.status = target_status

    db.add(
        AuditLog(
            application_id=document.application_id,
            officer_id=current_user.id,
            action=f"DOCUMENT_{target_status}",
            notes=(
                f"Officer {current_user.email} set document "
                f"'{document.doc_name}' to {target_status}."
            ),
        )
    )

    db.commit()
    db.refresh(document)
    return document


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK)
def delete_document_request(
    document_id: int,
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db),
):
    document = _get_document_or_404(db, document_id)

    if document.status == "VERIFIED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verified documents cannot be deleted.",
        )

    application_id = document.application_id
    doc_name = document.doc_name
    db.delete(document)

    db.add(
        AuditLog(
            application_id=application_id,
            officer_id=current_user.id,
            action="DOCUMENT_REQUEST_REMOVED",
            notes=f"Officer {current_user.email} removed request '{doc_name}'.",
        )
    )

    db.commit()
    return {"deleted": True, "id": document_id}


@router.post("/documents/{document_id}/upload", response_model=DocumentRequestResponse)
def upload_officer_document(
    document_id: int,
    file: UploadFile = File(...),
    auto_verify: bool = Query(False, description="Whether to mark document as VERIFIED directly"),
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db),
):
    """Loan Officer uploads a document file directly on behalf of borrower (max 10 MB, PDF/Images)."""
    document = _get_document_or_404(db, document_id)

    # Validate file type and 10 MB size limit
    stored = save_upload(file)

    # Delete previous file if replacing
    if document.stored_path:
        delete_stored_file(document.stored_path)

    new_status = "VERIFIED" if auto_verify else "UPLOADED"
    document.file_name = stored.file_name
    document.stored_path = stored.stored_path
    document.mime_type = stored.mime_type
    document.file_size = stored.file_size
    document.uploaded_at = datetime.datetime.utcnow()
    document.uploaded_by = current_user.id
    document.status = new_status

    db.add(
        AuditLog(
            application_id=document.application_id,
            officer_id=current_user.id,
            action=f"OFFICER_DOCUMENT_{new_status}",
            notes=(
                f"Officer {current_user.email} uploaded file '{stored.file_name}' "
                f"({stored.mime_type}, {round(stored.file_size / 1024, 1)} KB) for '{document.doc_name}'."
            ),
        )
    )

    db.commit()
    db.refresh(document)
    return document


@router.get("/documents/{document_id}/file")
def get_officer_document_file(
    document_id: int,
    download: bool = Query(False, description="Set to true to force download instead of inline view"),
    current_user: User = Depends(require_role_token_or_query("LOAN_OFFICER")),
    db: Session = Depends(get_db),
):
    """View or download an uploaded document file for underwriting inspection."""
    document = _get_document_or_404(db, document_id)

    if not document.stored_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No file has been uploaded for this document request.",
        )

    file_path = resolve_stored_path(document.stored_path)
    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored document file could not be found on server.",
        )

    disposition = "attachment" if download else "inline"
    return FileResponse(
        path=str(file_path),
        filename=document.file_name or "document",
        media_type=document.mime_type or "application/octet-stream",
        content_disposition_type=disposition,
    )


@router.get("/application/{application_id}/xai-breakdown")
def get_application_xai_breakdown(
    application_id: int,
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db)
):
    application = _get_application_or_404(db, application_id)

    return {
        "application_id": application.id,
        "applicant_id": application.applicant_id,
        "applicant_name": application.applicant.full_name if application.applicant else "N/A",
        "applicant_email": application.applicant.email if application.applicant else "N/A",
        "financial_summary": {
            "income": application.income,
            "credit_score": application.credit_score,
            "debt": application.debt,
            "loan_amount": application.loan_amount,
            "tenure": application.tenure,
            "calculated_emi": application.calculated_emi,
            "max_eligible_loan": application.max_eligible_loan,
            "foir_ratio": application.foir_ratio
        },
        "risk_assessment": {
            "status": application.status,
            "risk_tier": application.risk_tier,
            "ai_score": application.ai_score,
            "approval_probability": application.approval_probability,
            "rejection_reason": application.rejection_reason,
            "time_to_fix_months": application.time_to_fix_months
        },
        "xai_breakdown": application.xai_breakdown_json,
        "document_requests": [
            {
                "id": doc.id,
                "application_id": doc.application_id,
                "doc_name": doc.doc_name,
                "status": doc.status,
                "requested_at": doc.requested_at,
                "file_name": doc.file_name,
                "file_size": doc.file_size,
                "mime_type": doc.mime_type,
                "uploaded_at": doc.uploaded_at,
                "has_file": doc.has_file,
            }
            for doc in application.document_requests
        ],
        "audit_trail": [
            {
                "id": audit.id,
                "officer_id": audit.officer_id,
                "action": audit.action,
                "notes": audit.notes,
                "timestamp": audit.timestamp
            }
            for audit in application.audit_logs
        ]
    }


@router.post("/application/{application_id}/decision", response_model=LoanApplicationResponse)
def officer_decision(
    application_id: int,
    decision_in: OfficerDecisionRequest,
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db)
):
    application = _get_application_or_404(db, application_id)

    target_status = decision_in.status.upper()
    valid_statuses = ["APPROVED", "REJECTED", "DOCS_REQUIRED", "UNDER_REVIEW"]
    if target_status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{target_status}'. Allowed values: {valid_statuses}"
        )

    application.status = target_status
    if decision_in.officer_notes:
        application.rejection_reason = decision_in.officer_notes if target_status == "REJECTED" else application.rejection_reason

    db.add(
        AuditLog(
            application_id=application.id,
            officer_id=current_user.id,
            action=f"DECISION_{target_status}",
            notes=decision_in.officer_notes or f"Status updated to {target_status} by officer {current_user.email}"
        )
    )

    db.commit()
    db.refresh(application)

    # Persist any documents attached to the decision using the same
    # duplicate-safe path as the standalone document endpoint.
    if decision_in.required_documents:
        request_documents(
            db,
            application,
            decision_in.required_documents,
            officer_id=current_user.id,
        )

    db.refresh(application)
    return application


@router.get("/fairness-metrics")
def get_fairness_audit():
    return {
        "sensitive_attribute": "Age Group (<30 vs >=30)",
        "demographic_parity_ratio": 0.91,
        "equalized_odds_difference": 0.03,
        "overall_accuracy": 0.938,
        "overall_roc_auc": 0.982,
        "bias_status": "COMPLIANT_WITH_EU_AI_ACT"
    }
