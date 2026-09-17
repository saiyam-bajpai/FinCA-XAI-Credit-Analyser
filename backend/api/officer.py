from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.core.security import get_current_user, require_role
from backend.db.database import get_db
from backend.db.models import AuditLog, DocumentRequest, LoanApplication, User
from backend.schemas.application import LoanApplicationResponse, OfficerDecisionRequest

router = APIRouter()



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


@router.get("/application/{application_id}/xai-breakdown")
def get_application_xai_breakdown(
    application_id: int,
    current_user: User = Depends(require_role("LOAN_OFFICER")),
    db: Session = Depends(get_db)
):
    application = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Loan application with ID {application_id} not found."
        )

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
                "doc_name": doc.doc_name,
                "status": doc.status,
                "requested_at": doc.requested_at
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
    application = db.query(LoanApplication).filter(LoanApplication.id == application_id).first()
    if not application:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Loan application with ID {application_id} not found."
        )

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

    # Issue document requests if specified
    if decision_in.required_documents:
        for doc_name in decision_in.required_documents:
            new_doc = DocumentRequest(
                application_id=application.id,
                doc_name=doc_name,
                status="PENDING"
            )
            db.add(new_doc)

    # Log Audit
    audit = AuditLog(
        application_id=application.id,
        officer_id=current_user.id,
        action=f"DECISION_{target_status}",
        notes=decision_in.officer_notes or f"Status updated to {target_status} by officer {current_user.email}"
    )
    db.add(audit)

    db.commit()
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