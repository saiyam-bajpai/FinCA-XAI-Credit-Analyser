from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.core.finance_math import calculate_emi, calculate_foir_max_loan_limit
from backend.core.security import get_current_user
from backend.db.database import get_db
from backend.db.models import AuditLog, DocumentRequest, LoanApplication, User
from backend.ml.explainability import XAIEngine
from backend.schemas.application import LoanApplicationCreate, LoanApplicationResponse

router = APIRouter(prefix="/api/applications", tags=["Applications"])

# Global XAI Engine Instance
try:
    xai_engine = XAIEngine()
except Exception as e:
    print(f"Warning: XAIEngine initialization delayed until model artifact generation ({e})")
    xai_engine = None


@router.post("/apply", response_model=LoanApplicationResponse, status_code=status.HTTP_201_CREATED)
def apply_for_loan(
    app_in: LoanApplicationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    global xai_engine
    if xai_engine is None:
        try:
            xai_engine = XAIEngine()
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI scoring engine artifact unavailable: {err}"
            )

    # 1. Financial Math Calculations
    foir_data = calculate_foir_max_loan_limit(
        annual_income=app_in.income,
        existing_emis=app_in.existing_emis,
        tenure_months=app_in.tenure
    )
    calculated_emi = calculate_emi(
        principal=app_in.loan_amount,
        tenure_months=app_in.tenure
    )

    # 2. Prepare dict for ML / XAI Engine
    applicant_dict = {
        "income": app_in.income,
        "debt": app_in.debt,
        "credit_score": app_in.credit_score,
        "loan_amount": app_in.loan_amount,
        "loan_term": app_in.tenure,
        "employment_length": app_in.employment_length,
        "existing_emis": app_in.existing_emis,
        "dependents": app_in.dependents
    }

    # 3. Execute Explainable AI scoring
    xai_output = xai_engine.explain_instance(applicant_dict)

    decision_info = xai_output["decision"]
    prob = decision_info["approval_probability"]
    ai_score = decision_info["finCA_credit_score"]

    # 4. Risk Tier Classification
    if prob >= 0.70:
        risk_tier = "Low"
        initial_status = "UNDER_REVIEW"
    elif prob >= 0.45:
        risk_tier = "Medium"
        initial_status = "DOCS_REQUIRED"
    else:
        risk_tier = "High"
        initial_status = "REJECTED"

    # Rejection & Recourse formatting
    rejection_reason = None
    time_to_fix_months = None

    if initial_status == "REJECTED":
        top_neg = xai_output.get("top_negative_factors", [])
        if top_neg:
            neg_descriptions = [f"{factor['feature']} (impact: {factor['shap_impact']})" for factor in top_neg]
            rejection_reason = "Automated credit risk threshold exceeded based on: " + ", ".join(neg_descriptions)
        else:
            rejection_reason = "High debt-to-income and credit risk indicators."

        cfs = xai_output.get("counterfactual_recommendations", [])
        if cfs and len(cfs) > 0:
            time_to_fix_months = cfs[0].get("estimated_time_to_fix", "3-6 Months")

    # 5. Create DB Record
    new_application = LoanApplication(
        applicant_id=current_user.id,
        income=app_in.income,
        credit_score=app_in.credit_score,
        debt=app_in.debt,
        loan_amount=app_in.loan_amount,
        tenure=app_in.tenure,
        employment_length=app_in.employment_length,
        existing_emis=app_in.existing_emis,
        dependents=app_in.dependents,
        max_eligible_loan=foir_data["max_eligible_loan_amount"],
        calculated_emi=calculated_emi,
        foir_ratio=foir_data["foir_ratio"],
        status=initial_status,
        ai_score=ai_score,
        approval_probability=prob,
        risk_tier=risk_tier,
        rejection_reason=rejection_reason,
        time_to_fix_months=time_to_fix_months,
        xai_breakdown_json=xai_output
    )

    db.add(new_application)
    db.flush()

    # 6. Automatic Document Requests if required
    if initial_status == "DOCS_REQUIRED" or risk_tier == "Medium":
        doc1 = DocumentRequest(application_id=new_application.id, doc_name="Bank Statement (6 Months)", status="PENDING")
        doc2 = DocumentRequest(application_id=new_application.id, doc_name="Tax Return (Recent Year)", status="PENDING")
        db.add_all([doc1, doc2])

    # 7. Audit Log
    audit = AuditLog(
        application_id=new_application.id,
        officer_id=None,
        action="SUBMITTED",
        notes=f"Application submitted. AI Score: {ai_score}, Risk Tier: {risk_tier}, Status: {initial_status}"
    )
    db.add(audit)

    db.commit()
    db.refresh(new_application)

    return new_application


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
