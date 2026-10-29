"""Shared underwriting helpers.

Both the applicant self-service flow (``/api/applications/apply``) and the
officer-assisted flow (``/api/officer/applications``) use the same financial
math and XAI scoring pipeline so decisions stay consistent regardless of who
creates the application.
"""

from typing import Optional

from sqlalchemy.orm import Session

from backend.core.finance_math import calculate_emi, calculate_foir_max_loan_limit
from backend.db.models import AuditLog, DocumentRequest, LoanApplication, User
from backend.ml.explainability import XAIEngine
from backend.schemas.application import LoanApplicationCreate

# Documents auto-requested when the engine flags the file as medium risk.
AUTO_DOCUMENTS = ("Bank Statement (6 Months)", "Tax Return (Recent Year)")

_xai_engine: Optional[XAIEngine] = None


def get_xai_engine() -> XAIEngine:
    """Return the (lazily initialised) XAI engine singleton."""
    global _xai_engine
    if _xai_engine is None:
        _xai_engine = XAIEngine()
    return _xai_engine


def evaluate_application(app_in: LoanApplicationCreate) -> dict:
    """Run financial math + XAI scoring for an application payload.

    Returns every scored field needed to build a ``LoanApplication`` row plus a
    ``required_documents`` list that should be persisted as document requests.
    """
    xai_engine = get_xai_engine()

    foir_data = calculate_foir_max_loan_limit(
        annual_income=app_in.income,
        existing_emis=app_in.existing_emis,
        tenure_months=app_in.tenure,
    )
    calculated_emi = calculate_emi(
        principal=app_in.loan_amount,
        tenure_months=app_in.tenure,
    )

    applicant_dict = {
        "income": app_in.income,
        "debt": app_in.debt,
        "credit_score": app_in.credit_score,
        "loan_amount": app_in.loan_amount,
        "loan_term": app_in.tenure,
        "employment_length": app_in.employment_length,
        "existing_emis": app_in.existing_emis,
        "dependents": app_in.dependents,
    }

    xai_output = xai_engine.explain_instance(applicant_dict)
    decision_info = xai_output["decision"]
    prob = decision_info["approval_probability"]
    ai_score = decision_info["finCA_credit_score"]

    if prob >= 0.70:
        risk_tier = "Low"
        initial_status = "UNDER_REVIEW"
    elif prob >= 0.45:
        risk_tier = "Medium"
        initial_status = "DOCS_REQUIRED"
    else:
        risk_tier = "High"
        initial_status = "REJECTED"

    rejection_reason = None
    time_to_fix_months = None

    if initial_status == "REJECTED":
        top_neg = xai_output.get("top_negative_factors", [])
        if top_neg:
            neg_descriptions = [
                f"{factor['feature']} (impact: {factor['shap_impact']})"
                for factor in top_neg
            ]
            rejection_reason = (
                "Automated credit risk threshold exceeded based on: "
                + ", ".join(neg_descriptions)
            )
        else:
            rejection_reason = "High debt-to-income and credit risk indicators."

        counterfactuals = xai_output.get("counterfactual_recommendations", [])
        if counterfactuals:
            time_to_fix_months = counterfactuals[0].get(
                "estimated_time_to_fix", "3-6 Months"
            )

    required_documents = []
    if initial_status == "DOCS_REQUIRED" or risk_tier == "Medium":
        required_documents = list(AUTO_DOCUMENTS)

    return {
        "max_eligible_loan": foir_data["max_eligible_loan_amount"],
        "calculated_emi": calculated_emi,
        "foir_ratio": foir_data["foir_ratio"],
        "status": initial_status,
        "ai_score": ai_score,
        "approval_probability": prob,
        "risk_tier": risk_tier,
        "rejection_reason": rejection_reason,
        "time_to_fix_months": time_to_fix_months,
        "xai_breakdown_json": xai_output,
        "required_documents": required_documents,
    }


def create_application_for_user(
    db: Session,
    applicant: User,
    app_in: LoanApplicationCreate,
    officer_id: Optional[int] = None,
) -> LoanApplication:
    """Score an application and persist it with its audit log.

    Document requests are created separately via :func:`request_documents` so
    officer-triggered requests and auto-requests share one code path.
    """
    scored = evaluate_application(app_in)
    required_documents = scored.pop("required_documents", [])

    application = LoanApplication(
        applicant_id=applicant.id,
        income=app_in.income,
        credit_score=app_in.credit_score,
        debt=app_in.debt,
        loan_amount=app_in.loan_amount,
        tenure=app_in.tenure,
        employment_length=app_in.employment_length,
        existing_emis=app_in.existing_emis,
        dependents=app_in.dependents,
        **scored,
    )
    db.add(application)
    db.flush()

    for doc_name in required_documents:
        db.add(
            DocumentRequest(
                application_id=application.id,
                doc_name=doc_name,
                status="PENDING",
            )
        )

    db.add(
        AuditLog(
            application_id=application.id,
            officer_id=officer_id,
            action="SUBMITTED",
            notes=(
                f"Application submitted. AI Score: {scored['ai_score']}, "
                f"Risk Tier: {scored['risk_tier']}, Status: {scored['status']}"
            ),
        )
    )

    db.commit()
    db.refresh(application)
    return application


def request_documents(
    db: Session,
    application: LoanApplication,
    doc_names,
    officer_id: Optional[int] = None,
) -> list:
    """Persist document requests for an application, skipping duplicates.

    Returns the full, current list of document requests for the application so
    callers can send the authoritative state back to the client.
    """
    existing = {doc.doc_name.strip().lower() for doc in application.document_requests}
    created = []

    for raw_name in doc_names:
        doc_name = (raw_name or "").strip()
        if not doc_name:
            continue
        if doc_name.lower() in existing:
            continue
        doc = DocumentRequest(
            application_id=application.id,
            doc_name=doc_name,
            status="PENDING",
        )
        db.add(doc)
        existing.add(doc_name.lower())
        created.append(doc)

    if created:
        db.add(
            AuditLog(
                application_id=application.id,
                officer_id=officer_id,
                action="DOCS_REQUESTED",
                notes="Additional documents requested: "
                + ", ".join(doc.doc_name for doc in created),
            )
        )

    db.commit()
    db.refresh(application)
    return application.document_requests
