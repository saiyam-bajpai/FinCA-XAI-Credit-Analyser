import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.core.finance_math import calculate_emi, calculate_foir_max_loan_limit
from backend.core.security import get_password_hash
from backend.db.database import SessionLocal, Base, engine
from backend.db.models import AuditLog, DocumentRequest, LoanApplication, User
from backend.ml.explainability import XAIEngine

PASSWORD = "password123"

DEMO_USERS = [
    {"email": "applicant@finca.com", "role": "APPLICANT", "full_name": "Demo Applicant"},
    {"email": "officer@finca.com", "role": "LOAN_OFFICER", "full_name": "Demo Loan Officer"},
]

TEST_APPLICANTS = [
    {"email": "lowrisk1@test.com", "full_name": "Sofia Ramirez"},
    {"email": "lowrisk2@test.com", "full_name": "Ethan Nguyen"},
    {"email": "medium1@test.com", "full_name": "Priya Patel"},
    {"email": "medium2@test.com", "full_name": "Marcus Johnson"},
    {"email": "high1@test.com", "full_name": "James Osei"},
    {"email": "high2@test.com", "full_name": "Amara Diallo"},
]

APPLICATION_PROFILES = [
    {
        "applicant_email": "lowrisk1@test.com",
        "status": "APPROVED",
        "profile": dict(income=150000, debt=10000, credit_score=800, loan_amount=45000, tenure=48, employment_length=8.0, existing_emis=2500, dependents=1),
    },
    {
        "applicant_email": "lowrisk2@test.com",
        "status": "APPROVED",
        "profile": dict(income=120000, debt=15000, credit_score=785, loan_amount=30000, tenure=36, employment_length=6.0, existing_emis=2000, dependents=0),
    },
    {
        "applicant_email": "medium1@test.com",
        "status": "UNDER_REVIEW",
        "profile": dict(income=45000, debt=78000, credit_score=640, loan_amount=50000, tenure=60, employment_length=3.0, existing_emis=2000, dependents=2),
    },
    {
        "applicant_email": "medium2@test.com",
        "status": "DOCS_REQUIRED",
        "profile": dict(income=46000, debt=90000, credit_score=640, loan_amount=55000, tenure=60, employment_length=2.0, existing_emis=2900, dependents=2),
    },
    {
        "applicant_email": "high1@test.com",
        "status": "REJECTED",
        "profile": dict(income=45000, debt=60000, credit_score=470, loan_amount=35000, tenure=36, employment_length=2.0, existing_emis=2600, dependents=2),
    },
    {
        "applicant_email": "high2@test.com",
        "status": "REJECTED",
        "profile": dict(income=40000, debt=55000, credit_score=450, loan_amount=40000, tenure=60, employment_length=1.5, existing_emis=2300, dependents=3),
    },
]


def classify(prob):
    if prob >= 0.70:
        return "Low", "UNDER_REVIEW"
    if prob >= 0.45:
        return "Medium", "DOCS_REQUIRED"
    return "High", "REJECTED"


def get_or_create(db, email, role, full_name):
    user = db.query(User).filter(User.email == email).first()
    if user:
        print(f"  [skip] existing user '{email}'")
        return user, False
    hashed = get_password_hash(PASSWORD)
    user = User(email=email, password_hash=hashed, role=role, full_name=full_name)
    db.add(user)
    db.flush()
    print(f"  [create] user '{email}' ({role})")
    return user, True


def seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        print("Seeding default accounts...")
        demo_ids = {}
        for u in DEMO_USERS:
            user, _ = get_or_create(db, u["email"], u["role"], u["full_name"])
            demo_ids[u["email"]] = user.id
        db.commit()

        officer = db.query(User).filter(User.email == "officer@finca.com").first()

        xai_engine = XAIEngine()

        print("\nSeeding loan applications...")
        for spec in APPLICATION_PROFILES:
            applicant, _ = get_or_create(db, spec["applicant_email"], "APPLICANT",
                                         next(a["full_name"] for a in TEST_APPLICANTS if a["email"] == spec["applicant_email"]))
            db.flush()

            existing = (
                db.query(LoanApplication)
                .filter(LoanApplication.applicant_id == applicant.id)
                .order_by(LoanApplication.created_at.desc())
                .first()
            )
            if existing:
                print(f"  [skip] application already exists for '{spec['applicant_email']}'")
                db.rollback()
                continue

            p = spec["profile"]
            applicant_dict = dict(
                income=p["income"],
                debt=p["debt"],
                credit_score=p["credit_score"],
                loan_amount=p["loan_amount"],
                loan_term=p["tenure"],
                employment_length=p["employment_length"],
                existing_emis=p["existing_emis"],
                dependents=p["dependents"],
            )

            foir_data = calculate_foir_max_loan_limit(
                annual_income=p["income"],
                existing_emis=p["existing_emis"],
                tenure_months=p["tenure"],
            )
            calculated_emi = calculate_emi(principal=p["loan_amount"], tenure_months=p["tenure"])

            xai_output = xai_engine.explain_instance(applicant_dict)
            decision_info = xai_output["decision"]
            prob = decision_info["approval_probability"]
            ai_score = decision_info["finCA_credit_score"]

            target_status = spec["status"]

            if target_status == "APPROVED":
                risk_tier, computed_status = "Low", "UNDER_REVIEW"
            elif target_status == "REJECTED":
                risk_tier, computed_status = "High", "REJECTED"
            else:
                risk_tier, computed_status = classify(prob)

            if target_status != "APPROVED" and target_status != computed_status:
                risk_tier = {"UNDER_REVIEW": "Medium", "DOCS_REQUIRED": "Medium", "REJECTED": "High"}[target_status]

            rejection_reason = None
            time_to_fix_months = None
            if target_status == "REJECTED":
                top_neg = xai_output.get("top_negative_factors", [])
                if top_neg:
                    neg_descriptions = [f"{factor['feature']} (impact: {factor['shap_impact']})" for factor in top_neg]
                    rejection_reason = "Automated credit risk threshold exceeded based on: " + ", ".join(neg_descriptions)
                else:
                    rejection_reason = "High debt-to-income and credit risk indicators."
                cfs = xai_output.get("counterfactual_recommendations", [])
                if cfs and len(cfs) > 0:
                    time_to_fix_months = cfs[0].get("estimated_time_to_fix", "3-6 Months")

            new_application = LoanApplication(
                applicant_id=applicant.id,
                income=p["income"],
                credit_score=p["credit_score"],
                debt=p["debt"],
                loan_amount=p["loan_amount"],
                tenure=p["tenure"],
                employment_length=p["employment_length"],
                existing_emis=p["existing_emis"],
                dependents=p["dependents"],
                max_eligible_loan=foir_data["max_eligible_loan_amount"],
                calculated_emi=calculated_emi,
                foir_ratio=foir_data["foir_ratio"],
                status=target_status,
                ai_score=ai_score,
                approval_probability=prob,
                risk_tier=risk_tier,
                rejection_reason=rejection_reason,
                time_to_fix_months=time_to_fix_months,
                xai_breakdown_json=xai_output,
            )
            db.add(new_application)
            db.flush()

            if target_status == "DOCS_REQUIRED":
                doc = DocumentRequest(application_id=new_application.id, doc_name="Bank Statement", status="PENDING")
                db.add(doc)
                db.flush()

            audit = AuditLog(
                application_id=new_application.id,
                officer_id=officer.id if officer else None,
                action=f"DECISION_{target_status}",
                notes=f"Seeded {risk_tier} risk application. AI Score: {ai_score}, Probability: {prob:.4f}. "
                + (rejection_reason or "No adverse SHAP drivers."),
            )
            db.add(audit)
            db.flush()

            print(
                f"  [create] application #{new_application.id} for '{spec['applicant_email']}' "
                f"({risk_tier} risk, {target_status}, AI {ai_score})"
            )
            db.commit()

        print("\nSeeding summary:")
        print(f"  Demo users: {', '.join(f'{u['email']} / {PASSWORD}' for u in DEMO_USERS)}")
        total_apps = db.query(LoanApplication).count()
        total_users = db.query(User).count()
        print(f"  Total users: {total_users}")
        print(f"  Total applications: {total_apps}")
        print("Seed completed successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()