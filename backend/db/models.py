import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from backend.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="APPLICANT")  # 'APPLICANT' or 'LOAN_OFFICER'
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    applications = relationship("LoanApplication", back_populates="applicant")
    audit_logs = relationship("AuditLog", back_populates="officer")


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Financial parameters
    income = Column(Float, nullable=False)
    credit_score = Column(Integer, nullable=False)
    debt = Column(Float, nullable=False)
    loan_amount = Column(Float, nullable=False)
    tenure = Column(Integer, nullable=False)  # in months
    employment_length = Column(Float, default=0.0)
    existing_emis = Column(Float, default=0.0)
    dependents = Column(Integer, default=0)

    # Calculated Financial metrics
    max_eligible_loan = Column(Float, nullable=True)
    calculated_emi = Column(Float, nullable=True)
    foir_ratio = Column(Float, nullable=True)

    # Status & AI scoring
    status = Column(String, default="SUBMITTED")  # SUBMITTED, DOCS_REQUIRED, UNDER_REVIEW, APPROVED, REJECTED
    ai_score = Column(Integer, nullable=True)  # 300 - 850 FinCA score
    approval_probability = Column(Float, nullable=True)
    risk_tier = Column(String, default="Medium")  # Low, Medium, High
    rejection_reason = Column(Text, nullable=True)
    time_to_fix_months = Column(String, nullable=True)

    # SHAP & Recourse Payload
    xai_breakdown_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    # Relationships
    applicant = relationship("User", back_populates="applications")
    document_requests = relationship("DocumentRequest", back_populates="application", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="application", cascade="all, delete-orphan")


class DocumentRequest(Base):
    __tablename__ = "document_requests"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("loan_applications.id"), nullable=False)
    doc_name = Column(String, nullable=False)  # 'Bank Statement', 'Tax Return', 'Salary Slip', etc.
    status = Column(String, default="PENDING")  # PENDING, UPLOADED, VERIFIED
    requested_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    application = relationship("LoanApplication", back_populates="document_requests")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("loan_applications.id"), nullable=False)
    officer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)  # e.g., 'SUBMITTED', 'DECISION_APPROVED', 'DOCS_REQUESTED'
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    application = relationship("LoanApplication", back_populates="audit_logs")
    officer = relationship("User", back_populates="audit_logs")
