import datetime
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field


class LoanApplicationCreate(BaseModel):
    income: float = Field(..., description="Annual income in USD", example=65000.0)
    credit_score: int = Field(..., description="Credit score between 300 and 850", example=710)
    debt: float = Field(..., description="Total existing debt in USD", example=15000.0)
    loan_amount: float = Field(..., description="Requested loan amount in USD", example=25000.0)
    tenure: int = Field(..., description="Loan term in months", example=36)
    employment_length: float = Field(0.0, description="Employment length in years", example=4.5)
    existing_emis: float = Field(0.0, description="Existing monthly EMIs in USD", example=450.0)
    dependents: int = Field(0, description="Number of dependents", example=1)


class OfficerDecisionRequest(BaseModel):
    status: str = Field(..., description="Decision: 'APPROVED', 'REJECTED', or 'DOCS_REQUIRED'")
    officer_notes: Optional[str] = Field(None, description="Notes explaining officer decision")
    required_documents: Optional[List[str]] = Field(None, description="List of documents requested, e.g. ['Bank Statement', 'Tax Return']")


class DocumentRequestResponse(BaseModel):
    id: int
    doc_name: str
    status: str
    requested_at: datetime.datetime

    class Config:
        from_attributes = True


class AuditLogResponse(BaseModel):
    id: int
    officer_id: Optional[int] = None
    action: str
    notes: Optional[str] = None
    timestamp: datetime.datetime

    class Config:
        from_attributes = True


class LoanApplicationResponse(BaseModel):
    id: int
    applicant_id: int
    income: float
    credit_score: int
    debt: float
    loan_amount: float
    tenure: int
    employment_length: float
    existing_emis: float
    dependents: int
    
    max_eligible_loan: Optional[float] = None
    calculated_emi: Optional[float] = None
    foir_ratio: Optional[float] = None
    
    status: str
    ai_score: Optional[int] = None
    approval_probability: Optional[float] = None
    risk_tier: Optional[str] = None
    rejection_reason: Optional[str] = None
    time_to_fix_months: Optional[str] = None

    xai_breakdown_json: Optional[Dict[str, Any]] = None
    
    document_requests: List[DocumentRequestResponse] = []
    audit_logs: List[AuditLogResponse] = []
    
    created_at: datetime.datetime

    class Config:
        from_attributes = True
