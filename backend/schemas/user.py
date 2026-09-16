import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: Optional[str] = "APPLICANT"  # 'APPLICANT' or 'LOAN_OFFICER'


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str] = None
    role: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


Token.model_rebuild()
