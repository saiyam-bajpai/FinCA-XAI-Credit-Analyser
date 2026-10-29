import datetime
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.db.database import get_db
from backend.db.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def _truncate_password(password: str) -> str:
    if isinstance(password, str):
        # Truncate raw password to 72 bytes for bcrypt compatibility
        return password.encode('utf-8')[:72].decode('utf-8', errors='ignore')
    return password


def verify_password(plain_password: str, hashed_password: str) -> bool:
    safe_password = _truncate_password(plain_password)
    return pwd_context.verify(safe_password, hashed_password)


def get_password_hash(password: str) -> str:
    safe_password = _truncate_password(password)
    return pwd_context.hash(safe_password)


def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


from fastapi import Depends, HTTPException, Query, status

oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def decode_token_to_user(token: str, db: Session) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    return decode_token_to_user(token, db)


def get_user_from_token_or_query(
    token_query: Optional[str] = Query(None, alias="token"),
    auth_header: Optional[str] = Depends(oauth2_scheme_optional),
    db: Session = Depends(get_db),
) -> User:
    resolved = auth_header or token_query
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required (via Bearer header or ?token= parameter)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token_to_user(resolved, db)


def require_role(required_role: str):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation restricted to users with '{required_role}' role."
            )
        return current_user
    return role_checker


def require_role_token_or_query(required_role: str):
    def role_checker(current_user: User = Depends(get_user_from_token_or_query)):
        if current_user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation restricted to users with '{required_role}' role."
            )
        return current_user
    return role_checker
