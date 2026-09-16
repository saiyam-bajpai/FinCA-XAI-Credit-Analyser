import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "FinCA API"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "finca_secret_key_change_me_in_production_12345")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./finca.db")

    class Config:
        case_sensitive = True


settings = Settings()
