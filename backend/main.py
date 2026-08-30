from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.api import applications, auth, officer
from backend.core.security import get_password_hash
from backend.db.database import Base, SessionLocal, engine
from backend.db.models import User


def seed_demo_users():
    """
    Seeds initial demo users (Officer and Applicant) into the database if missing.
    """
    db: Session = SessionLocal()
    try:
        officer = db.query(User).filter(User.email == "officer@finca.com").first()
        if not officer:
            officer_user = User(
                email="officer@finca.com",
                password_hash=get_password_hash("password123"),
                full_name="Chief Risk Officer",
                role="LOAN_OFFICER"
            )
            db.add(officer_user)

        applicant = db.query(User).filter(User.email == "applicant@finca.com").first()
        if not applicant:
            applicant_user = User(
                email="applicant@finca.com",
                password_hash=get_password_hash("password123"),
                full_name="John Applicant",
                role="APPLICANT"
            )
            db.add(applicant_user)

        db.commit()
        print("Demo users checked/seeded successfully.")
    except Exception as e:
        print(f"Error seeding demo users: {e}")
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create tables on startup
    Base.metadata.create_all(bind=engine)
    seed_demo_users()
    yield


app = FastAPI(
    title="FinCA API",
    description="Financial Credit Analyser API for digital loan origination and explainable credit scoring.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration allowing frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth.router)
app.include_router(applications.router)
app.include_router(officer.router)


@app.get("/")
def read_root():
    return {
        "app": "FinCA API",
        "status": "online",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
