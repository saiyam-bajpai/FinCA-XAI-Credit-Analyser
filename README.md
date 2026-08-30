# FinCA (Financial Credit Analyser)

An AI-driven digital loan origination and explainable credit scoring web platform.

## Architecture
- **Backend:** Python, FastAPI, SQLAlchemy, PostgreSQL
- **Data/ML:** Scikit-Learn, XGBoost, SHAP
- **Frontend:** Next.js (App Router), Tailwind CSS, Recharts

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Node.js (v18+)
- Python (v3.10+)

### Quick Start with Docker
1. Copy `.env.example` to `.env` and configure your environment variables.
   ```bash
   cp .env.example .env
   ```
2. Run the application stack using Docker Compose:
   ```bash
   docker-compose up --build
   ```
3. Access the frontend at `http://localhost:3000` and the backend API docs at `http://localhost:8000/docs`.

### Local Development Setup

#### Backend
1. Navigate to the `backend` directory.
   ```bash
   cd backend
   ```
2. Create a virtual environment and install dependencies.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Start the FastAPI development server.
   ```bash
   uvicorn main:app --reload
   ```

#### Frontend
1. Navigate to the `frontend` directory.
   ```bash
   cd frontend
   ```
2. Install dependencies.
   ```bash
   npm install
   ```
3. Start the development server.
   ```bash
   npm run dev
   ```
