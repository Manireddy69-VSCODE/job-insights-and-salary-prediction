import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .model import SalaryPredictor
except ImportError:
    from model import SalaryPredictor


app = FastAPI(
    title="Salary Predictor API",
    description="Deployed XGBoost model for job salary prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = SalaryPredictor()


class SalaryInput(BaseModel):
    min_exp: float = Field(..., ge=0, description="Minimum years of experience")
    max_exp: float = Field(..., ge=0, description="Maximum years of experience")
    posted_days: float = Field(..., ge=0, description="Days since job posted")
    job_title: str = Field(..., max_length=100, description="Job title")
    location: str = Field(..., max_length=100, description="Job location")


@app.get("/")
async def root():
    return {
        "message": "Salary Predictor API",
        "model": "XGBoost (tuned_salary_model.joblib)",
        "endpoints": ["/docs", "/predict", "/health"],
        "example": "POST /predict with JSON input",
    }


@app.post("/predict", response_model=dict)
async def predict_salary(input_data: SalaryInput):
    """
    Predict salary from job features.
    """
    try:
        return predictor.predict(input_data.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
