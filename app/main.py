import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import SalaryPredictor


app = FastAPI(
    title="Salary Predictor API",
    description="A small FastAPI service that serves the same salary prediction pipeline used during training.",
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
    min_exp: float = Field(..., ge=0, description="Minimum years of experience for the role")
    max_exp: float = Field(..., ge=0, description="Maximum years of experience for the role")
    posted_days: float | None = Field(
        None,
        ge=0,
        description="Older field kept only for compatibility. The final model does not use it anymore.",
    )
    job_title: str = Field(..., max_length=100, description="Job title from the listing")
    location: str = Field(..., max_length=100, description="Location from the listing")


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "message": "Salary Predictor API is up and ready.",
        "model": "final_salary_model.joblib",
        "endpoints": ["/docs", "/predict", "/health"],
        "features": ["min_exp", "max_exp", "job_title", "location"],
    }


@app.post("/predict", response_model=dict)
async def predict_salary(input_data: SalaryInput) -> dict[str, object]:
    try:
        return predictor.predict(input_data.model_dump())
    except Exception as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=f"Something went wrong while generating the prediction: {exc}") from exc


@app.get("/health")
async def health_check() -> dict[str, object]:
    return {"status": "healthy", "model_loaded": True, "model_path": str(predictor.model_path)}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
