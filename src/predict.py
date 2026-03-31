from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import FINAL_MODEL_PATH
from .pipeline import FEATURE_COLUMNS


class SalaryPredictor:
    def __init__(self, model_path: str | Path | None = None):
        base_dir = Path(__file__).resolve().parents[1]
        resolved_model_path = Path(model_path) if model_path else FINAL_MODEL_PATH
        if not resolved_model_path.is_absolute():
            resolved_model_path = (base_dir / resolved_model_path).resolve()

        self.model = joblib.load(resolved_model_path)
        self.model_path = resolved_model_path

    def _build_features(self, input_data: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        min_exp = float(input_data.get("min_exp", 0))
        max_exp = float(input_data.get("max_exp", 0))
        job_title = str(input_data.get("job_title", "Unknown")).strip() or "Unknown"
        location = str(input_data.get("location", "Unknown")).strip() or "Unknown"

        features = pd.DataFrame([[min_exp, max_exp, job_title, location]], columns=FEATURE_COLUMNS)
        clean_inputs = {
            "min_exp": min_exp,
            "max_exp": max_exp,
            "job_title": job_title,
            "location": location,
        }
        return features, clean_inputs

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        features, clean_inputs = self._build_features(input_data)
        raw_prediction = float(self.model.predict(features)[0])

        prediction = max(raw_prediction, 0.0)
        lower_bound = max(prediction * 0.85, 0.0)
        upper_bound = max(prediction * 1.15, prediction)

        return {
            "prediction": round(prediction, 0),
            "confidence_range": [round(lower_bound, 0), round(upper_bound, 0)],
            "currency": "INR (approx)",
            "features_used": clean_inputs,
        }

