from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class SalaryPredictor:
    def __init__(self, model_path: str | Path | None = None):
        """
        Load the tuned model and derive category fallbacks from the training data.
        """
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parent

        resolved_model_path = Path(model_path) if model_path else project_root / "notebooks" / "tuned_salary_model.joblib"
        if not resolved_model_path.is_absolute():
            resolved_model_path = (base_dir / resolved_model_path).resolve()

        self.model = joblib.load(resolved_model_path)

        train_path = project_root / "data" / "train.csv"
        if train_path.exists():
            train = pd.read_csv(train_path)
            self.top_jobs = train["job_title"].value_counts().head(20).index.tolist()
            self.top_locations = train["location"].value_counts().head(20).index.tolist()
            self.target_median = float(train["avg_ctc"].median())
        else:
            self.top_jobs = [
                "Content Writer",
                "Graphic Designer",
                "Business Development Executive",
                "Sales Executive",
            ]
            self.top_locations = ["Mumbai", "Bangalore", "Delhi", "Pune"]
            self.target_median = 300000.0

        print(
            f"Model loaded from {resolved_model_path}. "
            f"Top jobs: {len(self.top_jobs)}, Top locs: {len(self.top_locations)}"
        )

    def preprocess(
        self,
        min_exp: float,
        max_exp: float,
        job_title: str,
        location: str,
    ) -> pd.DataFrame:
        """
        Apply the same feature preparation used during notebook training.
        """
        job_top = job_title if job_title in self.top_jobs else "Other"
        loc_top = location if location in self.top_locations else "Other"

        features = ["min_exp", "max_exp", "job_top", "loc_top"]
        data = pd.DataFrame([[min_exp, max_exp, job_top, loc_top]], columns=features)

        encoded = pd.get_dummies(data, columns=["job_top", "loc_top"], drop_first=True)
        train_cols = self.model.feature_names_in_
        encoded = encoded.reindex(columns=train_cols, fill_value=0)
        return encoded

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Generate a salary prediction from API or UI inputs.
        """
        try:
            min_exp = float(input_data.get("min_exp", 0))
            max_exp = float(input_data.get("max_exp", 0))
            job_title = input_data.get("job_title", "Other")
            location = input_data.get("location", "Other")

            features = self.preprocess(min_exp, max_exp, job_title, location)
            raw_pred = float(self.model.predict(features)[0])
            pred = max(raw_pred, 0.0)
            lower_bound = max(pred * 0.85, 0.0)
            upper_bound = max(pred * 1.15, pred)

            return {
                "prediction": round(pred, 0),
                "confidence_range": [round(lower_bound, 0), round(upper_bound, 0)],
                "currency": "INR (approx)",
                "features_used": {
                    "min_exp": min_exp,
                    "max_exp": max_exp,
                    "job_top": job_title[:30] + "..." if len(job_title) > 30 else job_title,
                    "loc_top": location[:30] + "..." if len(location) > 30 else location,
                },
            }
        except Exception as exc:
            return {"error": str(exc), "status": "prediction failed"}
