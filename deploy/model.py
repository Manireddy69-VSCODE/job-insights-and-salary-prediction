import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path

class SalaryPredictor:
    def __init__(self, model_path: str = '../notebooks/tuned_salary_model.joblib'):
        """
        Load tuned XGBoost model and preprocess logic matching notebook/03-model-tuning.ipynb
        """
        self.model = joblib.load(model_path)
        
        # Load train data to extract TOP 20 jobs/locations (exact match to notebook)
        train_path = Path('../data/train.csv')
        if train_path.exists():
            train = pd.read_csv(train_path)
            self.top_jobs = train['job_title'].value_counts().head(20).index.tolist()
            self.top_locations = train['location'].value_counts().head(20).index.tolist()
            self.target_median = train['avg_ctc'].median()
        else:
            # Fallback top lists from notebook analysis
            self.top_jobs = ['Content Writer', 'Graphic Designer', 'Business Development Executive', 'Sales Executive']  # Add all 20 from analysis
            self.top_locations = ['Mumbai', 'Bangalore', 'Delhi', 'Pune']
            self.target_median = 300000.0
        
        print(f"✅ Model loaded. Top jobs: {len(self.top_jobs)}, Top locs: {len(self.top_locations)}")

    def preprocess(self, min_exp: float, max_exp: float, posted_days: float, 
                   job_title: str, location: str) -> pd.DataFrame:
        """
        Exact preprocessing from 03-model-tuning.ipynb
        """
        # Create categoricals
        job_top = job_title if job_title in self.top_jobs else 'Other'
        loc_top = location if location in self.top_locations else 'Other'
        
        # Features list matching notebook
        features = ['min_exp', 'max_exp', 'posted_days', 'job_top', 'loc_top']
        data = pd.DataFrame([[min_exp, max_exp, posted_days, job_top, loc_top]], columns=features)
        
        # One-hot encode (drop_first=True like notebook)
        X = pd.get_dummies(data, columns=['job_top', 'loc_top'], drop_first=True)
        
        # Align columns to training (fill missing with 0)
        train_cols = self.model.feature_names_in_
        X = X.reindex(columns=train_cols, fill_value=0)
        
        return X

    def predict(self, input_data: Dict) -> Dict:
        """
        Main prediction endpoint logic
        """
        try:
            # Extract inputs
            min_exp = float(input_data.get('min_exp', 0))
            max_exp = float(input_data.get('max_exp', 0))
            posted_days = float(input_data.get('posted_days', 0))
            job_title = input_data.get('job_title', 'Other')
            location = input_data.get('location', 'Other')
            
            # Preprocess
            X = self.preprocess(min_exp, max_exp, posted_days, job_title, location)
            
            # Predict
            pred = self.model.predict(X)[0]
            
            # Confidence interval (XGBoost quantile or std)
            pred_lower = self.model.predict(X)[0] * 0.85  # Conservative 15% range
            pred_upper = self.model.predict(X)[0] * 1.15
            
            return {
                "prediction": round(float(pred), 0),
                "confidence_range": [round(pred_lower, 0), round(pred_upper, 0)],
                "currency": "₹ Lakhs (approx)",
                "features_used": {
                    "min_exp": min_exp,
                    "max_exp": max_exp,
                    "posted_days": posted_days,
                    "job_top": job_title[:30] + '...' if len(job_title) > 30 else job_title,
                    "loc_top": location[:30] + '...' if len(location) > 30 else location
                }
            }
        except Exception as e:
            return {"error": str(e), "status": "prediction failed"}
