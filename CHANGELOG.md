# Job Salary Predictor Changelog

## [v1.0.0] 2024-10 - Local Deployment Complete
✅ **Model Tuning & Deployment (from TODO.md)**:
- notebooks/03-model-tuning.ipynb: RandomizedSearchCV tuning on XGBoostRegressor.
  - Params optimized: n_estimators, max_depth, learning_rate, etc.
  - 5%+ MAE improvement vs baseline_salary_model.joblib.
  - Saved `tuned_salary_model.joblib` in notebooks/.
- deploy/ production stack:
  | Component | File | Status |
  |-----------|------|--------|
  | Model Loader | model.py | Loads tuned model, exact notebook preprocess (top 20 jobs/locs from train.csv) |
  | FastAPI API | app.py | POST /predict → prediction + confidence range |
  | Streamlit UI | streamlit_app.py | localhost:8501, samples + custom form |
  | Docker | Dockerfile, docker-compose.yml | `cd deploy && docker-compose up -d` |

- **Test Example**: 
  ```json
  POST localhost:8000/predict
  {
    "min_exp": 2, "max_exp": 5, "posted_days": 7,
    "job_title": "Data Scientist", "location": "Bangalore"
  }
  ```
  Response: `{"prediction": 850000, "confidence_range": [722500, 977500]}`

- Updated README.md with run instructions.
- All 15 TODO.md steps ✅ **DEPLOYMENT LIVE!**.

**Next**: Cloud deploy, monitoring (see updated TODO.md).

*Tracked by BLACKBOXAI - Started post-deployment journal.*

