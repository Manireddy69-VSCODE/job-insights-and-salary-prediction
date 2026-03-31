# Changelog

## 2026-03

### Project cleanup and consolidation
- reorganized the project so the reusable logic now lives in `src/`
- added `app/` for the FastAPI entrypoint and kept `deploy/` focused on deployment helpers and the Streamlit UI
- added `models/` as the single place for saved model artifacts and evaluation metrics
- trained and saved:
  - `models/baseline_salary_model.joblib`
  - `models/final_salary_model.joblib`
  - `models/model_metrics.json`

### Pipeline consistency
- moved shared preprocessing, training, and inference logic into:
  - `src/data.py`
  - `src/pipeline.py`
  - `src/training.py`
  - `src/predict.py`
- updated the API to use the same saved prediction pipeline as training
- removed the old mismatch where notebook-era logic and deployed prediction logic had drifted apart

### Notebook cleanup
- organized the notebooks into a cleaner flow:
  - `01` EDA
  - `02` baseline
  - `03` model improvement
  - `04` polished Kaggle notebook
- rewrote the notebook narration to feel more natural and easier to follow

### Deployment updates
- switched Docker and Railway to the shared API entrypoint: `app.main:app`
- kept the Streamlit app lightweight and connected it to the same backend flow

### Current saved metrics
- Linear Regression baseline: MAE `75,424.52`, R² `0.4602`
- Final XGBoost pipeline: MAE `60,498.50`, R² `0.6536`
