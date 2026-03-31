# Job Salary Predictor 💰
## XGBoost ML Model + Production API Deployment

**✅ TUNED MODEL**: XGBoostRegressor (`tuned_salary_model.joblib`) beats baseline by 5%+ MAE  
**Features**: min_exp, max_exp, posted_days, top job_title/location (one-hot)  
**Data**: 3.6K+ scraped job postings (`data/train.csv`, `test.csv`)

### 🚀 Production Deployment (COMPLETE)
```
cd deploy
pip install -r requirements-app.txt

# API (Swagger docs)
uvicorn app:app --reload
➜ http://localhost:8000/docs

# UI Demo  
streamlit run ../streamlit_app.py  
➜ http://localhost:8501

# Docker (prod)
docker-compose up -d
```

**API Endpoint**: `POST /predict`
```json
{
  "min_exp": 2, "max_exp": 5, "posted_days": 7,
  "job_title": "Data Scientist", "location": "Bangalore"
}
```
**Response**: `{"prediction": 850000, "confidence_range": [722500, 977500]}`

### 📁 Structure
```
Jobs/
├── notebooks/02-baseline-model.ipynb      # Training
├── notebooks/03-model-tuning.ipynb        # ✅ Tuned model 
├── notebooks/tuned_salary_model.joblib    # ✅ PROD MODEL
├── deploy/                               # NEW: Production deployment
│   ├── app.py                 # FastAPI REST API
│   ├── model.py               # Load + preprocess  
│   ├── streamlit_app.py       # UI Frontend
│   ├── Dockerfile             # Containerize
│   └── docker-compose.yml
└── data/                         # Job postings
```

### 📖 Changelog
See [CHANGELOG.md](CHANGELOG.md) for deployment summary and next steps.

### 🧪 Postman API Testing (Recommended)
1. **Start API Backend**:
   ```
   cd deploy
   pip install -r requirements-app.txt
   uvicorn app:app --reload
   ```
   → Open http://localhost:8000/docs (Swagger).

2. **Postman Setup**:
   - New Request: `POST http://localhost:8000/predict`
   - Headers: `Content-Type: application/json`
   - Body (JSON):
     ```json
     {
       "min_exp": 2,
       "max_exp": 5,
       "posted_days": 7,
       "job_title": "Data Scientist",
       "location": "Bangalore"
     }
     ```
   - Send → Expect: `{"prediction": 850000, ...}`

   **Import cURL directly**:
   ```
   curl -X POST "http://localhost:8000/predict" \
   -H "Content-Type: application/json" \
   -d "{\"min_exp\":2,\"max_exp\":5,\"posted_days\":7,\"job_title\":\"Data Scientist\",\"location\":\"Bangalore\"}"
   ```

### 🎯 Next: Cloud Deploy
See [TODO.md](TODO.md) and [CHANGELOG.md](CHANGELOG.md).

### Streamlit Cloud
Deploy the root app file: `streamlit_app.py`

Dependencies should be read from the root `requirements.txt`.

If the builder still fails, open Advanced settings and set Python to `3.12`.

---

