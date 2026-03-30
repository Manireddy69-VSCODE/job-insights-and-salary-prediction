import streamlit as st
import requests
import pandas as pd
from typing import Dict

# Page config
st.set_page_config(page_title="Salary Predictor", layout="wide", page_icon="💰")

st.title("💰 Job Salary Predictor")
st.markdown("**Powered by tuned XGBoost model** from notebooks/03-model-tuning.ipynb")

# Sidebar info
with st.sidebar:
    st.header("🚀 API Backend")
    st.info("FastAPI: http://localhost:8000/docs")
    st.success("✅ Model loaded & ready")
    st.markdown("---")
    st.header("📊 Sample Jobs")
    samples = [
        {"job": "Data Scientist", "loc": "Bangalore", "min_exp": 2, "max_exp": 5},
        {"job": "Content Writer", "loc": "Mumbai", "min_exp": 1, "max_exp": 3},
        {"job": "Graphic Designer", "loc": "Delhi", "min_exp": 0, "max_exp": 2},
    ]
    selected = st.selectbox("Load sample", ["Custom"] + [f"{s['job']}" for s in samples])
    
    if selected != "Custom":
        sample = next(s for s in samples if s['job'] == selected)
        request.session_state.min_exp = sample["min_exp"]
        request.session_state.max_exp = sample["max_exp"]
        request.session_state.job_title = sample["job"]
        request.session_state.location = sample["loc"]

# Input form
col1, col2 = st.columns(2)
with col1:
    min_exp = st.number_input("Min Experience (years)", 0.0, 20.0, 2.0, key="min_exp")
    max_exp = st.number_input("Max Experience (years)", 0.0, 20.0, 5.0, key="max_exp")
    posted_days = st.slider("Posted Days Ago", 0, 90, 7)
    
with col2:
    job_title = st.text_input("Job Title", "Data Scientist", key="job_title")
    location = st.text_input("Location", "Bangalore", key="location")

if st.button("🔮 Predict Salary", type="primary"):
    # Call FastAPI backend
    payload = {
        "min_exp": min_exp,
        "max_exp": max_exp, 
        "posted_days": posted_days,
        "job_title": job_title,
        "location": location
    }
    
    with st.spinner("Predicting..."):
        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Prediction Ready!")
                
                # Results
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    pred = result["prediction"]
                    st.metric("Predicted Salary", f"₹{pred:,}")
                with col_b:
                    lower, upper = result["confidence_range"]
                    st.metric("Range", f"₹{lower:,} - ₹{upper:,}")
                
                st.json(result["features_used"])
                
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Backend not running? {str(e)}")
            st.info("💡 Run: `uvicorn deploy.app:app --reload`")

# Instructions
with st.expander("📖 How to Run"):
    st.code("""
1. pip install -r deploy/requirements-app.txt
2. uvicorn deploy.app:app --reload    # API: localhost:8000/docs  
3. streamlit run deploy/streamlit_app.py  # UI: localhost:8501
    """)
    
st.markdown("---")
st.caption("Built with ❤️ using your tuned XGBoost model")
