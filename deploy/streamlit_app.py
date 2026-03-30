import requests
import streamlit as st


st.set_page_config(page_title="Salary Predictor", layout="wide", page_icon="💰")

API_URL = st.secrets.get("api_url", "http://localhost:8000")

st.title("💰 Job Salary Predictor")
st.markdown("**Powered by the tuned XGBoost model** from `notebooks/03-model-tuning.ipynb`")

with st.sidebar:
    st.header("API Backend")
    st.info(f"FastAPI docs: {API_URL}/docs")
    st.success("Model UI ready")
    st.markdown("---")
    st.header("Sample Jobs")
    samples = [
        {"job": "Data Scientist", "loc": "Bangalore", "min_exp": 2, "max_exp": 5},
        {"job": "Content Writer", "loc": "Mumbai", "min_exp": 1, "max_exp": 3},
        {"job": "Graphic Designer", "loc": "Delhi", "min_exp": 0, "max_exp": 2},
    ]
    selected = st.selectbox("Load sample", ["Custom"] + [sample["job"] for sample in samples])

    if selected != "Custom":
        sample = next(sample for sample in samples if sample["job"] == selected)
        st.session_state.min_exp = sample["min_exp"]
        st.session_state.max_exp = sample["max_exp"]
        st.session_state.job_title = sample["job"]
        st.session_state.location = sample["loc"]

col1, col2 = st.columns(2)
with col1:
    min_exp = st.number_input("Min Experience (years)", 0.0, 20.0, 2.0, key="min_exp")
    max_exp = st.number_input("Max Experience (years)", 0.0, 20.0, 5.0, key="max_exp")
    posted_days = st.slider("Posted Days Ago", 0, 90, 7)

with col2:
    job_title = st.text_input("Job Title", "Data Scientist", key="job_title")
    location = st.text_input("Location", "Bangalore", key="location")

if st.button("Predict Salary", type="primary"):
    payload = {
        "min_exp": min_exp,
        "max_exp": max_exp,
        "posted_days": posted_days,
        "job_title": job_title,
        "location": location,
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                st.success("Prediction ready")

                col_a, col_b = st.columns([2, 1])
                with col_a:
                    pred = result["prediction"]
                    st.metric("Predicted Salary", f"INR {pred:,.0f}")
                with col_b:
                    lower, upper = result["confidence_range"]
                    st.metric("Range", f"INR {lower:,.0f} - INR {upper:,.0f}")

                st.json(result["features_used"])
            else:
                st.error(f"API error: {response.status_code}")
                st.write(response.text)
        except Exception as exc:
            st.error(f"Backend not running? {exc}")
            st.info("Run: `uvicorn deploy.app:app --reload`")

with st.expander("How to Run"):
    st.code(
        """
1. pip install -r deploy/requirements-app.txt
2. uvicorn deploy.app:app --reload
3. streamlit run deploy/streamlit_app.py
        """
    )

st.markdown("---")
st.caption("Built with your tuned XGBoost model")
