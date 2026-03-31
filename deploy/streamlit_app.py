import os

import requests
import streamlit as st


st.set_page_config(page_title="Salary Predictor", layout="wide", page_icon="💰")


def get_default_api_url() -> str:
    secret_url = st.secrets.get("api_url", "")
    env_url = os.getenv("API_URL", "")
    return (secret_url or env_url).rstrip("/")


default_api_url = get_default_api_url()
if "api_url" not in st.session_state:
    st.session_state.api_url = default_api_url

st.title("💰 Job Salary Predictor")
st.markdown(
    "A simple frontend for the same XGBoost pipeline used in training and in the API."
)

with st.sidebar:
    st.header("API Backend")
    api_url = st.text_input(
        "Backend API URL",
        value=st.session_state.api_url,
        placeholder="https://your-railway-api.up.railway.app",
        help="Paste your deployed FastAPI URL here. If you're running everything locally, use http://localhost:8000.",
    ).rstrip("/")
    st.session_state.api_url = api_url

    if api_url:
        st.info(f"FastAPI docs: {api_url}/docs")
        st.success("Backend connected and ready")
    else:
        st.warning("Add your backend URL to start making predictions.")

    st.markdown("---")
    st.header("Sample Jobs")
    samples = [
        {"job": "Data Scientist", "loc": "Bangalore", "min_exp": 2, "max_exp": 5},
        {"job": "Content Writer", "loc": "Mumbai", "min_exp": 1, "max_exp": 3},
        {"job": "Graphic Designer", "loc": "Delhi", "min_exp": 0, "max_exp": 2},
    ]
    selected = st.selectbox("Try a sample role", ["Custom"] + [sample["job"] for sample in samples])

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

with col2:
    job_title = st.text_input("Job Title", "Data Scientist", key="job_title")
    location = st.text_input("Location", "Bangalore", key="location")

if st.button("Predict Salary", type="primary"):
    if not api_url:
        st.error("I need a backend URL before I can make a prediction. Add it in the sidebar or as `api_url` in Streamlit secrets.")
        st.stop()

    payload = {
        "min_exp": min_exp,
        "max_exp": max_exp,
        "job_title": job_title,
        "location": location,
    }

    with st.spinner("Predicting..."):
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                st.success("Here’s the model’s estimate")

                col_a, col_b = st.columns([2, 1])
                with col_a:
                    pred = result["prediction"]
                    st.metric("Predicted Salary", f"INR {pred:,.0f}")
                with col_b:
                    lower, upper = result["confidence_range"]
                    st.metric("Range", f"INR {lower:,.0f} - INR {upper:,.0f}")

                st.json(result["features_used"])
            else:
                st.error(f"The API returned an error: {response.status_code}")
                st.write(response.text)
        except Exception as exc:
            st.error(f"I couldn’t reach the backend: {exc}")
            if "localhost" in api_url or "127.0.0.1" in api_url:
                st.info("`localhost` only works on your own machine. If this app is deployed on Streamlit Cloud, point `api_url` to your deployed Railway or FastAPI backend.")
            else:
                st.info("Double-check that your backend is deployed, public, and exposes the `/predict` endpoint.")

with st.expander("How to Run"):
    st.code(
        """
1. pip install -r deploy/requirements-app.txt
2. uvicorn app.main:app --reload
3. streamlit run streamlit_app.py

For Streamlit Cloud, add a secret:
api_url = "https://your-backend-url.up.railway.app"
        """
    )

st.markdown("---")
st.caption("Built from a step-by-step ML project, then connected into one shared prediction pipeline.")
