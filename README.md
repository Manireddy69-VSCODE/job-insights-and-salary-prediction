# Job Salary Predictor

This project started as a simple learning exercise and slowly turned into a small end-to-end machine learning system.

I built it the way a lot of real projects actually happen:
- first by exploring the data in notebooks
- then by trying a simple baseline
- then by improving the model step by step
- and finally by wrapping the final model in an API and a small UI

The result still keeps that learning journey visible, but the code is now organized so the reusable parts live in Python files that training and deployment both share.

## What This Project Tries To Solve

Job listings often include useful information like role, location, and experience, but salary data is messy:
- some salaries are ranges
- some are undisclosed
- duplicate-style listings show up often
- not every scraped field is actually useful for prediction

The goal is to turn that messy listing data into a salary prediction pipeline that is simple, understandable, and consistent from training to deployment.

## How It Came Together

The project was built in four stages:

1. EDA:
   Understand the job market data, salary distribution, duplicates, and which fields look useful.
2. Baseline model:
   Measure how much signal we can get from a very simple model.
3. Improved model:
   Compare stronger models and choose the best one.
4. Deployment:
   Expose the final model through a FastAPI API and a small Streamlit frontend.

That step-by-step path is still there in the notebooks, but the reusable logic now lives in `src/` so the project feels like one connected system instead of a stack of disconnected experiments.

## Project Structure

```text
Jobs/
├── app/                       # FastAPI app
├── data/                      # Raw and split datasets
├── deploy/                    # Streamlit UI + Docker/deployment helpers
├── models/                    # Saved trained models + metrics
├── notebooks/                 # EDA, baseline, improvement, Kaggle walkthrough
├── src/                       # Shared preprocessing, training, and prediction logic
├── railway.toml               # Railway deployment config
├── requirements-ml.txt        # Notebook / experimentation dependencies
├── requirements.txt           # Streamlit Cloud dependencies
└── streamlit_app.py           # Root Streamlit entrypoint
```

## How The Pieces Connect

The project now works in one clear flow:

1. `notebooks/01-job-data-analysis.ipynb`
   Explains the data and the main decisions from EDA.
2. `notebooks/02-baseline-model.ipynb`
   Shows what the simplest reasonable model can do.
3. `notebooks/03-model-tuning.ipynb`
   Compares stronger models and justifies the final choice.
4. `src/training.py`
   Trains and saves the reusable model artifacts.
5. `src/predict.py`
   Loads the saved model and applies the exact same prediction pipeline at inference time.
6. `app/main.py`
   Exposes the predictor through FastAPI.
7. `deploy/streamlit_app.py`
   Calls the API and shows predictions in a simple UI.

## Final Model

The final model is an XGBoost pipeline saved in:

- [models/final_salary_model.joblib](models/final_salary_model.joblib)

It uses these features:
- `min_exp`
- `max_exp`
- `job_title`
- `location`

Why this matters:
- the training code and the API now use the same pipeline artifact
- preprocessing is no longer duplicated in notebooks and deployment code
- prediction behavior is consistent between local testing and deployed usage

Current saved metrics:
- Linear Regression baseline MAE: about `75,424.52`
- Final XGBoost pipeline MAE: about `60,498.50`

Metrics are also saved in:
- [models/model_metrics.json](models/model_metrics.json)

## Running The Project

### 1. Install dependencies

For notebooks and local ML work:

```bash
pip install -r requirements-ml.txt
```

For the API and Streamlit app:

```bash
pip install -r deploy/requirements-app.txt
```

### 2. Train the shared models

This trains the reusable pipelines and saves them into `models/`.

```bash
python -m src.training
```

Saved outputs:
- `models/baseline_salary_model.joblib`
- `models/final_salary_model.joblib`
- `models/model_metrics.json`

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Then open:
- `http://localhost:8000/docs`
- `http://localhost:8000/health`

### 4. Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

If running the UI against a deployed backend, set `api_url` in Streamlit secrets:

```toml
api_url = "https://your-backend-url.up.railway.app"
```

## API Usage

### Endpoint

`POST /predict`

### Example request

```json
{
  "min_exp": 2,
  "max_exp": 5,
  "job_title": "Data Scientist",
  "location": "Bangalore"
}
```

### Example response

```json
{
  "prediction": 633366.0,
  "confidence_range": [538361.0, 728371.0],
  "currency": "INR (approx)",
  "features_used": {
    "min_exp": 2.0,
    "max_exp": 5.0,
    "job_title": "Data Scientist",
    "location": "Bangalore"
  }
}
```

### cURL example

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"min_exp\":2,\"max_exp\":5,\"job_title\":\"Data Scientist\",\"location\":\"Bangalore\"}"
```

## About The Notebooks

The notebooks are still important because they show the thinking behind the project, not just the final answer.

- `01` is for EDA and analytical understanding
- `02` is for a true baseline
- `03` is for improvement and comparison
- `04` is a clean Kaggle-style end-to-end walkthrough

They are there to explain the journey.

The reusable project logic now lives in:
- `src/data.py`
- `src/pipeline.py`
- `src/training.py`
- `src/predict.py`

## Deployment Notes

- FastAPI lives in [app/main.py](app/main.py)
- Docker and Railway use the same API entrypoint: `app.main:app`
- Streamlit stays lightweight and talks to the API over HTTP

That keeps deployment simple:
- training happens once
- the saved model goes into `models/`
- the API loads that model
- the UI only calls the API

## Why This Version Feels More Cohesive

The project no longer feels like a few notebooks and scripts glued together at the end.

Now it has:
- one place for preprocessing logic
- one place for training logic
- one place for prediction logic
- one saved model location
- one API path that uses the same model pipeline as training

It still feels like a project built by learning and improving step by step, but now the pieces actually connect in a clean and readable way.
