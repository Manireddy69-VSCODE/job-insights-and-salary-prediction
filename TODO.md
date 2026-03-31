# Project Notes

## Current state
- notebooks are organized and cleaned up for explanation
- shared project logic lives in `src/`
- trained artifacts live in `models/`
- FastAPI uses the same prediction pipeline as training
- Streamlit calls the API instead of reimplementing model logic

## Good next steps
- add a small pytest suite around `src/data.py`, `src/predict.py`, and `app/main.py`
- add a simple CI workflow to run linting and smoke tests
- deploy the updated API and Streamlit app together after pushing the refactor
- remove any old local-only files that are no longer useful once the new flow is stable

## Handy commands
```bash
# train shared models
python -m src.training

# run API
uvicorn app.main:app --reload

# run Streamlit UI
streamlit run streamlit_app.py
```
