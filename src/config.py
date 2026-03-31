from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

RAW_DATA_PATH = DATA_DIR / "job.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

BASELINE_MODEL_PATH = MODELS_DIR / "baseline_salary_model.joblib"
FINAL_MODEL_PATH = MODELS_DIR / "final_salary_model.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

