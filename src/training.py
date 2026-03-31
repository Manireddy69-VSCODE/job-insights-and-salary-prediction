import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .config import BASELINE_MODEL_PATH, FINAL_MODEL_PATH, MODEL_METRICS_PATH, MODELS_DIR
from .data import load_training_splits
from .pipeline import FEATURE_COLUMNS, TARGET_COLUMN, create_baseline_pipeline, create_final_pipeline


def evaluate_model(name: str, model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = model.predict(x_test)
    return {
        "model": name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def train_project_models() -> list[dict[str, float | str]]:
    train_df, test_df = load_training_splits()
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_model = create_baseline_pipeline()
    final_model = create_final_pipeline()

    baseline_model.fit(x_train, y_train)
    final_model.fit(x_train, y_train)

    joblib.dump(baseline_model, BASELINE_MODEL_PATH)
    joblib.dump(final_model, FINAL_MODEL_PATH)

    metrics = [
        evaluate_model("Linear Regression baseline", baseline_model, x_test, y_test),
        evaluate_model("Final XGBoost pipeline", final_model, x_test, y_test),
    ]

    with MODEL_METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def main() -> None:
    metrics = train_project_models()
    for item in metrics:
        print(
            f"{item['model']}: "
            f"MAE={item['mae']:.2f}, "
            f"R2={item['r2']:.4f}"
        )
    print(f"Saved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()

