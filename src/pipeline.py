from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


TARGET_COLUMN = "avg_ctc"
FEATURE_COLUMNS = ["min_exp", "max_exp", "job_title", "location"]
NUMERIC_FEATURES = ["min_exp", "max_exp"]
CATEGORICAL_FEATURES = ["job_title", "location"]

FINAL_MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "n_estimators": 200,
    "max_depth": 8,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 1,
}


def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def create_baseline_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", create_preprocessor()),
            ("model", LinearRegression()),
        ]
    )


def create_final_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", create_preprocessor()),
            ("model", XGBRegressor(**FINAL_MODEL_PARAMS)),
        ]
    )

