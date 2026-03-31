import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RAW_DATA_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH


TARGET_COLUMN = "avg_ctc"
DUPLICATE_KEY = ["job_title", "company_name", "location"]


def parse_salary(text: str) -> pd.Series:
    text = str(text)
    if "competitive salary" in text.lower():
        return pd.Series([pd.NA, pd.NA, False], index=["min_ctc", "max_ctc", "is_salary_disclosed"])

    nums = re.findall(r"[\d,]+", text)
    nums = [int(num.replace(",", "")) for num in nums]
    if not nums:
        return pd.Series([pd.NA, pd.NA, False], index=["min_ctc", "max_ctc", "is_salary_disclosed"])
    if len(nums) == 1:
        return pd.Series([nums[0], nums[0], True], index=["min_ctc", "max_ctc", "is_salary_disclosed"])
    return pd.Series([nums[0], nums[1], True], index=["min_ctc", "max_ctc", "is_salary_disclosed"])


def parse_posted_days(value: object) -> int | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().lower().replace("\n\n\nbe an early applicant", "")
    if "today" in text or "few hours" in text or "just now" in text or "immediately" in text:
        return 0

    match = re.search(r"(\d+)", text)
    number = int(match.group(1)) if match else None
    if "week" in text:
        return (number or 1) * 7
    if "month" in text:
        return (number or 1) * 30
    return number if number is not None else pd.NA


def build_modeling_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    salary_parts = raw_df["ctc"].apply(parse_salary)
    df = pd.concat([raw_df.copy(), salary_parts], axis=1)

    df["avg_ctc"] = (
        pd.to_numeric(df["min_ctc"], errors="coerce")
        .add(pd.to_numeric(df["max_ctc"], errors="coerce"))
        .div(2)
    )
    df["min_exp"] = pd.to_numeric(df["experience"].str.extract(r"(\d+)")[0], errors="coerce")
    df["max_exp"] = (
        pd.to_numeric(df["experience"].str.extract(r"(?:\d+)-(\d+)")[0], errors="coerce")
        .fillna(df["min_exp"])
    )
    df["posted_days"] = df["posted"].apply(parse_posted_days)
    df["is_remote"] = df["location"].str.contains("work from home|remote", case=False, na=False)
    df["is_noisy_duplicate"] = df.duplicated(subset=DUPLICATE_KEY, keep="first")

    return df.loc[~df["is_noisy_duplicate"]].copy()


def load_raw_dataset(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_modeling_dataset(
    path: str | Path = RAW_DATA_PATH,
    *,
    require_target: bool = False,
) -> pd.DataFrame:
    model_df = build_modeling_frame(load_raw_dataset(path))
    if require_target:
        model_df = model_df.loc[model_df[TARGET_COLUMN].notna()].copy()
    return model_df


def load_training_splits(
    train_path: str | Path = TRAIN_DATA_PATH,
    test_path: str | Path = TEST_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = Path(train_path)
    test_path = Path(test_path)

    if train_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    else:
        model_df = load_modeling_dataset(require_target=True)
        train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)

    train_df = train_df.loc[train_df[TARGET_COLUMN].notna()].copy()
    test_df = test_df.loc[test_df[TARGET_COLUMN].notna()].copy()
    return train_df, test_df

