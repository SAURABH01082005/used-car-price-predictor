"""
Data cleaning and the sklearn preprocessing pipeline.

Design notes (for viva):
- Unit-stripping and dedup/outlier handling happen in pandas BEFORE the
  train/test split, because they do not learn any statistic from the data
  (they are deterministic string/row transforms) -- this is not leakage.
- Imputation, scaling and one-hot encoding are learned with
  ColumnTransformer inside a sklearn Pipeline that is fit ONLY on the
  training split (see src/train.py). This prevents test-set statistics
  (e.g. a column mean) from leaking into training.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, UNIT_COLUMNS

logger = logging.getLogger(__name__)


def _extract_numeric(series: pd.Series, pattern: str) -> pd.Series:
    """Extract the first numeric token from a text column, e.g. '1248 CC' -> 1248.0.

    Rows with no numeric token (e.g. an empty/garbled value) become NaN and
    are handled later by SimpleImputer -- they are not silently dropped.
    """
    extracted = series.astype(str).str.extract(pattern, expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw CarDekho DataFrame into analysis-ready form.

    Steps (each documented so preprocessing decisions are auditable):
    1. Drop exact duplicate rows (same listing scraped twice).
    2. Strip units from text-numeric columns: mileage ("23.4 kmpl" -> 23.4),
       engine ("1248 CC" -> 1248), max_power ("74 bhp" -> 74).
    3. Drop rows with a missing or non-positive selling_price (the target)
       -- a row with no valid target cannot be used for supervised learning
       and cannot be safely imputed.
    4. Drop rows with non-positive km_driven or a manufacturing year outside
       a plausible range (data-entry errors), which is a tiny fraction of
       rows for this dataset.

    Missing values in FEATURE columns (mileage/engine/max_power/seats) are
    NOT dropped here -- they are left as NaN and handled by SimpleImputer
    inside the sklearn Pipeline so the imputation statistic is learned only
    on the training fold.
    """
    df = df.copy()
    n_before = len(df)

    df = df.drop_duplicates()
    logger.info("Dropped %d exact duplicate rows", n_before - len(df))

    for col, pattern in UNIT_COLUMNS.items():
        df[col] = _extract_numeric(df[col], pattern)

    # torque is highly inconsistent in free-text format (units, RPM ranges,
    # different notations) across rows; it is not used as a model feature,
    # so it is dropped rather than partially parsed.
    if "torque" in df.columns:
        df = df.drop(columns=["torque"])

    n_before_target = len(df)
    df = df[df["selling_price"].notna() & (df["selling_price"] > 0)]
    logger.info("Dropped %d rows with missing/non-positive selling_price", n_before_target - len(df))

    n_before_sanity = len(df)
    df = df[(df["km_driven"] > 0) & (df["km_driven"] < 1_000_000)]
    df = df[(df["year"] >= 1980) & (df["year"] <= 2024)]
    logger.info("Dropped %d rows failing basic sanity checks (km_driven/year range)", n_before_sanity - len(df))

    df = df.reset_index(drop=True)
    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the ColumnTransformer used inside the modelling Pipeline.

    Numerical features: median-impute (robust to outliers) then standardize.
    Categorical features: most-frequent-impute then one-hot encode, with
    unknown categories at prediction time mapped to all-zero (handle_unknown
    ="ignore") instead of raising, since a real user may type a brand not
    seen during training.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor
