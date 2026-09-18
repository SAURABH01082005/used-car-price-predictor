"""
Central configuration for the Used Car Price Prediction project.

Every path, constant, and column-name assumption used elsewhere in the
codebase is defined here so the project can be adapted to a different
dataset (or re-run reproducibly) by editing a single file.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "used_cars.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "used_car_price_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
# Source dataset: "Car details v3.csv" from the public Kaggle dataset
# "Vehicle dataset" by Nehal Birla (nehalbirla/vehicle-dataset-from-cardekho),
# itself scraped from used-car listings on CarDekho.com. Retrieved for this
# project via a public GitHub mirror of the Kaggle files (Kaggle downloads
# require an authenticated account, which is not available in this build
# environment). Students should verify/re-download the original file from
# Kaggle directly for submission if their institution requires a primary
# source link: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
#
# Raw columns actually present in data/raw/used_cars.csv:
#   name, year, selling_price, km_driven, fuel, seller_type, transmission,
#   owner, mileage, engine, max_power, torque, seats
#
# NOTE: this dataset does not contain "location" or a separate brand/model
# split -- those are derived below from the free-text `name` column. If a
# different dataset is dropped into data/raw/used_cars.csv, update
# RAW_COLUMN_MAP below rather than editing the rest of the codebase.

TARGET_COLUMN = "selling_price"

# Maps this project's internal/canonical column names -> actual column
# names found in the raw CSV. Change the right-hand side if you swap in a
# dataset that uses different column headers (e.g. "Price" instead of
# "selling_price").
RAW_COLUMN_MAP = {
    "name": "name",
    "year": "year",
    "selling_price": "selling_price",
    "km_driven": "km_driven",
    "fuel": "fuel",
    "seller_type": "seller_type",
    "transmission": "transmission",
    "owner": "owner",
    "mileage": "mileage",
    "engine": "engine",
    "max_power": "max_power",
    "torque": "torque",
    "seats": "seats",
}

# Columns that require unit-stripping (e.g. "1248 CC" -> 1248.0) during
# cleaning. Maps column name -> regex pattern that captures the numeric part.
UNIT_COLUMNS = {
    "mileage": r"([\d.]+)",   # "23.4 kmpl" or "23.4 km/kg" -> 23.4
    "engine": r"([\d.]+)",    # "1248 CC" -> 1248
    "max_power": r"([\d.]+)", # "74 bhp" -> 74 (a few rows are "bhp" only -> NaN)
}

# Reference year used to compute vehicle age (Car_Age = CURRENT_YEAR - year).
# Fixed rather than datetime.now() so results are reproducible on any date.
CURRENT_YEAR = 2024

# Numerical features used by the model (after feature engineering).
NUMERICAL_FEATURES = [
    "car_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
]

# Categorical features used by the model.
CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Brands with fewer than this many listings are grouped into "Other" to
# avoid a high-cardinality one-hot explosion and unstable per-brand effects.
RARE_BRAND_THRESHOLD = 20
