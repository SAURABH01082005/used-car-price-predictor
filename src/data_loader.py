"""
Data loading and schema validation.

Keeps a single, explicit boundary between "whatever CSV the user drops in
data/raw/" and the rest of the pipeline, so column-mismatch errors surface
early with a readable message instead of failing deep inside sklearn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import RAW_COLUMN_MAP, RAW_DATA_PATH

logger = logging.getLogger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected raw dataset CSV is missing."""


class SchemaValidationError(ValueError):
    """Raised when the raw dataset does not contain the expected columns."""


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw used-car CSV and validate it has the expected columns.

    Parameters
    ----------
    path: location of the raw CSV. Defaults to data/raw/used_cars.csv.

    Returns
    -------
    Raw, unmodified DataFrame (no cleaning applied yet).
    """
    if not path.exists():
        raise DatasetNotFoundError(
            f"Dataset not found at {path}.\n"
            "This project expects the CarDekho used-car dataset "
            "('Car details v3.csv' from the Kaggle dataset "
            "'nehalbirla/vehicle-dataset-from-cardekho') saved as "
            f"{path.name} inside {path.parent}.\n"
            "Expected columns: name, year, selling_price, km_driven, fuel, "
            "seller_type, transmission, owner, mileage, engine, max_power, "
            "torque, seats."
        )

    df = pd.read_csv(path)
    _validate_schema(df, path)
    logger.info("Loaded raw dataset: %d rows, %d columns from %s", len(df), df.shape[1], path)
    return df


def _validate_schema(df: pd.DataFrame, path: Path) -> None:
    expected_columns = set(RAW_COLUMN_MAP.values())
    actual_columns = set(df.columns)
    missing = expected_columns - actual_columns

    if missing:
        raise SchemaValidationError(
            f"Dataset at {path} is missing expected column(s): {sorted(missing)}.\n"
            f"Columns found in file: {sorted(actual_columns)}.\n"
            "If you are using a different dataset, update RAW_COLUMN_MAP in "
            "src/config.py to map these canonical names onto your file's "
            "actual column headers."
        )

    if df.empty:
        raise SchemaValidationError(f"Dataset at {path} loaded but contains zero rows.")
