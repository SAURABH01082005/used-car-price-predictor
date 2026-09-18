"""Data validation tests (TC-001 series maps to reports/test_cases.md)."""

import pandas as pd
import pytest

from src.config import RAW_DATA_PATH, RAW_COLUMN_MAP, TARGET_COLUMN
from src.data_loader import DatasetNotFoundError, SchemaValidationError, load_raw_data


def test_dataset_file_exists():
    """TC-D01: the raw dataset file must exist at the configured path."""
    assert RAW_DATA_PATH.exists(), f"Expected dataset at {RAW_DATA_PATH}"


def test_load_raw_data_returns_dataframe():
    """TC-D02: loading the raw dataset returns a non-empty DataFrame."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_raw_data_has_expected_columns():
    """TC-D03: all canonical columns are present in the raw dataset."""
    df = load_raw_data()
    for col in RAW_COLUMN_MAP.values():
        assert col in df.columns, f"Missing expected column: {col}"


def test_target_column_present_and_numeric():
    """TC-D04: target column exists and is numeric."""
    df = load_raw_data()
    assert TARGET_COLUMN in df.columns
    assert pd.api.types.is_numeric_dtype(df[TARGET_COLUMN])


def test_missing_dataset_raises_clear_error(tmp_path):
    """TC-D05: loading a nonexistent CSV raises DatasetNotFoundError with a helpful message."""
    fake_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
        load_raw_data(path=fake_path)


def test_dataset_missing_required_column_raises_schema_error(tmp_path):
    """TC-D06: a CSV missing a required column raises SchemaValidationError."""
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["Test Car"], "year": [2020]}).to_csv(bad_csv, index=False)
    with pytest.raises(SchemaValidationError, match="missing expected column"):
        load_raw_data(path=bad_csv)


def test_empty_dataset_raises_schema_error(tmp_path):
    """TC-D07: an empty CSV (headers only, zero rows) raises SchemaValidationError."""
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame(columns=list(RAW_COLUMN_MAP.values())).to_csv(empty_csv, index=False)
    with pytest.raises(SchemaValidationError, match="zero rows"):
        load_raw_data(path=empty_csv)
