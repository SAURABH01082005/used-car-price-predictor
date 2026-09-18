"""Preprocessing and feature-engineering tests (TC-P series)."""

import numpy as np
import pandas as pd
import pytest

from src.config import CURRENT_YEAR
from src.feature_engineering import add_brand, add_car_age, group_rare_brands
from src.preprocessing import _extract_numeric, build_preprocessing_pipeline, clean_raw_data


def _sample_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["Maruti Swift Dzire VDI", "Maruti Swift Dzire VDI", "Hyundai i20 Sportz"],
        "year": [2014, 2014, 2010],
        "selling_price": [450000, 450000, 0],  # duplicate row + invalid target
        "km_driven": [145500, 145500, 127000],
        "fuel": ["Diesel", "Diesel", "Diesel"],
        "seller_type": ["Individual", "Individual", "Individual"],
        "transmission": ["Manual", "Manual", "Manual"],
        "owner": ["First Owner", "First Owner", "First Owner"],
        "mileage": ["23.4 kmpl", "23.4 kmpl", "23.0 kmpl"],
        "engine": ["1248 CC", "1248 CC", "1396 CC"],
        "max_power": ["74 bhp", "74 bhp", "90 bhp"],
        "torque": ["190Nm@ 2000rpm", "190Nm@ 2000rpm", "22.4 kgm at 1750rpm"],
        "seats": [5, 5, 5],
    })


def test_extract_numeric_strips_units():
    """TC-P01: '1248 CC' -> 1248.0, '74 bhp' -> 74.0, '23.4 kmpl' -> 23.4."""
    series = pd.Series(["1248 CC", "74 bhp", "23.4 kmpl"])
    result = _extract_numeric(series, r"([\d.]+)")
    assert list(result) == [1248.0, 74.0, 23.4]


def test_extract_numeric_handles_garbled_value():
    """TC-P02: a value with no numeric token becomes NaN instead of crashing."""
    series = pd.Series(["not available"])
    result = _extract_numeric(series, r"([\d.]+)")
    assert pd.isna(result.iloc[0])


def test_clean_raw_data_removes_duplicates():
    """TC-P03: exact duplicate rows are dropped."""
    df = _sample_raw_df()
    cleaned = clean_raw_data(df)
    # duplicate row (rows 0 and 1 are identical) collapses to one,
    # and the zero-price row (row 2) is dropped for invalid target.
    assert len(cleaned) == 1


def test_clean_raw_data_drops_invalid_target():
    """TC-P04: rows with selling_price <= 0 are dropped."""
    df = _sample_raw_df()
    cleaned = clean_raw_data(df)
    assert (cleaned["selling_price"] > 0).all()


def test_clean_raw_data_units_stripped_to_numeric():
    """TC-P05: after cleaning, mileage/engine/max_power are numeric dtypes."""
    df = _sample_raw_df()
    cleaned = clean_raw_data(df)
    for col in ["mileage", "engine", "max_power"]:
        assert pd.api.types.is_numeric_dtype(cleaned[col])


def test_clean_raw_data_drops_torque_column():
    """TC-P06: torque column is dropped due to inconsistent formatting."""
    df = _sample_raw_df()
    cleaned = clean_raw_data(df)
    assert "torque" not in cleaned.columns


def test_car_age_computed_correctly():
    """TC-P07: car_age = CURRENT_YEAR - year, never negative."""
    df = pd.DataFrame({"year": [2020, CURRENT_YEAR + 5]})
    result = add_car_age(df)
    assert result.loc[0, "car_age"] == CURRENT_YEAR - 2020
    assert result.loc[1, "car_age"] == 0  # clipped, no negative ages


def test_brand_extracted_from_name():
    """TC-P08: brand = first token of `name`, title-cased."""
    df = pd.DataFrame({"name": ["maruti swift dzire", "BMW X5"]})
    result = add_brand(df)
    assert list(result["brand"]) == ["Maruti", "Bmw"]


def test_rare_brands_grouped_into_other():
    """TC-P09: brands below the frequency threshold are relabeled 'Other'."""
    df = pd.DataFrame({"brand": ["Maruti"] * 30 + ["RareBrand"] * 2})
    result = group_rare_brands(df, threshold=20)
    assert "RareBrand" not in result["brand"].values
    assert (result["brand"] == "Other").sum() == 2


def test_preprocessing_pipeline_fits_and_transforms():
    """TC-P10: the ColumnTransformer fits on training-shaped data without error
    and produces a numeric, dense-compatible output."""
    df = pd.DataFrame({
        "car_age": [5, 10, np.nan],
        "km_driven": [50000, 80000, 30000],
        "mileage": [18.0, np.nan, 20.0],
        "engine": [1200, 1500, 1000],
        "max_power": [80, 100, np.nan],
        "seats": [5, 5, 7],
        "brand": ["Maruti", "Hyundai", None],
        "fuel": ["Petrol", "Diesel", "Petrol"],
        "seller_type": ["Individual", "Dealer", "Individual"],
        "transmission": ["Manual", "Automatic", "Manual"],
        "owner": ["First Owner", "Second Owner", "First Owner"],
    })
    preprocessor = build_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    assert not np.isnan(transformed.toarray() if hasattr(transformed, "toarray") else transformed).any()
