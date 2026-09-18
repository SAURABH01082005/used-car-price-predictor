"""
Prediction, input-validation, edge-case and reliability tests.

Maps to reports/test_cases.md (TC-001 .. TC-0xx). Assumes a trained model
exists at models/used_car_price_model.pkl (run 'python -m src.train' first).
"""

import time

import pytest

from src.predict import InputValidationError, predict_price, validate_input

VALID_INPUT = {
    "brand": "Maruti",
    "year": 2018,
    "km_driven": 40000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": 18.5,
    "engine": 1200,
    "max_power": 82,
    "seats": 5,
}


def test_valid_input_generates_prediction():
    """TC-001: valid vehicle input -> prediction generated successfully."""
    result = predict_price(VALID_INPUT)
    assert result["predicted_price"] > 0
    assert result["price_range_low"] <= result["predicted_price"] <= result["price_range_high"]


def test_missing_required_field_raises_validation_error():
    """TC-002: missing required feature -> validation error."""
    data = VALID_INPUT.copy()
    del data["engine"]
    with pytest.raises(InputValidationError, match="Missing required field"):
        validate_input(data)


def test_negative_km_driven_rejected():
    """TC-003: negative kilometers -> invalid input rejected."""
    data = VALID_INPUT.copy()
    data["km_driven"] = -500
    with pytest.raises(InputValidationError, match="cannot be negative"):
        validate_input(data)


def test_future_manufacturing_year_rejected():
    """TC-004: future manufacturing year -> invalid input rejected."""
    data = VALID_INPUT.copy()
    data["year"] = 2099
    with pytest.raises(InputValidationError, match="Manufacturing year"):
        validate_input(data)


def test_extremely_high_mileage_rejected_safely():
    """TC-005: implausibly high mileage -> handled safely (rejected with clear message, no crash)."""
    data = VALID_INPUT.copy()
    data["mileage"] = 5000.0
    with pytest.raises(InputValidationError, match="Mileage must be between"):
        validate_input(data)


def test_valid_categorical_input_generates_prediction():
    """TC-007: valid categorical combination -> prediction generated."""
    data = VALID_INPUT.copy()
    data["fuel"] = "Diesel"
    data["transmission"] = "Automatic"
    result = predict_price(data)
    assert result["predicted_price"] > 0


def test_zero_engine_capacity_rejected():
    """TC-009: zero engine capacity -> invalid input rejected."""
    data = VALID_INPUT.copy()
    data["engine"] = 0
    with pytest.raises(InputValidationError, match="Engine capacity"):
        validate_input(data)


def test_unknown_fuel_category_rejected():
    """TC-010: unrecognized fuel type string -> validation error with allowed values."""
    data = VALID_INPUT.copy()
    data["fuel"] = "Hydrogen"
    with pytest.raises(InputValidationError, match="Unknown fuel type"):
        validate_input(data)


def test_seats_out_of_range_rejected():
    """TC-011: seats outside plausible range -> invalid input rejected."""
    data = VALID_INPUT.copy()
    data["seats"] = 50
    with pytest.raises(InputValidationError, match="Seats must be between"):
        validate_input(data)


def test_empty_string_field_rejected():
    """TC-012: empty-string field treated as missing -> validation error."""
    data = VALID_INPUT.copy()
    data["brand"] = ""
    with pytest.raises(InputValidationError, match="Missing required field"):
        validate_input(data)


def test_non_numeric_value_in_numeric_field_rejected():
    """TC-013: non-numeric string in a numeric field -> validation error, not a crash."""
    data = VALID_INPUT.copy()
    data["km_driven"] = "abc"
    with pytest.raises(InputValidationError, match="could not be parsed"):
        validate_input(data)


def test_unknown_brand_handled_gracefully():
    """TC-014: a brand never seen during training does not crash prediction
    (OneHotEncoder handle_unknown='ignore' zeroes the brand contribution)."""
    data = VALID_INPUT.copy()
    data["brand"] = "TotallyUnknownBrandXYZ"
    result = predict_price(data)
    assert result["predicted_price"] >= 0


def test_extreme_but_valid_km_driven_handled():
    """TC-015: very high but technically valid km_driven does not crash the pipeline."""
    data = VALID_INPUT.copy()
    data["km_driven"] = 999999
    result = predict_price(data)
    assert result["predicted_price"] >= 0


# ---------------------------------------------------------------------------
# Reliability testing (see reports/reliability_test.md for the generated report)
# ---------------------------------------------------------------------------

def test_repeated_identical_prediction_is_consistent():
    """TC-016 (Reliability): running the same input 10 times gives an identical
    prediction every time (deterministic model, no crashes)."""
    predictions = [predict_price(VALID_INPUT)["predicted_price"] for _ in range(10)]
    assert len(set(predictions)) == 1, "Prediction should be identical for identical input"


def test_prediction_execution_time_is_reasonable():
    """TC-017 (Reliability): a single prediction completes in well under 2 seconds
    on a warm (already-loaded) model."""
    predict_price(VALID_INPUT)  # warm the cached model load
    start = time.time()
    predict_price(VALID_INPUT)
    duration = time.time() - start
    assert duration < 2.0, f"Prediction took {duration:.3f}s, expected < 2.0s"


def test_multiple_distinct_valid_inputs_all_succeed():
    """TC-018 (Reliability): a batch of varied valid inputs all predict without error."""
    variants = [
        {**VALID_INPUT, "fuel": "Diesel", "year": 2012},
        {**VALID_INPUT, "transmission": "Automatic", "seats": 7},
        {**VALID_INPUT, "seller_type": "Dealer", "owner": "Second Owner"},
        {**VALID_INPUT, "brand": "Hyundai", "km_driven": 120000},
    ]
    for data in variants:
        result = predict_price(data)
        assert result["predicted_price"] > 0
