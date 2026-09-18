"""
Prediction interface used by both the Streamlit app and the test suite.

Keeps input validation and price-range logic in one place so the GUI layer
stays thin and every validation rule is unit-testable in isolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import pandas as pd

from src.config import ALL_FEATURES, CURRENT_YEAR, MODEL_METADATA_PATH, MODEL_PATH


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model pipeline has not been generated yet."""


class InputValidationError(ValueError):
    """Raised when user-supplied vehicle details fail a validation rule."""


VALID_FUEL_TYPES = {"Petrol", "Diesel", "CNG", "LPG", "Electric"}
VALID_SELLER_TYPES = {"Individual", "Dealer", "Trustmark Dealer"}
VALID_TRANSMISSIONS = {"Manual", "Automatic"}
VALID_OWNER_TYPES = {
    "First Owner", "Second Owner", "Third Owner",
    "Fourth & Above Owner", "Test Drive Car",
}

MIN_YEAR = 1980
MAX_KM_DRIVEN = 1_000_000
MAX_MILEAGE_KMPL = 50.0
MAX_ENGINE_CC = 8000
MAX_POWER_BHP = 1000
MIN_SEATS, MAX_SEATS = 2, 10


@dataclass
class CarInput:
    brand: str
    year: int
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    seats: int


def validate_input(data: dict) -> CarInput:
    """Validate a raw dict of user-supplied vehicle details.

    Raises InputValidationError with a human-readable message on the first
    rule violated. Returns a CarInput on success.
    """
    required_fields = [
        "brand", "year", "km_driven", "fuel", "seller_type",
        "transmission", "owner", "mileage", "engine", "max_power", "seats",
    ]
    missing = [f for f in required_fields if f not in data or data[f] in (None, "")]
    if missing:
        raise InputValidationError(f"Missing required field(s): {', '.join(missing)}")

    try:
        year = int(data["year"])
        km_driven = float(data["km_driven"])
        mileage = float(data["mileage"])
        engine = float(data["engine"])
        max_power = float(data["max_power"])
        seats = int(data["seats"])
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"Numeric field could not be parsed: {exc}") from exc

    if not (MIN_YEAR <= year <= CURRENT_YEAR):
        raise InputValidationError(
            f"Manufacturing year must be between {MIN_YEAR} and {CURRENT_YEAR}, got {year}."
        )
    if km_driven < 0:
        raise InputValidationError(f"Kilometers driven cannot be negative, got {km_driven}.")
    if km_driven > MAX_KM_DRIVEN:
        raise InputValidationError(f"Kilometers driven exceeds plausible maximum ({MAX_KM_DRIVEN}), got {km_driven}.")
    if mileage <= 0 or mileage > MAX_MILEAGE_KMPL:
        raise InputValidationError(f"Mileage must be between 0 and {MAX_MILEAGE_KMPL} kmpl, got {mileage}.")
    if engine <= 0 or engine > MAX_ENGINE_CC:
        raise InputValidationError(f"Engine capacity must be between 0 and {MAX_ENGINE_CC} CC, got {engine}.")
    if max_power <= 0 or max_power > MAX_POWER_BHP:
        raise InputValidationError(f"Max power must be between 0 and {MAX_POWER_BHP} bhp, got {max_power}.")
    if not (MIN_SEATS <= seats <= MAX_SEATS):
        raise InputValidationError(f"Seats must be between {MIN_SEATS} and {MAX_SEATS}, got {seats}.")
    if data["fuel"] not in VALID_FUEL_TYPES:
        raise InputValidationError(f"Unknown fuel type '{data['fuel']}'. Expected one of {sorted(VALID_FUEL_TYPES)}.")
    if data["seller_type"] not in VALID_SELLER_TYPES:
        raise InputValidationError(f"Unknown seller type '{data['seller_type']}'. Expected one of {sorted(VALID_SELLER_TYPES)}.")
    if data["transmission"] not in VALID_TRANSMISSIONS:
        raise InputValidationError(f"Unknown transmission '{data['transmission']}'. Expected one of {sorted(VALID_TRANSMISSIONS)}.")
    if data["owner"] not in VALID_OWNER_TYPES:
        raise InputValidationError(f"Unknown owner type '{data['owner']}'. Expected one of {sorted(VALID_OWNER_TYPES)}.")
    if not str(data["brand"]).strip():
        raise InputValidationError("Brand cannot be empty.")

    return CarInput(
        brand=str(data["brand"]).strip().title(),
        year=year,
        km_driven=km_driven,
        fuel=data["fuel"],
        seller_type=data["seller_type"],
        transmission=data["transmission"],
        owner=data["owner"],
        mileage=mileage,
        engine=engine,
        max_power=max_power,
        seats=seats,
    )


@lru_cache(maxsize=1)
def load_model():
    """Load the trained pipeline (cached so Streamlit doesn't reload per interaction)."""
    if not MODEL_PATH.exists():
        raise ModelNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run 'python -m src.train' first to train and save the model pipeline."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_model_metadata() -> dict:
    if not MODEL_METADATA_PATH.exists():
        return {}
    with open(MODEL_METADATA_PATH) as f:
        return json.load(f)


def predict_price(data: dict) -> dict:
    """Validate input, run the trained pipeline, and build a result dict.

    Returns
    -------
    dict with keys: predicted_price, price_range_low, price_range_high,
    range_method, car_age, factors (list of human-readable strings).
    """
    car = validate_input(data)
    model = load_model()
    metadata = load_model_metadata()

    car_age = CURRENT_YEAR - car.year
    row = pd.DataFrame([{
        "car_age": car_age,
        "km_driven": car.km_driven,
        "mileage": car.mileage,
        "engine": car.engine,
        "max_power": car.max_power,
        "seats": car.seats,
        "brand": car.brand,
        "fuel": car.fuel,
        "seller_type": car.seller_type,
        "transmission": car.transmission,
        "owner": car.owner,
    }])[ALL_FEATURES]

    predicted_price = float(model.predict(row)[0])
    predicted_price = max(predicted_price, 0.0)

    residual_p10 = metadata.get("residual_p10")
    residual_p90 = metadata.get("residual_p90")
    if residual_p10 is not None and residual_p90 is not None:
        range_low = max(predicted_price + residual_p10, 0.0)
        range_high = predicted_price + residual_p90
        range_method = (
            "Empirical range: on the held-out test set, the actual price fell "
            "within [prediction + p10 residual, prediction + p90 residual] for "
            "80% of vehicles. This is NOT a statistically calibrated confidence "
            "interval -- it is an empirical historical-error band."
        )
    else:
        range_low = predicted_price * 0.9
        range_high = predicted_price * 1.1
        range_method = "Approximate +/-10% range (model metadata unavailable; not a calibrated interval)."

    factors = [
        f"Vehicle age: {car_age} years (manufactured {car.year})",
        f"Kilometers driven: {car.km_driven:,.0f} km",
        f"Engine: {car.engine:.0f} CC, {car.max_power:.0f} bhp",
        f"Fuel type: {car.fuel}",
        f"Transmission: {car.transmission}",
        f"Ownership: {car.owner}",
    ]

    return {
        "predicted_price": predicted_price,
        "price_range_low": range_low,
        "price_range_high": range_high,
        "range_method": range_method,
        "car_age": car_age,
        "factors": factors,
    }
