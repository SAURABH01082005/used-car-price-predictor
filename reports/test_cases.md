# Test Case Report

Project: Used Car Price Prediction and Analysis System
Test framework: `pytest`
Test files: `tests/test_data.py`, `tests/test_preprocessing.py`, `tests/test_model.py`, `tests/test_prediction.py`

Result of last full run: **40 / 40 passed** (`pytest tests/ -v`, ~1.4s total).
Re-run yourself with `pytest -v` any time after `python -m src.train`.

## Data Validation Tests (`tests/test_data.py`)

| ID | Description | Expected Result | Status |
|----|---|---|---|
| TC-D01 | Raw dataset file exists at `data/raw/used_cars.csv` | File found | PASS |
| TC-D02 | Loading raw dataset returns a non-empty DataFrame | DataFrame with >0 rows | PASS |
| TC-D03 | Raw dataset has all expected columns | All 13 canonical columns present | PASS |
| TC-D04 | Target column (`selling_price`) present and numeric | Numeric dtype | PASS |
| TC-D05 | Loading a nonexistent CSV path | `DatasetNotFoundError` with actionable message | PASS |
| TC-D06 | CSV missing a required column | `SchemaValidationError` naming the missing column | PASS |
| TC-D07 | CSV with headers but zero rows | `SchemaValidationError` ("zero rows") | PASS |

## Preprocessing / Feature Engineering Tests (`tests/test_preprocessing.py`)

| ID | Description | Expected Result | Status |
|----|---|---|---|
| TC-P01 | Unit stripping: `"1248 CC"` → `1248.0`, `"74 bhp"` → `74.0`, `"23.4 kmpl"` → `23.4` | Correct numeric extraction | PASS |
| TC-P02 | Unit stripping on a garbled/non-numeric string | Returns `NaN`, does not raise | PASS |
| TC-P03 | Duplicate rows removed | Row count reduced correctly | PASS |
| TC-P04 | Rows with `selling_price <= 0` removed | All remaining rows have positive price | PASS |
| TC-P05 | `mileage`/`engine`/`max_power` are numeric after cleaning | Correct dtypes | PASS |
| TC-P06 | `torque` column dropped (inconsistent free-text format) | Column absent after cleaning | PASS |
| TC-P07 | `car_age` computed correctly, never negative | `car_age = CURRENT_YEAR - year`, clipped at 0 | PASS |
| TC-P08 | `brand` extracted from first token of `name`, title-cased | Correct brand strings | PASS |
| TC-P09 | Brands below frequency threshold grouped into `"Other"` | Rare brands relabeled | PASS |
| TC-P10 | `ColumnTransformer` fits/transforms data with missing values, no NaNs remain | Dense numeric array, no NaN | PASS |

## Model Loading Tests (`tests/test_model.py`)

| ID | Description | Expected Result | Status |
|----|---|---|---|
| TC-M01 | Trained model file exists | `models/used_car_price_model.pkl` found | PASS |
| TC-M02 | Saved artifact loads as an sklearn `Pipeline` with `preprocessor` + `model` steps | Correct pipeline structure | PASS |
| TC-M06 | Missing model file | `ModelNotFoundError` telling the user to run `python -m src.train` | PASS |
| TC-M07 | Corrupted (non-pickle) model file | Loading raises an exception rather than returning garbage | PASS |
| TC-M03 | Model metadata JSON exists with `final_model_name` | Present and non-empty | PASS |
| TC-M04 | `model_comparison.csv` generated with ≥5 candidate models | RMSE/R2 columns present | PASS |
| TC-M05 | Final model quality bar | Test R² > 0.5 (actual: see `reports/results.md`) | PASS |

## Prediction & Input Validation Tests (`tests/test_prediction.py`)

| ID | Description | Expected Result | Status |
|----|---|---|---|
| TC-001 | Valid vehicle input | Prediction generated successfully | PASS |
| TC-002 | Missing required feature | Validation error naming the missing field | PASS |
| TC-003 | Negative kilometers driven | Invalid input rejected | PASS |
| TC-004 | Future manufacturing year | Invalid input rejected | PASS |
| TC-005 | Extremely high mileage (5000 kmpl) | Input handled safely (rejected with clear message) | PASS |
| TC-007 | Valid categorical input (Diesel + Automatic) | Prediction generated | PASS |
| TC-009 | Zero engine capacity | Invalid input rejected | PASS |
| TC-010 | Unknown fuel category (`"Hydrogen"`) | Validation error listing allowed values | PASS |
| TC-011 | Seats outside plausible range (50) | Invalid input rejected | PASS |
| TC-012 | Empty-string required field | Treated as missing, validation error | PASS |
| TC-013 | Non-numeric value in a numeric field | Validation error, not a crash | PASS |
| TC-014 | Brand never seen during training | Handled gracefully (`OneHotEncoder(handle_unknown="ignore")`), prediction still returned | PASS |
| TC-015 | Very high but technically valid km_driven (999,999) | Prediction returned, no crash | PASS |

## Reliability Tests (`tests/test_prediction.py`, detailed report: `reports/reliability_test.md`)

| ID | Description | Expected Result | Status |
|----|---|---|---|
| TC-016 | Identical input predicted 10× | All 10 predictions identical (deterministic) | PASS |
| TC-017 | Single prediction execution time | < 2.0 seconds on a warm model | PASS |
| TC-018 | Batch of 4 distinct valid inputs | All predict successfully | PASS |

**Total: 40 automated test cases across 4 files, all passing at time of writing.**
