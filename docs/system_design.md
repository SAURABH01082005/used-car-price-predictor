# System Design

## Architecture Overview

```
                User
                 |
                 v
         Streamlit UI (app.py)
                 |
                 v
        Input Validation (src/predict.py)
                 |
                 v
        Feature Engineering (src/feature_engineering.py)
                 |
                 v
     Preprocessing Pipeline (src/preprocessing.py,
       fitted, bundled inside the saved model)
                 |
                 v
        Trained ML Model (models/used_car_price_model.pkl)
                 |
                 v
           Price Prediction
                 |
                 v
        Result Visualization (Streamlit)
```

## Layered Design

### 1. Input Layer
`app.py` (Streamlit form widgets) collects raw user input as a plain
Python dict. No business logic lives here — it is a thin presentation
layer that calls into `src/predict.py`.

### 2. Processing Layer
- `src/predict.py::validate_input` — rejects out-of-range, missing, or
  unrecognized values before anything touches the model (year range,
  km_driven ≥ 0, mileage/engine/power plausibility bounds, known
  categorical values).
- `src/feature_engineering.py` — derives `car_age` and `brand` from raw
  fields; the same functions are used identically during training
  (`src/train.py`) and inference (`src/predict.py`), so there is no
  train/serve skew.
- The fitted `ColumnTransformer` (imputation, scaling, one-hot encoding)
  is bundled **inside** the saved sklearn `Pipeline`, so inference always
  applies the exact statistics (medians, categories) learned at training
  time — not recomputed per request.

### 3. ML Layer
A single sklearn `Pipeline` object
(`Pipeline([("preprocessor", ColumnTransformer), ("model", estimator)])`)
is trained once (`python -m src.train`), evaluated, and serialized with
`joblib`. The Streamlit app loads this one artifact
(`models/used_car_price_model.pkl`) and never retrains at request time —
prediction is a single `pipeline.predict(row)` call (~3 ms measured, see
`reports/reliability_test.md`).

### 4. Presentation Layer
Streamlit renders: the point prediction, an empirical price range (labeled
with its actual method — residual quantiles from the test set, not a
fabricated confidence interval), a list of human-readable prediction
factors, and (on other pages) EDA charts and model-comparison tables read
directly from generated CSV/JSON reports — nothing is hard-coded in the UI.

## Data Flow (Training Time vs. Inference Time)

| Stage | Training (`src/train.py`) | Inference (`src/predict.py`) |
|---|---|---|
| Load data | Full CSV from `data/raw/` | Single user-submitted row |
| Clean | `clean_raw_data()` | N/A (user input is already structured) |
| Engineer features | `engineer_features()` | Same functions, applied to the one row |
| Preprocess | `ColumnTransformer.fit_transform()` | `ColumnTransformer.transform()` (already fitted) |
| Model | `.fit()` on training split | `.predict()` only |
| Output | Saved pipeline + metrics + plots | Predicted price + range + factors |

## Why a Single Bundled Pipeline (Not Separate Preprocessor + Model Files)

Bundling preprocessing and the model into one `sklearn.pipeline.Pipeline`
guarantees the exact same transformation is applied at inference time as
was fit during training — there is no way for the app to accidentally load
a mismatched preprocessor/model pair, and no risk of the app re-deriving
imputation statistics from whatever data happens to be present at runtime
(which would itself be a subtle form of data leakage / train-serve skew).
