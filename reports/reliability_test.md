# Reliability Test Report

All numbers on this page were measured by actually running the trained
pipeline (`models/used_car_price_model.pkl`, final model: **Gradient
Boosting**) on this machine. Reproduce with:

```
python -m pytest tests/test_prediction.py -k reliability -v
```

or the standalone script used to generate these numbers (50 repeated calls
+ a 5-variant batch), executed 2026-09-18.

## 1. Prediction Consistency

Same input predicted **50 times** in a row:

| Metric | Result |
|---|---|
| Number of runs | 50 |
| Distinct prediction values produced | 1 |
| Predicted price (every run) | ₹583,568.94 |

**Conclusion:** the model is deterministic for a fixed input — no
randomness leaks into inference (the underlying Gradient Boosting model has
`random_state=42` fixed at *training* time; at *prediction* time there is no
stochastic component at all). No crashes across 50 consecutive calls.

## 2. Execution Time

Timed on a warm process (model already loaded via `functools.lru_cache`),
50 repeated single-prediction calls:

| Metric | Value |
|---|---|
| Mean | 2.99 ms |
| Median | 2.93 ms |
| Min | 2.90 ms |
| Max | 4.71 ms |
| Std. deviation | 0.26 ms |

**Conclusion:** a single prediction consistently completes in under 5 ms
once the model is loaded, well within any reasonable interactive UI budget
(Streamlit's own render cycle dominates perceived latency, not this call).
The one-time model *load* from disk (`joblib.load`) is not included here —
it happens once per app session, not per prediction.

## 3. Multiple Valid Inputs (Robustness Across Varied Requests)

Five structurally different valid inputs (varying fuel, transmission,
seller type, ownership, brand, and one deliberately unseen brand) were all
predicted successfully with no exceptions:

| Variant | Predicted Price |
|---|---|
| Diesel, 2012 | ₹363,326.26 |
| Automatic, 7 seats | ₹646,203.54 |
| Dealer, Second Owner | ₹570,438.67 |
| Hyundai, 120,000 km | ₹561,037.80 |
| Unknown brand (`UnknownBrandXYZ`) | ₹574,049.38 |

**Conclusion:** the pipeline's `OneHotEncoder(handle_unknown="ignore")`
correctly absorbs a brand never seen during training instead of raising —
the prediction still returns a plausible, non-crashing result rather than
an error, satisfying the "input handled safely" requirement for out-of-
vocabulary categories.

## Summary

| Reliability Dimension | Result |
|---|---|
| Consistency (same input → same output) | 100% (50/50 identical) |
| Crash rate over 50 + 5 = 55 calls | 0% |
| Mean latency per prediction (warm) | ~3 ms |
| Handles unseen category gracefully | Yes |

No performance or reliability numbers on this page were estimated or
assumed — all were produced by executing `src/predict.py` against the
actual trained model.
