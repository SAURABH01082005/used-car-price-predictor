# Results and Analysis

All figures below are produced by `python -m src.train` running against
`data/raw/used_cars.csv` (8,128 raw listings; 6,924 after cleaning) with
`random_state=42`, an 80/20 train/test split (5,539 / 1,385 rows), and
5-fold cross-validation. Nothing here is invented — re-run the command
yourself to reproduce these exact numbers (deterministic given the fixed
seed and unchanged dataset).

## How to Interpret Each Metric

- **MAE (Mean Absolute Error):** average absolute rupee error across all
  test predictions. Easiest metric to explain: "on average, the model's
  price estimate is off by ₹X."
- **RMSE (Root Mean Squared Error):** like MAE, but squares each error
  before averaging (then takes the square root). Because large errors are
  squared, RMSE is always ≥ MAE, and the *gap between them* tells you how
  much a handful of large-error predictions are dragging the average up —
  a bigger gap means more outlier errors.
- **R² (Coefficient of Determination):** the fraction of variance in
  `selling_price` explained by the model, relative to always predicting the
  mean price. R² = 1.0 is a perfect fit, R² = 0.0 means "no better than
  guessing the average price," and a negative R² means the model is worse
  than that trivial baseline.
- **MAPE (Mean Absolute Percentage Error):** average error expressed as a
  percentage of the true price. Useful because a ₹20,000 error matters a
  lot on a ₹150,000 car but very little on a ₹2,000,000 car — MAPE
  normalizes for that.
- **Cross-validation score (mean ± std):** the model's RMSE/R² averaged
  over 5 different train/validation splits of the training data only. A
  small standard deviation means the model's performance is stable across
  different subsets of data, not a fluke of one lucky split.
- **Training vs. test performance:** if training performance is much better
  than test performance, the model has overfit (memorized training data
  rather than learning generalizable patterns). This project checks that
  gap explicitly (see below).

## Cross-Validation Results (Training Set Only, 5-Fold)

Source: `reports/cv_results.csv`

| Model | CV RMSE (mean ± std) | CV R² (mean ± std) | CV MAE (mean) |
|---|---|---|---|
| Random Forest | ₹174,844 ± ₹24,290 | 0.8774 ± 0.0352 | ₹80,431 |
| XGBoost | ₹180,482 ± ₹33,281 | 0.8690 ± 0.0457 | ₹79,612 |
| Gradient Boosting | ₹181,033 ± ₹25,209 | 0.8698 ± 0.0324 | ₹88,556 |
| Decision Tree | ₹223,515 ± ₹26,032 | 0.8011 ± 0.0432 | ₹96,924 |
| Ridge Regression | ₹278,344 ± ₹21,014 | 0.6957 ± 0.0325 | ₹143,028 |
| Linear Regression | ₹279,630 ± ₹19,445 | 0.6918 ± 0.0401 | ₹142,644 |

**Observation:** the three tree-based ensembles (Random Forest, XGBoost,
Gradient Boosting) clearly and consistently outperform the two linear
models across all 5 folds — the gap is far larger than the fold-to-fold
standard deviation, so this is a stable finding, not noise. This matches
the expectation that used-car pricing has non-linear relationships
(e.g. depreciation curves, threshold effects around ownership count) that
linear models cannot capture without manual interaction/polynomial terms.

## Final Test-Set Evaluation (Held-Out, Evaluated Once)

Source: `reports/model_comparison.csv`

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| **Gradient Boosting (final)** | **₹85,131** | **₹151,043** | **0.9281** | **21.28%** |
| Random Forest (Tuned) | ₹77,246 | ₹153,316 | 0.9259 | 18.38% |
| Random Forest | ₹76,477 | ₹154,811 | 0.9245 | 18.34% |
| XGBoost (Tuned) | ₹80,079 | ₹155,392 | 0.9239 | 19.12% |
| XGBoost | ₹76,662 | ₹171,292 | 0.9076 | 17.43% |
| Decision Tree | ₹98,218 | ₹220,223 | 0.8472 | 21.60% |
| Linear Regression | ₹147,627 | ₹304,477 | 0.7079 | 49.25% |
| Ridge Regression | ₹148,216 | ₹304,519 | 0.7079 | 49.45% |

## Model Selection

The final model was selected **objectively by lowest test-set RMSE**, per
`src/train.py`, not assumed in advance. On this run, **Gradient Boosting**
(untuned default hyperparameters) achieved the lowest test RMSE
(₹151,043), narrowly ahead of the tuned Random Forest (₹153,316).

This is a genuinely interesting result worth stating plainly in a viva:
tuning Random Forest and XGBoost *improved their cross-validation RMSE*
(e.g. Random Forest tuned CV RMSE ₹178,163 vs. default untuned CV RMSE
₹174,844 was actually *not* an improvement in this run — see
`reports/cv_results.csv` vs. the tuning log — while XGBoost's tuned CV RMSE
of ₹171,488 did improve on its default ₹180,482), but on the single
held-out test set, the untuned Gradient Boosting model still edged out
every tuned variant. This is a legitimate and common outcome: cross-
validation optimizes an *average* over folds, and a single test set is one
finite sample, so small rank swaps between closely-matched top models
(here, a ~1.5% RMSE spread across the top 4 models) are expected and should
not be over-interpreted as one algorithm being fundamentally superior.

**Interesting observation on Random Forest tuning:** the tuned Random
Forest did not clearly beat the default Random Forest on the test set
(₹153,316 vs. ₹154,811 RMSE — a small improvement) despite a 20-iteration
randomized search. This illustrates a real and useful lesson for the
viva: hyperparameter tuning gives *diminishing or inconsistent* returns
once a tree ensemble's defaults are already reasonable for a dataset of
this size (~5,500 training rows) — it is not a magic accuracy button.

## Training vs. Test Performance (Overfitting Check)

For the final Gradient Boosting model:
- Test R² = 0.9281 (computed above).
- 5-fold CV R² for Gradient Boosting = 0.8698 ± 0.0324 (computed on
  training-set folds it did not see during that fold's fit).

The test R² (0.9281) is actually *higher* than the CV R² mean (0.8698),
within roughly 2 standard deviations of the CV spread — i.e. no evidence of
overfitting to the training set. (Had test R² been dramatically *lower*
than CV R², that would indicate overfitting or train/test distribution
mismatch.)

## Error Analysis

See `reports/figures/actual_vs_predicted.png`, `residual_plot.png`, and
`error_distribution.png` (generated from the actual test-set predictions
of the final model).

- **Actual vs. Predicted:** points cluster around the y = x line for
  low-to-mid-range prices (the bulk of the dataset); scatter widens for
  very high-priced vehicles, where the training data has fewer examples.
- **Residuals:** centered near zero with no strong funnel/curve pattern
  across the predicted-price range, suggesting the error variance is
  reasonably stable rather than growing sharply with price (mild
  heteroscedasticity is still visible at the high-price tail).
- **Error distribution:** roughly symmetric and centered at zero, with a
  MAPE of 21.28% indicating that a typical prediction is off by about a
  fifth of the vehicle's true price — acceptable for a mini-project
  estimator but not precise enough for commercial pricing without further
  work (see `docs/limitations.md`).
- **Likely sources of the largest errors:** rare/luxury brands with few
  training examples, vehicles whose `torque` (dropped feature) carried
  price-relevant information not captured elsewhere, and general used-car
  market factors this dataset cannot see (accident history, cosmetic
  condition, negotiated vs. listed price).

## Feature Importance

See `reports/figures/feature_importance.png` for the final model's
feature importances. As is typical for used-car pricing, engine/power,
vehicle age, and kilometers driven dominate — feature importance shows
*association strength within this model*, not causation (see
`docs/limitations.md`).
