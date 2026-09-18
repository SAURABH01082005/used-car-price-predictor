# Presentation Outline (10–12 Slides)

**Slide 1 — Title**
Used Car Price Prediction and Analysis System Using Machine Learning
HOCO492 Mini Project · [Your Name] · [Date]

**Slide 2 — Problem Statement**
Used-car pricing is inconsistent and hard to estimate without domain
expertise or access to comparable listings. Need: a data-driven price
estimator.

**Slide 3 — Objectives**
Predict selling price from vehicle attributes; compare multiple ML
algorithms objectively; provide an analytics dashboard; demonstrate the
full ML lifecycle end-to-end.

**Slide 4 — Existing vs. Proposed System**
Existing: manual estimation, dealer quotes, inconsistent online
calculators with opaque methodology.
Proposed: transparent, reproducible ML pipeline with documented
preprocessing, objective model comparison, and an explainable prediction
interface (approximate range explicitly labeled, not a fake confidence
score).

**Slide 5 — System Architecture**
User → Streamlit UI → Input Validation → Feature Engineering →
Preprocessing Pipeline → Trained ML Model → Prediction → Result
Visualization. (Diagram from `docs/system_design.md`.)

**Slide 6 — Dataset**
CarDekho used-car listings (Kaggle `nehalbirla/vehicle-dataset-from-cardekho`),
8,128 raw rows → 6,924 after cleaning, 13 raw columns, target =
`selling_price`.

**Slide 7 — Preprocessing + EDA**
Unit stripping (`"1248 CC"` → `1248`), duplicate/invalid-row removal,
`ColumnTransformer` (imputation + scaling + one-hot encoding). Key EDA
findings: price vs. age, price vs. km_driven, correlation heatmap
(`reports/figures/`).

**Slide 8 — ML Algorithms**
Linear Regression, Ridge Regression, Decision Tree, Random Forest,
Gradient Boosting, XGBoost — 5-fold cross-validation, then
`RandomizedSearchCV` tuning on the top two tree-based candidates.

**Slide 9 — Model Comparison**
Table/chart from `reports/model_comparison.csv`: final model = Gradient
Boosting, test RMSE ≈ ₹151,043, R² ≈ 0.928, MAPE ≈ 21.3%. Show the
RMSE/R² bar charts.

**Slide 10 — Application Demo**
Live or screenshot walkthrough: Home → Price Prediction (form + result) →
Data Analysis dashboard → Model Performance page.

**Slide 11 — Testing + Results**
40 automated tests (data/preprocessing/model/prediction + reliability),
all passing. Reliability: 100% consistent predictions across 50 runs,
~3ms latency, 0% crash rate (`reports/reliability_test.md`).

**Slide 12 — Conclusion + Future Scope**
Delivered a complete, reproducible ML lifecycle project with an
interpretable, tested prediction system. Future work: real-time data,
regional models, calibrated prediction intervals, REST API/cloud
deployment (`docs/future_scope.md`).
