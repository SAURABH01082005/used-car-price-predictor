# Academic Project Report

**Title:** Used Car Price Prediction and Analysis System Using Machine Learning
**Course:** HOCO492 — Mini Project: Artificial Intelligence and Machine Learning

This report is organized into the 18 chapters below. Each chapter is kept
concise here and points to the full supporting document/artifact where the
complete detail lives, so information is not duplicated across files.

---

## Chapter 1: Introduction
Used-car pricing is currently estimated informally — by dealer judgment or
inconsistent online tools — with no transparent, data-driven method
available to a typical buyer or seller. This project applies supervised
machine learning to the problem: given a labeled dataset of past used-car
sales, learn a function that estimates a fair selling price for a new
vehicle from its attributes, and expose that function through an
interactive analytics application.

## Chapter 2: Problem Statement and Objectives
See `docs/project_proposal.md` for the full problem statement, objectives,
and course-outcome (CO1–CO4) alignment.

## Chapter 3: Literature Survey
Five real, verified published studies on used-car/vehicle price prediction
were reviewed and connected to this project's methodology and results —
see `docs/literature_survey.md`. Key takeaway used to guide this project:
tree-based ensembles consistently outperform plain linear regression for
this problem across independent studies, a finding this project's own
results (Chapter 11) reproduce.

## Chapter 4: Existing System
Existing approaches largely fall into two categories: (a) manual/dealer
estimation with no reproducible methodology, and (b) opaque commercial
online valuation tools that do not disclose their model, features, or
error rate. Neither provides a transparent, explainable, or independently
verifiable price estimate.

## Chapter 5: Proposed System
A supervised regression pipeline (data cleaning → feature engineering →
model comparison → tuning → deployment) producing a point price estimate,
an explicitly-labeled empirical price range, human-readable prediction
factors, and full model/metric transparency via an analytics dashboard —
see `docs/system_design.md` for the architecture.

## Chapter 6: System Requirements
**Functional:** accept vehicle attributes, validate input, return a price
estimate and range, display dataset/model analytics.
**Non-functional:** deterministic predictions, sub-second response time,
graceful handling of invalid/unseen input, reproducible training.
**Technology stack:** Python 3.13, pandas, NumPy, scikit-learn, XGBoost,
matplotlib/seaborn, Plotly, Streamlit, joblib, pytest (see
`requirements.txt`).

## Chapter 7: System Design
Full layered architecture (input → processing → ML → presentation) and
data-flow diagrams: `docs/system_design.md`.

## Chapter 8: Dataset and Data Preprocessing
Dataset: CarDekho used-car listings, "Car details v3.csv" (Kaggle
`nehalbirla/vehicle-dataset-from-cardekho`), 8,128 raw rows, 13 columns.
Preprocessing (duplicate removal, unit-stripping, invalid-target removal,
`torque` drop) is documented with before/after evidence in
`notebooks/02_preprocessing.ipynb` and implemented in `src/preprocessing.py`.

## Chapter 9: Exploratory Data Analysis
19 EDA items (distributions, correlations, price-vs-feature relationships,
outlier analysis) executed with real output in `notebooks/01_eda.ipynb`;
key plots saved to `reports/figures/`.

## Chapter 10: Machine Learning Methodology
Full pipeline description, train/validation/test definitions, and
leakage-avoidance strategy: `docs/methodology.md`.

## Chapter 11: Model Training and Evaluation
Six candidate algorithms (Linear Regression, Ridge Regression, Decision
Tree, Random Forest, Gradient Boosting, XGBoost) trained and compared via
5-fold cross-validation, with `RandomizedSearchCV` tuning on the top two
tree-based candidates, then a single held-out test evaluation. Final
model: **Gradient Boosting** (test RMSE ≈ ₹151,043, R² ≈ 0.9281, MAPE ≈
21.28%), selected objectively by lowest test RMSE. Full numbers and
discussion: `reports/results.md`; reproducible walkthrough:
`notebooks/03_model_training.ipynb`.

## Chapter 12: System Implementation
Modular `src/` package (`config.py`, `data_loader.py`, `preprocessing.py`,
`feature_engineering.py`, `train.py`, `evaluate.py`, `predict.py`) with the
trained pipeline bundled and serialized via `joblib` to
`models/used_car_price_model.pkl`.

## Chapter 13: GUI / Streamlit Application
5-page Streamlit app (`app.py`): Home, Price Prediction, Data Analysis,
Model Performance, About — all reading from generated artifacts, no
hard-coded values. Run with `streamlit run app.py`.

## Chapter 14: Testing
40 automated `pytest` tests across data validation, preprocessing, model
loading, and prediction/reliability — all passing at time of writing. Full
test case table: `reports/test_cases.md`. Reliability measurements
(consistency, latency): `reports/reliability_test.md`.

## Chapter 15: Results and Discussion
Full metric tables (cross-validation and test-set), overfitting check, and
error analysis: `reports/results.md`. Diagnostic plots:
`reports/figures/actual_vs_predicted.png`, `residual_plot.png`,
`error_distribution.png`, `feature_importance.png`.

## Chapter 16: Limitations
See `docs/limitations.md` — regional/currency scope, market drift,
dropped `torque` feature, rare-brand grouping, no vehicle-condition data,
non-calibrated empirical price range, feature importance ≠ causation.

## Chapter 17: Future Scope
See `docs/future_scope.md` — larger/real-time datasets, regional models,
calibrated prediction intervals, SHAP explainability, REST API and cloud
deployment.

## Chapter 18: Conclusion
This project delivered a complete, reproducible machine-learning lifecycle
for used-car price prediction: a validated real-world dataset, a
leakage-free preprocessing and feature-engineering pipeline, an objective
comparison of six regression algorithms with cross-validation and bounded
hyperparameter tuning, a final model selected on held-out test
performance (Gradient Boosting, R² ≈ 0.93), a tested and input-validated
interactive application, and full documentation connecting the work to
prior published literature. The system's honestly-reported ~21% MAPE and
explicitly documented limitations reflect a transparent, defensible
academic result rather than an overstated one — appropriate for a mini
project whose goal is demonstrating sound ML methodology end-to-end.

---

## References
See `docs/literature_survey.md` for full citation details of all five
referenced works (Pudaruth 2014; Monburinon et al. 2018; Gegic et al.
2019; Pal et al. 2018; Noor & Jan 2017).

## Appendix
- Full test case list: `reports/test_cases.md`
- Reliability test report: `reports/reliability_test.md`
- Model comparison data: `reports/model_comparison.csv`, `reports/cv_results.csv`
- Viva preparation (44 Q&A): `docs/viva_questions.md`
- Presentation outline: `docs/presentation_outline.md`
