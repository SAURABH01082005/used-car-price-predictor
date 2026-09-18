# SDLC Documentation

**Methodology used:** Agile (iterative sprints), appropriate for a mini
project of this scope where requirements (which features the dataset
actually supports, which algorithms perform best) were partially
discovered during development rather than fully known upfront.

## Sprint Plan

### Sprint 1 — Requirement Analysis + Dataset Acquisition
- Defined problem statement and course-outcome alignment (CO1–CO4).
- Identified and validated a real, public dataset (CarDekho used-car
  listings, "Car details v3.csv", Kaggle `nehalbirla/vehicle-dataset-from-cardekho`).
- Inspected actual raw columns and built the column-mapping config layer
  (`src/config.py::RAW_COLUMN_MAP`) so the project is not hard-wired to one
  dataset's exact headers.
- **Deliverable:** `data/raw/used_cars.csv`, `src/config.py`, `src/data_loader.py`.

### Sprint 2 — Preprocessing + Exploratory Data Analysis
- Implemented unit-stripping (`"1248 CC"` → `1248.0`), duplicate removal,
  invalid-target filtering, and basic sanity-range filtering.
- Built the `ColumnTransformer`-based preprocessing pipeline
  (imputation + scaling + one-hot encoding).
- Explored distributions, correlations, and price relationships in
  `notebooks/01_eda.ipynb`.
- **Deliverable:** `src/preprocessing.py`, `notebooks/01_eda.ipynb`,
  `reports/figures/*` (EDA plots).

### Sprint 3 — Feature Engineering + Model Development
- Engineered `car_age`, `brand` (extracted from free-text `name`), and
  rare-brand grouping.
- Implemented and trained five baseline candidate regressors (Linear,
  Ridge, Decision Tree, Random Forest, Gradient Boosting) plus XGBoost
  (available in this environment).
- **Deliverable:** `src/feature_engineering.py`, `src/train.py`
  (baseline training stage).

### Sprint 4 — Evaluation + Hyperparameter Tuning
- Implemented 5-fold cross-validation comparison across all candidates.
- Ran `RandomizedSearchCV` (20 iterations, bounded search space) on the two
  strongest tree-based candidates.
- Performed a single, final held-out test-set evaluation and selected the
  final model objectively by lowest test RMSE.
- **Deliverable:** `reports/cv_results.csv`, `reports/model_comparison.csv`,
  `models/used_car_price_model.pkl`, `models/model_metadata.json`.

### Sprint 5 — GUI + Testing
- Built the 5-page Streamlit application (`app.py`): Home, Price Prediction,
  Data Analysis, Model Performance, About.
- Implemented input validation (`src/predict.py`) covering negative values,
  out-of-range years, unknown categories, missing fields.
- Wrote 40 automated `pytest` tests across data, preprocessing, model, and
  prediction/reliability layers.
- **Deliverable:** `app.py`, `tests/*.py`, `reports/test_cases.md`,
  `reports/reliability_test.md`.

### Sprint 6 — Documentation + Final Packaging
- Wrote system design, methodology, limitations, future scope, literature
  survey, viva questions, and presentation outline.
- Wrote `README.md`, `requirements.txt`, and this SDLC document.
- **Deliverable:** everything under `docs/`, `README.md`,
  `reports/results.md`.

## SDLC Phase Mapping

| Phase | Activities | Artifacts |
|---|---|---|
| Requirement Analysis | Problem definition, dataset validation, feature-list scoping | `docs/project_proposal.md` |
| System Design | Architecture, layered design, data flow | `docs/system_design.md` |
| Implementation | `src/` modules, `app.py` | Source code |
| Testing | Unit tests, reliability tests, edge-case tests | `tests/`, `reports/test_cases.md`, `reports/reliability_test.md` |
| Deployment | Local Streamlit run (`streamlit run app.py`) | `README.md` (run instructions) |
| Maintenance | Re-run `python -m src.train` to retrain if the dataset changes; config-driven column mapping supports dataset swaps without code changes | `src/config.py` |
