"""
End-to-end training pipeline.

Run with:  python -m src.train

Pipeline stages (see docs/system_design.md for the full diagram):
  raw CSV -> clean -> feature engineering -> train/test split
  -> 5 candidate models compared via 5-fold cross-validation on the
     training split -> the two strongest candidates are hyperparameter-tuned
     with RandomizedSearchCV (still training-data only) -> every candidate
     (baseline + tuned) is scored ONCE on the held-out test set -> the model
     with the best test RMSE is selected as final -> final pipeline
     (preprocessing + model bundled together) is saved with joblib.

All reported numbers are computed from the actual dataset at run time --
nothing in this file is a hard-coded metric.
"""

from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    ALL_FEATURES,
    CV_FOLDS,
    CV_RESULTS_PATH,
    FIGURES_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_raw_data
from src.evaluate import (
    compute_metrics,
    plot_actual_vs_predicted,
    plot_error_distribution,
    plot_feature_importance,
    plot_residuals,
)
from src.feature_engineering import engineer_features
from src.preprocessing import build_preprocessing_pipeline, clean_raw_data

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_candidate_models() -> dict:
    """Baseline (untuned) candidate models with reasonable default hyperparameters."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    return models


def get_tuning_search_spaces() -> dict:
    """Small, deliberately bounded hyperparameter search spaces (RandomizedSearchCV)."""
    return {
        "Random Forest": {
            "model__n_estimators": [100, 200, 300, 400],
            "model__max_depth": [8, 12, 16, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 1.0],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4, 5],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        },
    }


def prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    raw_df = load_raw_data()
    clean_df = clean_raw_data(raw_df)
    feat_df = engineer_features(clean_df)

    X = feat_df[ALL_FEATURES]
    y = feat_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train set: %d rows | Test set: %d rows", len(X_train), len(X_test))

    PROCESSED_TRAIN = REPORTS_DIR.parent / "data" / "processed" / "train.csv"
    PROCESSED_TEST = REPORTS_DIR.parent / "data" / "processed" / "test.csv"
    PROCESSED_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(PROCESSED_TRAIN, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_TEST, index=False)

    return X_train, y_train, X_test, y_test


def cross_validate_models(models: dict, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """5-fold cross-validation on the TRAINING split only (no test-set peeking)."""
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            ("model", model),
        ])
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2", "mae": "neg_mean_absolute_error"},
            n_jobs=-1,
        )
        row = {
            "Model": name,
            "CV_RMSE_mean": -scores["test_rmse"].mean(),
            "CV_RMSE_std": scores["test_rmse"].std(),
            "CV_R2_mean": scores["test_r2"].mean(),
            "CV_R2_std": scores["test_r2"].std(),
            "CV_MAE_mean": -scores["test_mae"].mean(),
        }
        rows.append(row)
        logger.info(
            "%s | CV RMSE = %.0f (+/- %.0f) | CV R2 = %.4f",
            name, row["CV_RMSE_mean"], row["CV_RMSE_std"], row["CV_R2_mean"],
        )

    return pd.DataFrame(rows).sort_values("CV_RMSE_mean").reset_index(drop=True)


def tune_model(name: str, model, param_space: dict, X_train: pd.DataFrame, y_train: pd.Series):
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("model", model),
    ])
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_space,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    logger.info("Tuning %s with RandomizedSearchCV (20 iterations, %d-fold CV)...", name, CV_FOLDS)
    search.fit(X_train, y_train)
    logger.info("%s tuned. Best CV RMSE = %.0f | Best params = %s", name, -search.best_score_, search.best_params_)
    return search.best_estimator_, search.best_params_, -search.best_score_


def main() -> None:
    start_time = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = prepare_data()

    # --- Stage 1: baseline candidate models, compared via cross-validation ---
    candidate_models = get_candidate_models()
    cv_results = cross_validate_models(candidate_models, X_train, y_train)
    cv_results.to_csv(CV_RESULTS_PATH, index=False)
    logger.info("Cross-validation results saved to %s", CV_RESULTS_PATH)

    # --- Stage 2: hyperparameter-tune the two strongest tree-based candidates ---
    search_spaces = get_tuning_search_spaces()
    top_for_tuning = [m for m in cv_results["Model"].tolist() if m in search_spaces][:2]
    logger.info("Selected for hyperparameter tuning: %s", top_for_tuning)

    tuned_pipelines = {}
    for name in top_for_tuning:
        best_pipeline, best_params, best_cv_rmse = tune_model(
            name, candidate_models[name], search_spaces[name], X_train, y_train
        )
        tuned_pipelines[f"{name} (Tuned)"] = best_pipeline

    # --- Stage 3: fit every baseline candidate on the full training set too,
    #     so ALL candidates (baseline + tuned) get a fair, identical test-set
    #     evaluation. Tuned pipelines are already fit (refit=True above). ---
    all_pipelines = {}
    for name, model in candidate_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        all_pipelines[name] = pipeline
    all_pipelines.update(tuned_pipelines)

    # --- Stage 4: single, final evaluation on the untouched test set ---
    test_rows = []
    for name, pipeline in all_pipelines.items():
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test.to_numpy(), y_pred)
        test_rows.append({"Model": name, **metrics})
        logger.info("[TEST] %s | MAE=%.0f RMSE=%.0f R2=%.4f MAPE=%.2f%%", name, metrics["MAE"], metrics["RMSE"], metrics["R2"], metrics["MAPE"])

    comparison_df = pd.DataFrame(test_rows).sort_values("RMSE").reset_index(drop=True)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    logger.info("Model comparison table saved to %s", MODEL_COMPARISON_PATH)

    # --- Stage 5: select final model objectively (lowest test RMSE) ---
    final_model_name = comparison_df.iloc[0]["Model"]
    final_pipeline = all_pipelines[final_model_name]
    y_pred_final = final_pipeline.predict(X_test)
    final_metrics = compute_metrics(y_test.to_numpy(), y_pred_final)
    logger.info("FINAL SELECTED MODEL: %s | Test RMSE=%.0f R2=%.4f", final_model_name, final_metrics["RMSE"], final_metrics["R2"])

    # --- Stage 6: diagnostic plots for the final model ---
    y_test_arr = y_test.to_numpy()
    plot_actual_vs_predicted(y_test_arr, y_pred_final, FIGURES_DIR / "actual_vs_predicted.png")
    plot_residuals(y_test_arr, y_pred_final, FIGURES_DIR / "residual_plot.png")
    plot_error_distribution(y_test_arr, y_pred_final, FIGURES_DIR / "error_distribution.png")

    # Empirical residual quantiles on the test set, used by the Streamlit app
    # to show an approximate price RANGE (not a statistically calibrated
    # prediction interval -- explicitly labelled as such in the UI). Method:
    # residual = actual - predicted on the test set; the 10th/90th percentile
    # of that distribution gives an empirical "80% of test vehicles fell
    # within this offset of the prediction" band.
    residuals = y_test_arr - y_pred_final
    residual_low = float(np.percentile(residuals, 10))
    residual_high = float(np.percentile(residuals, 90))

    final_model_step = final_pipeline.named_steps["model"]
    if hasattr(final_model_step, "feature_importances_"):
        feature_names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()
        plot_feature_importance(
            list(feature_names), final_model_step.feature_importances_, FIGURES_DIR / "feature_importance.png"
        )
        logger.info("Feature importance plot saved.")

    # --- Stage 7: persist final pipeline + metadata ---
    joblib.dump(final_pipeline, MODEL_PATH)
    logger.info("Final model pipeline saved to %s", MODEL_PATH)

    metadata = {
        "final_model_name": final_model_name,
        "trained_at": pd.Timestamp.now().isoformat(),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "features": ALL_FEATURES,
        "test_metrics": final_metrics,
        "residual_p10": residual_low,
        "residual_p90": residual_high,
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "training_duration_seconds": round(time.time() - start_time, 1),
        "models_compared": list(candidate_models.keys()),
        "models_tuned": top_for_tuning,
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Model metadata saved to %s", MODEL_METADATA_PATH)

    logger.info("Training pipeline complete in %.1f seconds.", time.time() - start_time)


if __name__ == "__main__":
    main()
