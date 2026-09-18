"""
Regression evaluation metrics and diagnostic plots.

Kept separate from train.py so the Streamlit app's "Model Performance" page
can reuse the exact same metric definitions instead of re-implementing them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: safe for scripts/CI, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics.

    MAE  (Mean Absolute Error): average absolute rupee error. Easiest metric
         to explain to a non-technical audience.
    RMSE (Root Mean Squared Error): like MAE but squares errors before
         averaging, so large individual errors are penalised more heavily.
         RMSE >= MAE always; the gap indicates how much a few large errors
         (outlier predictions) dominate.
    R2   (Coefficient of Determination): fraction of the variance in
         selling_price explained by the model. 1.0 = perfect, 0.0 = no
         better than predicting the mean, negative = worse than the mean.
    MAPE (Mean Absolute Percentage Error): average error as a percentage of
         the true price -- useful because a ₹20,000 error matters more on a
         ₹100,000 car than a ₹2,000,000 car.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#2563eb", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction (y = x)")
    ax.set_xlabel("Actual Selling Price (₹)")
    ax.set_ylabel("Predicted Selling Price (₹)")
    ax.set_title("Actual vs Predicted Selling Price (Test Set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_pred, residuals, alpha=0.4, s=15, color="#059669", edgecolors="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Selling Price (₹)")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residual Plot (Test Set)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(residuals, bins=50, color="#7c3aed", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Prediction Error (Actual − Predicted, ₹)")
    ax.set_ylabel("Number of Vehicles")
    ax.set_title("Prediction Error Distribution (Test Set)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(feature_names: list[str], importances: np.ndarray, save_path: Path, top_n: int = 20) -> None:
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#ea580c",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {min(top_n, len(feature_names))} Feature Importances")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
