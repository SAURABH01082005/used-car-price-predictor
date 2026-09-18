"""Model loading and artifact tests (TC-M series)."""

import joblib
import pytest
from sklearn.pipeline import Pipeline

from src.config import MODEL_COMPARISON_PATH, MODEL_METADATA_PATH, MODEL_PATH
from src.predict import ModelNotFoundError, load_model, load_model_metadata


def test_model_file_exists():
    """TC-M01: trained model pipeline file exists (run 'python -m src.train' first)."""
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}. Run: python -m src.train"


def test_model_loads_as_sklearn_pipeline():
    """TC-M02: the saved artifact loads and is a sklearn Pipeline (preprocessing + model bundled)."""
    load_model.cache_clear()
    model = load_model()
    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "model" in model.named_steps


def test_missing_model_file_raises_useful_error(tmp_path, monkeypatch):
    """TC-M06: a missing model file raises ModelNotFoundError with actionable guidance."""
    import src.predict as predict_module
    load_model.cache_clear()
    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "nonexistent.pkl")
    with pytest.raises(ModelNotFoundError, match="Run 'python -m src.train'"):
        predict_module.load_model()
    load_model.cache_clear()


def test_corrupted_model_file_raises_error(tmp_path, monkeypatch):
    """TC-M07: a corrupted (non-pickle) model file fails loudly rather than silently."""
    import src.predict as predict_module
    bad_model_path = tmp_path / "corrupted.pkl"
    bad_model_path.write_text("this is not a valid pickle file")
    load_model.cache_clear()
    monkeypatch.setattr(predict_module, "MODEL_PATH", bad_model_path)
    with pytest.raises(Exception):
        predict_module.load_model()
    load_model.cache_clear()


def test_model_metadata_file_exists_and_has_final_model_name():
    """TC-M03: metadata JSON exists and records which model was selected."""
    assert MODEL_METADATA_PATH.exists()
    metadata = load_model_metadata()
    assert "final_model_name" in metadata
    assert metadata["final_model_name"]


def test_model_comparison_report_generated():
    """TC-M04: model_comparison.csv exists and lists more than one candidate model."""
    assert MODEL_COMPARISON_PATH.exists()
    import pandas as pd
    df = pd.read_csv(MODEL_COMPARISON_PATH)
    assert len(df) >= 5  # Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting (+ optional XGBoost/tuned variants)
    assert "RMSE" in df.columns and "R2" in df.columns


def test_final_model_meets_minimum_quality_bar():
    """TC-M05: the selected final model's test R2 is meaningfully better than a
    mean-only baseline (R2 > 0.5), i.e. training actually learned signal."""
    metadata = load_model_metadata()
    assert metadata["test_metrics"]["R2"] > 0.5
