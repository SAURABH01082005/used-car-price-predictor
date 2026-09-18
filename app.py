"""
Streamlit application: Used Car Price Prediction & Analysis System.

Run with:  streamlit run app.py

This file is intentionally a thin presentation layer -- all data cleaning,
feature engineering, model training and metric computation live in src/.
The app only loads already-generated artifacts (trained pipeline, metadata,
comparison CSVs, dataset) and renders them.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    CURRENT_YEAR,
    CV_RESULTS_PATH,
    FIGURES_DIR,
    MODEL_COMPARISON_PATH,
    RAW_DATA_PATH,
)
from src.data_loader import DatasetNotFoundError, load_raw_data
from src.feature_engineering import engineer_features
from src.predict import (
    InputValidationError,
    ModelNotFoundError,
    VALID_FUEL_TYPES,
    VALID_OWNER_TYPES,
    VALID_SELLER_TYPES,
    VALID_TRANSMISSIONS,
    load_model_metadata,
    predict_price,
)
from src.preprocessing import clean_raw_data

st.set_page_config(
    page_title="Used Car Price Prediction & Analysis System",
    page_icon="🚗",
    layout="wide",
)


@st.cache_data
def get_processed_dataset() -> pd.DataFrame:
    raw_df = load_raw_data()
    clean_df = clean_raw_data(raw_df)
    return engineer_features(clean_df)


def format_inr(amount: float) -> str:
    """Format a rupee amount into Lakh notation, e.g. 625000 -> '₹6.25 Lakh'."""
    if amount >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Crore"
    if amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.2f} Lakh"
    return f"₹{amount:,.0f}"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🚗 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Price Prediction", "Data Analysis", "Model Performance", "About Project"],
)

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("Used Car Price Prediction & Analysis System")
    st.caption("HOCO492 — Mini Project: Artificial Intelligence and Machine Learning")

    st.markdown("""
    ### Project Description
    This application predicts the estimated selling price of a used car from
    its characteristics (age, mileage, engine, fuel type, transmission, and
    more), using a supervised machine-learning regression model trained on
    real CarDekho used-car listing data.

    ### Problem Statement
    Used-car pricing is inconsistent and hard to estimate manually — buyers
    and sellers rely on guesswork or dealer quotes that vary widely. This
    project builds a data-driven price estimator and an accompanying
    analytics dashboard to make that process transparent and explainable.

    ### Objectives
    - Predict a used car's selling price from its listed attributes.
    - Compare multiple classical ML regression algorithms objectively.
    - Provide an interactive analytics dashboard for dataset exploration.
    - Demonstrate the complete ML lifecycle: preprocessing → EDA → feature
      engineering → training → validation → tuning → deployment.

    ### ML Approach
    Supervised regression. Five classical algorithms were trained and
    compared via 5-fold cross-validation, the two strongest candidates were
    hyperparameter-tuned with `RandomizedSearchCV`, and the final model was
    selected on held-out test-set performance (see **Model Performance**).

    ### Technology Stack
    Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib/seaborn,
    Streamlit, joblib.
    """)

    st.markdown("### Workflow")
    cols = st.columns(6)
    steps = ["Raw Data", "Preprocessing", "Feature\nEngineering", "Model\nTraining", "Evaluation", "Prediction"]
    for c, step in zip(cols, steps):
        c.info(step)

# ---------------------------------------------------------------------------
# PRICE PREDICTION
# ---------------------------------------------------------------------------
elif page == "Price Prediction":
    st.title("Price Prediction")
    st.write("Enter the vehicle's details below to get an estimated selling price.")

    try:
        df = get_processed_dataset()
        brands = sorted(df["brand"].unique())
    except DatasetNotFoundError as e:
        st.error(str(e))
        brands = ["Maruti"]

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            brand = st.selectbox("Car Brand", brands)
            year = st.number_input("Manufacturing Year", min_value=1980, max_value=CURRENT_YEAR, value=2017, step=1)
            km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1_000_000, value=50000, step=1000)
            seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
        with col2:
            fuel = st.selectbox("Fuel Type", sorted(VALID_FUEL_TYPES))
            transmission = st.selectbox("Transmission", sorted(VALID_TRANSMISSIONS))
            seller_type = st.selectbox("Seller Type", sorted(VALID_SELLER_TYPES))
            owner = st.selectbox("Owner Type", list(VALID_OWNER_TYPES))
        with col3:
            mileage = st.number_input("Mileage (kmpl)", min_value=0.1, max_value=50.0, value=18.0, step=0.1)
            engine = st.number_input("Engine Capacity (CC)", min_value=50, max_value=8000, value=1200, step=50)
            max_power = st.number_input("Max Power (bhp)", min_value=1.0, max_value=1000.0, value=80.0, step=1.0)

        submitted = st.form_submit_button("Predict Price", type="primary")

    if submitted:
        input_data = {
            "brand": brand, "year": year, "km_driven": km_driven, "fuel": fuel,
            "seller_type": seller_type, "transmission": transmission, "owner": owner,
            "mileage": mileage, "engine": engine, "max_power": max_power, "seats": seats,
        }
        try:
            result = predict_price(input_data)
        except ModelNotFoundError as e:
            st.error(str(e))
        except InputValidationError as e:
            st.warning(f"Invalid input: {e}")
        else:
            st.success("Prediction generated successfully.")
            c1, c2 = st.columns(2)
            c1.metric("Estimated Selling Price", format_inr(result["predicted_price"]))
            c2.metric(
                "Approximate Price Range",
                f"{format_inr(result['price_range_low'])} – {format_inr(result['price_range_high'])}",
            )
            st.caption(result["range_method"])

            st.markdown("#### Prediction Factors")
            for f in result["factors"]:
                st.write(f"- {f}")

# ---------------------------------------------------------------------------
# DATA ANALYSIS
# ---------------------------------------------------------------------------
elif page == "Data Analysis":
    st.title("Data Analysis Dashboard")

    try:
        df = get_processed_dataset()
    except DatasetNotFoundError as e:
        st.error(str(e))
        st.stop()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1]}")
    c3.metric("Avg Price", format_inr(df["selling_price"].mean()))
    c4.metric("Median Price", format_inr(df["selling_price"].median()))
    c5.metric("Min Price", format_inr(df["selling_price"].min()))
    c6.metric("Max Price", format_inr(df["selling_price"].max()))

    st.markdown("### Price Distribution")
    st.plotly_chart(px.histogram(df, x="selling_price", nbins=60, title="Selling Price Distribution"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Price vs Manufacturing Year")
        st.plotly_chart(px.scatter(df, x="year", y="selling_price", opacity=0.4, trendline=None), use_container_width=True)
    with col2:
        st.markdown("### Price vs Kilometers Driven")
        st.plotly_chart(px.scatter(df, x="km_driven", y="selling_price", opacity=0.4), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Price by Fuel Type")
        st.plotly_chart(px.box(df, x="fuel", y="selling_price"), use_container_width=True)
    with col4:
        st.markdown("### Price by Transmission")
        st.plotly_chart(px.box(df, x="transmission", y="selling_price"), use_container_width=True)

    st.markdown("### Price by Owner Type")
    st.plotly_chart(px.box(df, x="owner", y="selling_price"), use_container_width=True)

    st.markdown("### Correlation Heatmap (Numerical Features)")
    numeric_cols = ["selling_price", "car_age", "km_driven", "mileage", "engine", "max_power", "seats"]
    corr = df[numeric_cols].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)

# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    if not MODEL_COMPARISON_PATH.exists():
        st.error(f"No model comparison results found at {MODEL_COMPARISON_PATH}. Run 'python -m src.train' first.")
        st.stop()

    comparison_df = pd.read_csv(MODEL_COMPARISON_PATH)
    metadata = load_model_metadata()

    if metadata:
        st.success(f"Final selected model: **{metadata.get('final_model_name', 'N/A')}** "
                    f"(selected objectively by lowest test-set RMSE among {len(metadata.get('models_compared', []))} candidates)")

    st.markdown("### Test-Set Model Comparison")
    st.dataframe(comparison_df.style.format({"MAE": "{:,.0f}", "MSE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.4f}", "MAPE": "{:.2f}%"}), use_container_width=True)

    if CV_RESULTS_PATH.exists():
        st.markdown("### 5-Fold Cross-Validation Results (Training Set)")
        cv_df = pd.read_csv(CV_RESULTS_PATH)
        st.dataframe(cv_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### RMSE Comparison")
        st.plotly_chart(px.bar(comparison_df.sort_values("RMSE"), x="Model", y="RMSE"), use_container_width=True)
    with col2:
        st.markdown("### R² Comparison")
        st.plotly_chart(px.bar(comparison_df.sort_values("R2", ascending=False), x="Model", y="R2"), use_container_width=True)

    st.markdown("### Diagnostic Plots (Final Model)")
    fig_cols = st.columns(2)
    figure_files = [
        ("actual_vs_predicted.png", "Actual vs Predicted"),
        ("residual_plot.png", "Residual Plot"),
        ("error_distribution.png", "Error Distribution"),
        ("feature_importance.png", "Feature Importance"),
    ]
    for i, (fname, caption) in enumerate(figure_files):
        fpath = FIGURES_DIR / fname
        if fpath.exists():
            fig_cols[i % 2].image(str(fpath), caption=caption, use_container_width=True)

# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif page == "About Project":
    st.title("About This Project")
    st.markdown("""
    **Project Title:** Used Car Price Prediction and Analysis System Using Machine Learning
    **Course:** HOCO492 — Mini Project: Artificial Intelligence and Machine Learning

    ### Problem Statement
    Estimating a fair used-car price is difficult without domain expertise;
    this project uses supervised machine learning to produce a data-driven
    estimate from vehicle attributes.

    ### Objectives
    - Build a complete, reproducible ML pipeline (preprocessing → EDA →
      feature engineering → model comparison → tuning → deployment).
    - Demonstrate objective model selection using cross-validation and
      held-out test performance, not assumption.
    - Deliver an interactive, explainable prediction interface.

    ### Methodology
    CRISP-DM-style pipeline: data cleaning, feature engineering (car age,
    brand extraction, rare-category grouping), an 80/20 train/test split,
    5-fold cross-validation on training data, `RandomizedSearchCV`
    hyperparameter tuning on the strongest candidates, and a single
    final evaluation on the untouched test set.

    ### Algorithms Compared
    Linear Regression, Ridge Regression, Decision Tree, Random Forest,
    Gradient Boosting, XGBoost.

    ### Technology Stack
    Python 3, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn,
    Plotly, Streamlit, joblib, pytest.

    ### Dataset
    CarDekho used-car listings ("Car details v3.csv", Kaggle dataset
    `nehalbirla/vehicle-dataset-from-cardekho`).

    ### Limitations
    - Dataset reflects Indian used-car listings at time of scraping — prices
      will not generalize to other regions or currencies without retraining.
    - No real-time market data; prices drift over time as the market moves.
    - Vehicle condition (accident history, cosmetic state) is not captured.
    - `torque` was dropped due to inconsistent free-text formatting; some
      price signal may be lost.
    - Rare brands are grouped into "Other", losing brand-specific detail
      for low-volume manufacturers.

    ### Future Scope
    Larger / real-time datasets, regional pricing models, calibrated
    prediction intervals, SHAP-based explainability, REST API deployment,
    condition assessment from images.
    """)
