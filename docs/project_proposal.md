# Project Proposal

**Title:** Used Car Price Prediction and Analysis System Using Machine Learning
**Course:** HOCO492 — Mini Project: Artificial Intelligence and Machine Learning

## Problem Statement

Determining a fair price for a used car is difficult without domain
expertise or access to comparable listings. Buyers and sellers commonly
rely on informal estimates or dealer quotes that vary widely for
comparable vehicles. This project builds a supervised machine-learning
system that predicts a used car's selling price from its listed
attributes, plus an interactive dashboard for exploring the underlying
pricing data.

## Objectives

1. Acquire and validate a real, public used-car dataset.
2. Build a leakage-free preprocessing and feature-engineering pipeline.
3. Train and objectively compare multiple classical regression algorithms.
4. Tune the strongest candidates and select a final model on held-out
   test performance.
5. Package the trained pipeline behind an interactive Streamlit interface
   with input validation and an analytics dashboard.
6. Validate the system with automated tests covering data, preprocessing,
   model loading, prediction correctness, and reliability.

## Course Outcome Alignment

- **CO1** (independent investigation, critical analysis): objective model
  comparison via cross-validation and test metrics rather than assumed
  results; explicit overfitting check (`reports/results.md`).
- **CO2** (orderly presentation, open questions): structured reports
  (`reports/results.md`, `reports/test_cases.md`) and documented
  limitations (`docs/limitations.md`) rather than presenting the system as
  a finished, unqualified solution.
- **CO3** (literature linkage): `docs/literature_survey.md` connects this
  project's methodology and results to five real published studies.
- **CO4** (practical implications and constraints): dataset limitations,
  regional scope, and the empirical (not statistically calibrated) price
  range are explicitly documented and surfaced in the UI itself.

## Scope

**In scope:** classical ML regression, a single-region dataset, a local
Streamlit application, automated testing, full documentation.
**Out of scope:** deep learning, real-time data ingestion, multi-region
pricing, cloud deployment, mobile apps (see `docs/future_scope.md`).

## Dataset

CarDekho used-car listings, "Car details v3.csv"
(Kaggle: `nehalbirla/vehicle-dataset-from-cardekho`), 8,128 raw rows, 13
columns. See `src/config.py` for the full column documentation and
`docs/methodology.md` for how it is cleaned and used.

## Deliverables

Trained model (`models/used_car_price_model.pkl`), Streamlit application
(`app.py`), EDA notebook, 40 automated tests, model comparison and
reliability reports, and this full documentation set.
