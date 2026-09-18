"""
Feature engineering.

Each engineered feature is deliberately simple and justified individually
(see docstrings) rather than generated in bulk -- an academic mini project
should be able to explain every feature in a viva.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import CURRENT_YEAR, RARE_BRAND_THRESHOLD

logger = logging.getLogger(__name__)


def add_car_age(df: pd.DataFrame) -> pd.DataFrame:
    """Car_Age = CURRENT_YEAR - manufacturing year.

    Why: buyers reason about a car's age, not its absolute manufacturing
    year. Age also has a monotonic (roughly) relationship with depreciation,
    which is easier for linear models to exploit than the raw year value.
    """
    df = df.copy()
    df["car_age"] = CURRENT_YEAR - df["year"]
    df["car_age"] = df["car_age"].clip(lower=0)
    return df


def add_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Brand as the first token of the free-text `name` column.

    Why: `name` (e.g. "Maruti Swift Dzire VDI") is too high-cardinality to
    one-hot encode directly (thousands of unique trims). The brand alone is
    a strong, low-cardinality price signal and is trivial to extract
    reliably since every listing in this dataset starts with the brand.
    """
    df = df.copy()
    df["brand"] = df["name"].astype(str).str.split().str[0].str.title()
    return df


def group_rare_brands(df: pd.DataFrame, threshold: int = RARE_BRAND_THRESHOLD) -> pd.DataFrame:
    """Group brands with fewer than `threshold` listings into 'Other'.

    Why: rare brands (e.g. a single-listing exotic marque) give one-hot
    columns that are almost always zero and whose learned coefficient is
    estimated from very few examples -- grouping them reduces variance and
    keeps the one-hot feature space small and stable.
    """
    df = df.copy()
    counts = df["brand"].value_counts()
    rare_brands = counts[counts < threshold].index
    n_grouped = df["brand"].isin(rare_brands).sum()
    df.loc[df["brand"].isin(rare_brands), "brand"] = "Other"
    logger.info("Grouped %d rows across %d rare brands into 'Other'", n_grouped, len(rare_brands))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full, ordered feature-engineering sequence to a cleaned DataFrame."""
    df = add_car_age(df)
    df = add_brand(df)
    df = group_rare_brands(df)
    return df
