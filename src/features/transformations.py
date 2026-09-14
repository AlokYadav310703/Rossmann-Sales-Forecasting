"""
Pure, stateless feature transformations shared by both:
  - the batch training pipeline (src/features/build_features.py)
  - the live inference pipeline (src/features/inference_pipeline.py)

Keeping these functions here (instead of duplicating logic in both places)
guarantees training and serving compute features identically -- avoiding the
classic "training/serving skew" bug where a model is trained on one feature
definition and served on a slightly different one.

None of these functions touch the database or do lag/rolling calculations
(those need historical context and live in build_features.py / inference_pipeline.py).
"""

import numpy as np
import pandas as pd

MONTH_MAP = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
             7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Year/month/day/quarter/weekend flags + cyclical day-of-week/month encodings."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek + 1  # 1=Mon ... 7=Sun, matches Rossmann convention
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def impute_competition_distance(df: pd.DataFrame, far_away_constant: float = None) -> pd.DataFrame:
    """Fill missing competition_distance with a large constant ('no nearby competitor').

    far_away_constant should be computed ONCE from the training set and reused
    at inference time -- passing it in explicitly (rather than recomputing
    max()*1.5 on whatever data is present) keeps train/serve consistent.
    """
    df = df.copy()
    df["has_competition_info"] = df["competition_distance"].notna().astype(int)
    if far_away_constant is None:
        far_away_constant = df["competition_distance"].max() * 1.5
    df["competition_distance"] = df["competition_distance"].fillna(far_away_constant)
    return df


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """months_since_competition_open, vectorized (no row-wise .apply)."""
    df = df.copy()
    comp_date = pd.to_datetime(
        dict(year=df["competition_open_since_year"].fillna(df["date"].dt.year),
             month=df["competition_open_since_month"].fillna(df["date"].dt.month),
             day=1),
        errors="coerce",
    )
    months = (df["date"].dt.year - comp_date.dt.year) * 12 + (df["date"].dt.month - comp_date.dt.month)
    months = months.where(df["competition_open_since_year"].notna(), 0)
    df["months_since_competition_open"] = months.clip(lower=0).fillna(0)
    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """is_promo2_active flag, vectorized. promo_start_flag needs row order within
    a store's history and is computed separately in the pipelines that call this."""
    df = df.copy()

    def _row_active(row):
        if row["promo2"] == 0 or pd.isna(row["promo_interval"]):
            return 0
        iso_year, iso_week, _ = row["date"].isocalendar()
        if pd.isna(row.get("promo2_since_year")):
            return 0
        if iso_year < row["promo2_since_year"]:
            return 0
        if iso_year == row["promo2_since_year"] and iso_week < row["promo2_since_week"]:
            return 0
        active_months = row["promo_interval"].split(",")
        return int(MONTH_MAP[row["month"]] in active_months)

    # Only rows with promo2 == 1 need the (relatively slow) per-row check;
    # everything else is trivially 0. Cuts down .apply() cost significantly.
    df["is_promo2_active"] = 0
    mask = df["promo2"] == 1
    if mask.any():
        df.loc[mask, "is_promo2_active"] = df.loc[mask].apply(_row_active, axis=1)
    return df


def add_promo_start_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flags the first day a promo turns on, per store. Requires df sorted by
    (store_id, date) and a full/contiguous history per store to be accurate."""
    df = df.sort_values(["store_id", "date"]).copy()
    df["promo_start_flag"] = (
        (df["promo"] == 1) &
        (df.groupby("store_id")["promo"].shift(1).fillna(0) == 0)
    ).astype(int)
    return df


def encode_store_categoricals(df: pd.DataFrame, known_store_types=("a", "b", "c", "d"),
                                known_assortments=("a", "b", "c")) -> pd.DataFrame:
    """One-hot encode store_type/assortment against a FIXED known category list.

    Using pd.get_dummies directly on live data is dangerous in production: if
    a single inference request only has store_type='a', pd.get_dummies won't
    create columns for b/c/d, and the resulting feature vector won't match
    what the model was trained on. Passing explicit categories avoids this.
    """
    df = df.copy()
    df["store_type"] = pd.Categorical(df["store_type"], categories=known_store_types)
    df["assortment"] = pd.Categorical(df["assortment"], categories=known_assortments)
    df = pd.get_dummies(df, columns=["store_type", "assortment"],
                         prefix=["store_type", "assortment"])
    return df


def add_competition_distance_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["competition_distance_bucket"] = pd.cut(
        df["competition_distance"],
        bins=[-1, 1000, 5000, 20000, np.inf],
        labels=["under_1km", "1_5km", "5_20km", "over_20km"],
    )
    return df


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sales" in df.columns:
        df["log_sales"] = np.log1p(df["sales"])
    return df
