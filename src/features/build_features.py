"""
Batch feature engineering pipeline.

Rebuilds the full model-ready feature table from the database and saves it to
data/processed/features.parquet -- this is the production equivalent of
notebooks/02_feature_engineering.ipynb, runnable as a standalone script (e.g.
on a schedule, whenever new sales data lands in the database) rather than by
hand in Jupyter.

Run with:
    python -m src.features.build_features
"""

import json
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.config import DATABASE_URL, PROCESSED_DATA_DIR
from src.features.transformations import (
    add_calendar_features,
    impute_competition_distance,
    add_competition_features,
    add_promo_features,
    add_promo_start_flag,
    encode_store_categoricals,
    add_competition_distance_bucket,
    add_log_target,
)

FEATURE_METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, "feature_metadata.json")
FEATURES_OUTPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "features.parquet")

VALIDATION_WEEKS = 6


def load_base_data(engine) -> pd.DataFrame:
    query = """
    SELECT
        sd.store_id, sd.date, sd.day_of_week, sd.sales, sd.customers, sd.open,
        sd.promo, sd.state_holiday, sd.school_holiday,
        s.store_type, s.assortment, s.competition_distance,
        s.competition_open_since_month, s.competition_open_since_year,
        s.promo2, s.promo2_since_week, s.promo2_since_year, s.promo_interval
    FROM sales_daily sd
    JOIN stores s ON sd.store_id = s.store_id
    WHERE sd.open = true
    ORDER BY sd.store_id, sd.date;
    """
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_lag_rolling_features(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Lag/rolling features via SQL window functions (see sql/feature_engineering.sql
    for the standalone, documented version of these same queries)."""
    lag_query = """
    SELECT
        store_id, date,
        LAG(sales, 1)  OVER (PARTITION BY store_id ORDER BY date) AS sales_lag_1,
        LAG(sales, 7)  OVER (PARTITION BY store_id ORDER BY date) AS sales_lag_7,
        LAG(sales, 14) OVER (PARTITION BY store_id ORDER BY date) AS sales_lag_14,
        LAG(sales, 28) OVER (PARTITION BY store_id ORDER BY date) AS sales_lag_28,
        LAG(customers, 7) OVER (PARTITION BY store_id ORDER BY date) AS customers_lag_7
    FROM sales_daily
    WHERE open = true
    ORDER BY store_id, date;
    """
    rolling_query = """
    SELECT
        store_id, date,
        AVG(sales) OVER (PARTITION BY store_id ORDER BY date
                          ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS sales_rolling_mean_7,
        AVG(sales) OVER (PARTITION BY store_id ORDER BY date
                          ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS sales_rolling_mean_30,
        MIN(sales) OVER (PARTITION BY store_id ORDER BY date
                          ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS sales_rolling_min_7,
        MAX(sales) OVER (PARTITION BY store_id ORDER BY date
                          ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS sales_rolling_max_7
    FROM sales_daily
    WHERE open = true
    ORDER BY store_id, date;
    """

    lag_df = pd.read_sql(lag_query, engine)
    lag_df["date"] = pd.to_datetime(lag_df["date"])
    rolling_df = pd.read_sql(rolling_query, engine)
    rolling_df["date"] = pd.to_datetime(rolling_df["date"])

    df = df.merge(lag_df, on=["store_id", "date"], how="left")
    df = df.merge(rolling_df, on=["store_id", "date"], how="left")

    # Rolling std -- not a single SQL window function in SQLite/Postgres, done in pandas
    df = df.sort_values(["store_id", "date"])
    df["sales_rolling_std_7"] = (
        df.groupby("store_id")["sales"]
          .apply(lambda s: s.shift(1).rolling(7).std())
          .reset_index(level=0, drop=True)
    )
    return df


def build_feature_table(engine) -> tuple[pd.DataFrame, dict]:
    """Runs the full pipeline and returns (feature_df, metadata_dict).

    metadata_dict captures values that MUST be reused identically at inference
    time (e.g. the competition_distance imputation constant, split date, final
    column list) -- saved to feature_metadata.json so the inference pipeline
    and the training pipeline never drift apart.
    """
    df = load_base_data(engine)

    df = add_calendar_features(df)

    far_away_constant = float(df["competition_distance"].max() * 1.5)
    df = impute_competition_distance(df, far_away_constant=far_away_constant)

    df = add_competition_features(df)
    df = add_promo_features(df)
    df = add_promo_start_flag(df)
    df = add_lag_rolling_features(df, engine)
    df = encode_store_categoricals(df)
    df = add_competition_distance_bucket(df)
    df = add_log_target(df)

    split_date = df["date"].max() - pd.Timedelta(weeks=VALIDATION_WEEKS)
    df["split"] = np.where(df["date"] <= split_date, "train", "validation")

    lag_roll_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
    before = len(df)
    df = df.dropna(subset=lag_roll_cols)
    dropped = before - len(df)

    metadata = {
        "far_away_competition_distance_constant": far_away_constant,
        "split_date": str(split_date.date()),
        "validation_weeks": VALIDATION_WEEKS,
        "n_rows_final": len(df),
        "n_rows_dropped_for_lag_warmup": dropped,
        "feature_columns": [c for c in df.columns if c not in ("sales", "log_sales", "split")],
        "known_store_types": ["a", "b", "c", "d"],
        "known_assortments": ["a", "b", "c"],
    }
    return df, metadata


def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    engine = create_engine(DATABASE_URL)

    print("Building feature table...")
    df, metadata = build_feature_table(engine)

    df.to_parquet(FEATURES_OUTPUT_PATH, index=False)
    with open(FEATURE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(df)} rows -> {FEATURES_OUTPUT_PATH}")
    print(f"Saved feature metadata -> {FEATURE_METADATA_PATH}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()