"""
Live inference feature pipeline.

Given a store_id and a forecast horizon, builds the exact feature rows the
trained model expects for each future date -- one day at a time, recursively.

WHY RECURSIVE: lag/rolling features (sales_lag_1, sales_rolling_mean_7, ...)
are computed from *actual* past sales during training. At inference time,
"yesterday's sales" is known for the first forecast day, but by the second
forecast day "yesterday" is itself a prediction, not a fact. This module
fetches real history from the database, then for each successive forecast day:
  1. builds that day's feature row from the history buffer (actual + predicted
     sales so far),
  2. hands the row to the caller's `predict_fn` (the trained model),
  3. appends the returned prediction into the buffer as if it were real,
  4. moves to the next day.

This keeps the module model-agnostic -- it doesn't import SARIMA/Prophet/LSTM
code directly, it just needs a `predict_fn(feature_row: dict) -> float`.

Usage (from the FastAPI service):
    from src.features.inference_pipeline import InferenceFeatureBuilder

    builder = InferenceFeatureBuilder(engine)
    forecast_df = builder.forecast(store_id=1, horizon_days=14, predict_fn=my_model.predict_one)
"""

import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.features.transformations import (
    add_calendar_features,
    impute_competition_distance,
    add_competition_features,
    add_promo_features,
    encode_store_categoricals,
    add_competition_distance_bucket,
)

FEATURE_METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, "feature_metadata.json")

# Needs to cover the largest lag/rolling window used in training (lag_28, rolling_mean_30)
HISTORY_LOOKBACK_DAYS = 60


class InferenceFeatureBuilder:
    def __init__(self, engine, metadata_path: str = FEATURE_METADATA_PATH):
        self.engine = engine
        with open(metadata_path) as f:
            self.metadata = json.load(f)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------
    def _fetch_store_static(self, store_id: int) -> dict:
        """Store-level metadata that doesn't change day to day."""
        query = f"""
        SELECT store_type, assortment, competition_distance,
               competition_open_since_month, competition_open_since_year,
               promo2, promo2_since_week, promo2_since_year, promo_interval
        FROM stores
        WHERE store_id = {store_id};
        """
        row = pd.read_sql(query, self.engine)
        if row.empty:
            raise ValueError(f"store_id {store_id} not found in stores table")
        return row.iloc[0].to_dict()

    def _fetch_recent_history(self, store_id: int) -> pd.DataFrame:
        """Last HISTORY_LOOKBACK_DAYS of *actual* sales for this store -- the
        real data the recursive lag/rolling calculation is seeded with."""
        query = f"""
        SELECT date, sales, customers
        FROM sales_daily
        WHERE store_id = {store_id} AND open = true
        ORDER BY date DESC
        LIMIT {HISTORY_LOOKBACK_DAYS};
        """
        df = pd.read_sql(query, self.engine)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Feature construction for a single future date
    # ------------------------------------------------------------------
    def _build_row(self, store_id: int, target_date: pd.Timestamp,
                    store_static: dict, history: pd.DataFrame,
                    promo_calendar: dict) -> dict:
        """Build one feature row for `target_date`, using `history` (actual +
        already-predicted sales up to but not including target_date)."""

        row = {"store_id": store_id, "date": target_date, **store_static}
        row["promo"] = promo_calendar.get(target_date.date(), 0)
        row["school_holiday"] = 0  # caller can override via promo_calendar/holiday_calendar if known in advance
        row["state_holiday"] = "0"

        df_row = pd.DataFrame([row])
        df_row = add_calendar_features(df_row)
        df_row = impute_competition_distance(
            df_row, far_away_constant=self.metadata["far_away_competition_distance_constant"]
        )
        df_row = add_competition_features(df_row)
        df_row = add_promo_features(df_row)
        df_row = encode_store_categoricals(
            df_row,
            known_store_types=tuple(self.metadata["known_store_types"]),
            known_assortments=tuple(self.metadata["known_assortments"]),
        )
        df_row = add_competition_distance_bucket(df_row)

        # Lag / rolling features computed from the history buffer (mix of real
        # + already-predicted sales), NOT from SQL -- there's no future data
        # in the database to query yet.
        sales_series = history.set_index("date")["sales"]
        df_row["sales_lag_1"] = sales_series.get(target_date - timedelta(days=1), np.nan)
        df_row["sales_lag_7"] = sales_series.get(target_date - timedelta(days=7), np.nan)
        df_row["sales_lag_14"] = sales_series.get(target_date - timedelta(days=14), np.nan)
        df_row["sales_lag_28"] = sales_series.get(target_date - timedelta(days=28), np.nan)

        customers_series = history.set_index("date")["customers"] if "customers" in history.columns else pd.Series(dtype=float)
        df_row["customers_lag_7"] = customers_series.get(target_date - timedelta(days=7), np.nan)

        window_7 = sales_series[(sales_series.index < target_date) &
                                 (sales_series.index >= target_date - timedelta(days=7))]
        window_30 = sales_series[(sales_series.index < target_date) &
                                  (sales_series.index >= target_date - timedelta(days=30))]

        df_row["sales_rolling_mean_7"] = window_7.mean() if len(window_7) else np.nan
        df_row["sales_rolling_std_7"] = window_7.std() if len(window_7) else np.nan
        df_row["sales_rolling_min_7"] = window_7.min() if len(window_7) else np.nan
        df_row["sales_rolling_max_7"] = window_7.max() if len(window_7) else np.nan
        df_row["sales_rolling_mean_30"] = window_30.mean() if len(window_30) else np.nan

        return df_row.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forecast(self, store_id: int, horizon_days: int, predict_fn,
                 promo_calendar: dict = None) -> pd.DataFrame:
        """
        Recursively build features and predict `horizon_days` days ahead.

        Args:
            store_id: which store to forecast for
            horizon_days: how many days ahead to forecast
            predict_fn: callable(feature_row: dict) -> float (predicted sales)
                        Supplied by the caller -- keeps this module decoupled
                        from any specific model implementation.
            promo_calendar: optional {date: 0/1} map if future promo activity
                        is already known/planned; defaults to no promo (0)
                        for all forecast dates when not provided.

        Returns:
            DataFrame with columns [date, predicted_sales]
        """
        store_static = self._fetch_store_static(store_id)
        history = self._fetch_recent_history(store_id)
        promo_calendar = promo_calendar or {}

        if history.empty:
            raise ValueError(f"No historical sales found for store_id {store_id}")

        last_date = history["date"].max()
        results = []

        for step in range(1, horizon_days + 1):
            target_date = last_date + timedelta(days=step)
            feature_row = self._build_row(store_id, target_date, store_static, history, promo_calendar)

            prediction = predict_fn(feature_row)

            # Feed the prediction back into the history buffer so the NEXT
            # day's lag/rolling features can see it, exactly as if it were real.
            history = pd.concat([
                history,
                pd.DataFrame([{"date": target_date, "sales": prediction, "customers": np.nan}])
            ], ignore_index=True)

            results.append({"date": target_date, "predicted_sales": prediction})

        return pd.DataFrame(results)
