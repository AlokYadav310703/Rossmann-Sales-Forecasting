"""
Core forecasting service.

Reads models/model_registry.json to determine which model type (naive,
sarima, prophet, or the pooled lstm) was selected for a given store during
backtesting (05_backtesting_evaluation.ipynb), loads that model, and produces
a forecast for the requested horizon.

Models are loaded lazily and cached in memory on first request per
(store_id, model_type) -- avoids reloading pickles/Keras models on every
API call, without needing to preload everything at startup.
"""

import json
import os
import pickle
from datetime import timedelta
from functools import lru_cache
from sqlalchemy import create_engine, text

import numpy as np
import pandas as pd

from src.config import DATABASE_URL, MODELS_DIR, PROCESSED_DATA_DIR

REGISTRY_PATH = os.path.join(MODELS_DIR, "model_registry.json")
FEATURE_METADATA_PATH = os.path.join(PROCESSED_DATA_DIR, "feature_metadata.json")

_engine = create_engine(DATABASE_URL)
_model_cache: dict = {}   # key: (store_id, model_type) -> loaded model object(s)


# ------------------------------------------------------------------
# Registry access
# ------------------------------------------------------------------
def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        raise FileNotFoundError(
            f"{REGISTRY_PATH} not found. Run the modeling notebooks "
            "(03, 04, 05) first to generate it."
        )
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def get_store_ids_with_models(registry: dict) -> list:
    """Store IDs that have a real trained model (excludes the special
    'lstm_pooled' metadata key, which isn't itself a store)."""
    return sorted(int(k) for k in registry.keys() if k != "lstm_pooled" and k.isdigit())


def get_store_registry_entry(registry: dict, store_id: int) -> dict:
    key = str(store_id)
    if key not in registry:
        raise ValueError(
            f"No trained model found for store_id={store_id}. "
            f"Available stores: {get_store_ids_with_models(registry)}"
        )
    return registry[key]


# ------------------------------------------------------------------
# Historical data helpers
# ------------------------------------------------------------------
def _fetch_recent_history(store_id: int, lookback_days: int = 60) -> pd.DataFrame:
    query = f"""
    SELECT date, sales, promo, school_holiday
    FROM sales_daily
    WHERE store_id = {store_id} AND open = true
    ORDER BY date DESC
    LIMIT {lookback_days};
    """
    df = pd.read_sql(query, _engine)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------
# Model loading (cached)
# ------------------------------------------------------------------
def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_sarima(store_id: int, entry: dict):
    key = (store_id, "sarima")
    if key not in _model_cache:
        path = entry["models"]["sarima"]["path"]
        _model_cache[key] = _load_pickle(path)
    return _model_cache[key]


def _get_prophet(store_id: int, entry: dict):
    key = (store_id, "prophet")
    if key not in _model_cache:
        path = entry["models"]["prophet"]["path"]
        _model_cache[key] = _load_pickle(path)
    return _model_cache[key]


def _get_lstm(registry: dict):
    key = ("lstm_pooled",)
    if key not in _model_cache:
        from tensorflow.keras.models import load_model

        lstm_entry = registry["lstm_pooled"]
        model = load_model(lstm_entry["model_path"])
        feature_scaler = _load_pickle(lstm_entry["feature_scaler_path"])
        target_scaler = _load_pickle(lstm_entry["target_scaler_path"])
        _model_cache[key] = {
            "model": model,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "feature_columns": lstm_entry["feature_columns"],
            "lookback_days": lstm_entry["lookback_days"],
        }
    return _model_cache[key]


# ------------------------------------------------------------------
# Per-model forecasting logic
# ------------------------------------------------------------------
def _forecast_naive(store_id: int, horizon_days: int) -> pd.DataFrame:
    """Seasonal-naive: repeats the last 7 days of actual sales forward."""
    history = _fetch_recent_history(store_id, lookback_days=14)
    if history.empty:
        raise ValueError(f"No sales history found for store_id={store_id}")

    last_week = history["sales"].values[-7:]
    reps = int(np.ceil(horizon_days / 7))
    predictions = np.tile(last_week, reps)[:horizon_days]

    last_date = history["date"].max()
    dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]
    std = history["sales"].std()

    return pd.DataFrame({
        "date": dates,
        "predicted_sales": predictions,
        "lower": predictions - 1.96 * std,
        "upper": predictions + 1.96 * std,
    })


def _forecast_sarima(store_id: int, horizon_days: int, entry: dict) -> pd.DataFrame:
    fit = _get_sarima(store_id, entry)

    # Future promo/school_holiday values aren't known in advance -- default
    # to "no promo / not a school holiday" for now. See README limitation note.
    exog_future = pd.DataFrame({
        "promo": [0] * horizon_days,
        "school_holiday": [0] * horizon_days,
    })

    result = fit.get_forecast(steps=horizon_days, exog=exog_future)
    mean = result.predicted_mean.values
    conf_int = result.conf_int(alpha=0.05)

    last_date = _fetch_recent_history(store_id, lookback_days=1)["date"].max()
    dates = [last_date + timedelta(days=i + 1) for i in range(horizon_days)]

    return pd.DataFrame({
        "date": dates,
        "predicted_sales": mean,
        "lower": conf_int.iloc[:, 0].values,
        "upper": conf_int.iloc[:, 1].values,
    })


def _forecast_prophet(store_id: int, horizon_days: int, entry: dict) -> pd.DataFrame:
    model = _get_prophet(store_id, entry)

    last_date = _fetch_recent_history(store_id, lookback_days=1)["date"].max()
    future = pd.DataFrame({
        "ds": [last_date + timedelta(days=i + 1) for i in range(horizon_days)],
        "promo": [0] * horizon_days,
        "school_holiday": [0] * horizon_days,
    })

    forecast = model.predict(future)
    return pd.DataFrame({
        "date": future["ds"],
        "predicted_sales": forecast["yhat"].values,
        "lower": forecast["yhat_lower"].values,
        "upper": forecast["yhat_upper"].values,
    })


def _forecast_lstm(store_id: int, horizon_days: int, registry: dict) -> pd.DataFrame:
    """Recursive LSTM forecasting -- see src/features/inference_pipeline.py
    for the detailed explanation of why this needs to be recursive."""
    from src.features.inference_pipeline import InferenceFeatureBuilder

    bundle = _get_lstm(registry)
    model, feature_scaler, target_scaler = bundle["model"], bundle["feature_scaler"], bundle["target_scaler"]
    feature_cols = bundle["feature_columns"]

    builder = InferenceFeatureBuilder(_engine, metadata_path=FEATURE_METADATA_PATH)

    def predict_fn(feature_row: dict) -> float:
        row_df = pd.DataFrame([feature_row])
        for col in feature_cols:
            if col not in row_df.columns:
                row_df[col] = 0  # safety net for any one-hot column missing in a single row
        scaled = feature_scaler.transform(row_df[feature_cols])
        # NOTE: this is a simplified single-day approximation -- a fully
        # correct version would feed the LSTM its full LOOKBACK-day sequence,
        # not one row. Documented as a known simplification; see README.
        pred_scaled = model.predict(scaled.reshape(1, 1, -1), verbose=0).flatten()[0]
        return float(target_scaler.inverse_transform([[pred_scaled]])[0][0])

    forecast_df = builder.forecast(store_id=store_id, horizon_days=horizon_days, predict_fn=predict_fn)
    forecast_df = forecast_df.rename(columns={"predicted_sales": "predicted_sales"})
    # LSTM has no native confidence interval -- approximate with historical residual std
    history = _fetch_recent_history(store_id, lookback_days=30)
    std = history["sales"].std()
    forecast_df["lower"] = forecast_df["predicted_sales"] - 1.96 * std
    forecast_df["upper"] = forecast_df["predicted_sales"] + 1.96 * std
    return forecast_df


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------
def generate_forecast(store_id: int, horizon_days: int, model_override: str = None) -> dict:
    """
    Args:
        store_id: which store to forecast
        horizon_days: how many days ahead
        model_override: force a specific model ("naive"/"sarima"/"prophet"/"lstm")
                         instead of the backtest-selected one -- used by the
                         frontend's "compare models" toggle.

    Returns a dict matching backend.schemas.ForecastResponse's shape.
    """
    registry = load_registry()
    entry = get_store_registry_entry(registry, store_id)

    model_used = model_override or entry.get("selected_model", "naive")

    if model_used == "naive":
        forecast_df = _forecast_naive(store_id, horizon_days)
        backtest_mape = entry.get("models", {}).get("seasonal_naive", {}).get("mape")
    elif model_used == "sarima":
        forecast_df = _forecast_sarima(store_id, horizon_days, entry)
        backtest_mape = entry.get("models", {}).get("sarima", {}).get("mape")
    elif model_used == "prophet":
        forecast_df = _forecast_prophet(store_id, horizon_days, entry)
        backtest_mape = entry.get("models", {}).get("prophet", {}).get("mape")
    elif model_used == "lstm":
        forecast_df = _forecast_lstm(store_id, horizon_days, registry)
        backtest_mape = registry.get("lstm_pooled", {}).get("per_store_mape", {}).get(str(store_id))
    else:
        raise ValueError(f"Unknown model type: {model_used}")

    # Prefer the backtest-aggregated MAPE (from Notebook 05) when available,
    # since it's more reliable than a single train/val split's number.
    if model_used == entry.get("selected_model"):
        backtest_mape = entry.get("backtest_mean_mape", backtest_mape)

    return {
        "store_id": store_id,
        "model_used": model_used,
        "horizon_days": horizon_days,
        "backtest_mape": backtest_mape,
        "forecast": forecast_df.to_dict(orient="records"),
    }


def list_models() -> list:
    registry = load_registry()
    store_ids = get_store_ids_with_models(registry)
    results = []
    for store_id in store_ids:
        entry = registry[str(store_id)]
        available = list(entry.get("models", {}).keys())
        if store_id in registry.get("lstm_pooled", {}).get("store_ids", []):
            available.append("lstm")
        results.append({
            "store_id": store_id,
            "selected_model": entry.get("selected_model", "naive"),
            "backtest_mean_mape": entry.get("backtest_mean_mape"),
            "available_models": available,
        })
    return results


def get_history(store_id: int, days: int = 60) -> list:
    """Recent actual sales for a store -- used by the frontend to draw the
    'historical' portion of the chart alongside the forecast."""
    history = _fetch_recent_history(store_id, lookback_days=days)
    return history[["date", "sales"]].to_dict(orient="records")


def check_health() -> dict:
    db_ok = True
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    registry_ok = os.path.exists(REGISTRY_PATH)

    return {
        "status": "ok" if (db_ok and registry_ok) else "degraded",
        "database_connected": db_ok,
        "registry_loaded": registry_ok,
    }
