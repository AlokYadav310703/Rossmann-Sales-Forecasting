"""
Shared forecast evaluation metrics.
Used across 03_modeling_classical.ipynb, 04_modeling_lstm.ipynb, and
05_backtesting_evaluation.ipynb so every model is scored identically.
"""

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1e-6) -> float:
    """Mean Absolute Percentage Error, expressed as a percentage.

    Rows where y_true == 0 are excluded (a store closed / zero-sales day makes
    percentage error undefined or explosive) rather than silently distorting
    the metric with a near-infinite term.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > epsilon
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_forecast(y_true, y_pred) -> dict:
    """Convenience wrapper returning all three metrics at once."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def evaluate_forecast_df(df: pd.DataFrame, actual_col: str, pred_col: str) -> dict:
    """Same as evaluate_forecast, but takes a DataFrame + column names."""
    return evaluate_forecast(df[actual_col], df[pred_col])
