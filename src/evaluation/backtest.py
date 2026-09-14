"""
Rolling-window backtesting utilities.

A single train/validation split (used in 03_modeling_classical.ipynb and
04_modeling_lstm.ipynb) tells you how a model did on ONE holdout period --
which could just be luck. Rolling-window backtesting repeats the
train/predict/evaluate cycle across several holdout periods, giving a much
more reliable read on which model actually performs best.

Used by: 05_backtesting_evaluation.ipynb
"""

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class BacktestFold:
    fold_id: int
    train_end: pd.Timestamp     # last date included in training for this fold
    val_start: pd.Timestamp     # first date of the validation window
    val_end: pd.Timestamp       # last date of the validation window


def generate_rolling_windows(max_date: pd.Timestamp, n_folds: int,
                               horizon_days: int) -> List[BacktestFold]:
    """
    Builds `n_folds` non-overlapping backtest windows, walking backward from
    `max_date`. Fold 1 (most recent) uses the most training data; each
    earlier fold has a shorter training set (an "expanding window" backtest,
    the standard approach for time series -- never train on future data
    relative to that fold's validation period).

    Example with n_folds=3, horizon_days=14, max_date=2015-07-31:
        Fold 1: train up to 2015-07-17, validate 2015-07-18 -> 2015-07-31
        Fold 2: train up to 2015-07-03, validate 2015-07-04 -> 2015-07-17
        Fold 3: train up to 2015-06-19, validate 2015-06-20 -> 2015-07-03
    """
    folds = []
    for i in range(n_folds):
        val_end = max_date - pd.Timedelta(days=i * horizon_days)
        val_start = val_end - pd.Timedelta(days=horizon_days - 1)
        train_end = val_start - pd.Timedelta(days=1)
        folds.append(BacktestFold(fold_id=i + 1, train_end=train_end,
                                   val_start=val_start, val_end=val_end))
    return list(reversed(folds))  # chronological order: oldest fold first


def summarize_backtest_results(results_df: pd.DataFrame, group_cols: list,
                                 metric_col: str = "MAPE") -> pd.DataFrame:
    """Aggregates per-fold metric results into mean/std/min/max per group
    (e.g. per store per model), for comparing model stability across folds."""
    return (
        results_df.groupby(group_cols)[metric_col]
        .agg(["mean", "std", "min", "max", "count"])
        .rename(columns={"mean": f"{metric_col}_mean", "std": f"{metric_col}_std",
                          "min": f"{metric_col}_min", "max": f"{metric_col}_max",
                          "count": "n_folds"})
        .reset_index()
    )
