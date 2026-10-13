# 🛒 Rossmann Sales Forecasting

## Business Question

**How much will each Rossmann store sell per day over the next few weeks, and which forecasting model should we trust for each store?**

Rossmann runs 3,000+ drug stores across Europe. Inaccurate demand forecasts lead to overstaffing, stockouts, and poorly timed promotions. This project builds and compares four forecasting approaches on the [Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data) data, and selects the best-performing model per store using rolling-window backtesting.

## Overview

The pipeline runs in five stages, each in its own notebook:

```
Raw data → SQL EDA → Feature engineering → Classical models ─┐
                                         └→ Pooled LSTM ─────┼→ Rolling backtest → Per-store model registry
                                                             ┘
```

Data is stored in SQLite and queried with SQL (SQLAlchemy), including window functions for lag and rolling features. The code is written so the same queries can later point at a hosted Postgres database by changing only the connection string.

## Key Findings

| Finding | Evidence |
|---|---|
| **Promotions drive sales** | Promo days average 8,228 vs 5,929 on non-promo days, a **+38.8% lift** |
| **Strong weekly seasonality** | Mondays and Sundays are highest, Saturdays lowest |
| **December peak** | Clear holiday-season spike in both years |
| **Holidays boost sales** | State-holiday open days average 8,487–9,888 vs 6,954 on normal days |
| **Customers track sales closely** | Strongest correlation with sales; average basket is about 9.49 per customer |
| **Large store-to-store spread** | Average daily sales range from ~2,700 (bottom) to ~21,700 (top) |
| **Store type matters** | Type `b` stores with extra or extended assortment have the highest average sales |
| **No single model wins everywhere** | Best model differs by store (see results below) |
| **Simple models are hard to beat** | Per-store SARIMA/Prophet matched or outperformed a pooled LSTM on this sample |

## Dataset

- **1,017,209** daily records · **1,115** stores · **2013-01-01 → 2015-07-31**
- No duplicate `(store, date)` pairs; 844,392 open-day records after removing closed days
- Missing values in `stores.csv` are structural (no competitor recorded, or store not in Promo2), not random
- Gap found: 181 stores have no records for roughly Jul 2014 – Dec 2014 (758 of 942 days)
- Data quality checks: no closed days with sales, and only 54 open days with zero sales

### EDA notes worth knowing

- **Sales are right-skewed**, so a `log1p` target was prepared alongside raw sales.
- **Promo2 stores average lower sales** than non-Promo2 stores (6,558 vs 7,351). This compares store segments, not promo effect, so it should not be read as Promo2 hurting sales.
- **Closer competitors did not mean lower sales.** Stores with a competitor under 1 km averaged the highest sales, likely because competitors cluster in busy areas.
- **Sunday trading is rare**, which is why most closures cluster on Sundays.

## Project Workflow

| # | Notebook | What it does |
|---|---|---|
| 1 | `01_eda.ipynb` | SQL-driven EDA: data quality, seasonality, holidays, promos, store segments, closures |
| 2 | `02_feature_engineering.ipynb` | Builds a model-ready table (813,172 rows × 51 columns), saved to `features.parquet` |
| 3 | `03_modeling_classical.ipynb` | Seasonal-naive baseline, SARIMA, and Prophet |
| 4 | `04_modeling_lstm.ipynb` | Pooled two-layer LSTM |
| 5 | `05_backtesting_evaluation.ipynb` | 3-fold expanding-window backtest and per-store model selection |

### Feature engineering

| Group | Features |
|---|---|
| **Calendar** | year, month, week of year, quarter, weekend flag, sin/cos encodings of day of week and month |
| **Competition** | months since competitor opened, bucketed distance, missing-info flag (missing distance filled with a large constant) |
| **Promo** | `is_promo2_active` (parsed from `PromoInterval`), `promo_start_flag` |
| **Lags** | sales at 1, 7, 14, 28 days; customers at lag 7 |
| **Rolling stats** | 7- and 30-day mean; 7-day std, min, max |
| **Store segments** | one-hot store type and assortment |
| **Gap handling** | `days_since_prev_open`, since `sales_lag_1` means the last *open* day |

**Leakage control:** lags and rolling windows use only prior rows (`ROWS BETWEEN N PRECEDING AND 1 PRECEDING`), scalers are fit on training data only, and the backtest refits scalers per fold. Rows without enough history for lags (3.7% of data) are dropped.

## Models

| Model | Setup |
|---|---|
| **Seasonal-naive** | Repeats the last 7 days forward; the baseline every other model must beat |
| **SARIMA** | `(1,1,1)(1,1,1,7)` with `promo` and `school_holiday` as exogenous regressors; refit per store and fold |
| **Prophet** | Weekly and yearly seasonality with `promo` and `school_holiday` regressors; refit per store and fold |
| **LSTM** | One global model across stores: LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1); 14-day lookback, 27 input features, Adam with MSE loss, early stopping |

## Evaluation Approach

An initial single split (last 6 weeks held out) suggested different winners per store, but one split is a noisy basis for choosing a model. Notebook 05 therefore uses an **expanding-window backtest**:

| Fold | Train up to | Validate on |
|---|---|---|
| 3 | 2015-06-19 | 2015-06-20 → 2015-07-03 |
| 2 | 2015-07-03 | 2015-07-04 → 2015-07-17 |
| 1 | 2015-07-17 | 2015-07-18 → 2015-07-31 |

Models are compared on mean MAPE across folds, and fold-to-fold standard deviation is checked so an unstable model is not chosen on average alone.

## Results

Scores are **mean MAPE (%)** across the 3 folds, on a 5-store sample (stores 1, 2, 85, 262, 733) chosen to span different store types.

| Store | Seasonal-Naive | SARIMA | Prophet | LSTM | **Selected** |
|---|---|---|---|---|---|
| 1 | 19.87 | **7.41** | 8.20 | 9.67 | SARIMA |
| 2 | 29.10 | 15.18 | **11.62** | 12.47 | Prophet |
| 85 | 16.51 | **7.20** | 7.27 | 8.02 | SARIMA |
| 262 | 9.45 | **5.72** | 7.74 | 8.24 | SARIMA |
| 733 | 11.17 | 7.93 | **6.28** | 6.86 | Prophet |
| **Average** | 17.22 | 8.69 | 8.22 | 9.05 | **7.65** |

**Takeaways**

- Every trained model roughly **halves the error** of the seasonal-naive baseline.
- Picking the best model per store (avg MAPE **7.65%**) beats any single model applied everywhere.
- SARIMA was the most stable across folds in most stores (lowest fold-to-fold variance); the LSTM was the least stable.
- The pooled LSTM was not selected for any store, likely because it was trained on only ~4,000 sequences from 5 stores.
- The winning model per store is saved to `models/model_registry.json`, so a serving layer can load the right model for a given `store_id`.

## Repository Structure

```
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_classical.ipynb
│   ├── 04_modeling_lstm.ipynb
│   └── 05_backtesting_evaluation.ipynb
├── src/
│   └── evaluation/
│       ├── metrics.py          # evaluate_forecast (MAE, RMSE, MAPE)
│       └── backtest.py         # rolling-window generation and summaries
├── data/
│   └── processed/              # rossmann.db, features.parquet
├── models/                     # saved SARIMA/Prophet/LSTM artifacts + model_registry.json
└── README.md
```

## Tech Stack

Python · pandas · NumPy · SQLAlchemy / SQLite · statsmodels (SARIMAX) · Prophet · TensorFlow / Keras · scikit-learn · matplotlib · seaborn


## Limitations

- Model comparison covers **5 of 1,115 stores**; results may not generalize to all store types.
- Backtest windows are short (14 days) and all fall in summer 2015, so Christmas and other seasonal peaks are untested.
- SARIMA orders `(1,1,1)(1,1,1,7)` were not grid-searched; the LSTM was trained for 30 epochs in backtesting.
- Metrics are MAE, RMSE, and MAPE, not the Kaggle competition's RMSPE.