"""
Streamlit dashboard for the Rossmann Sales Forecasting API.

Run locally with:
    streamlit run frontend/app.py

Expects the FastAPI backend to be running (default: http://127.0.0.1:8000).
Set API_BASE_URL as an environment variable to point at a deployed backend
instead (e.g. your Render URL) without changing any code.
"""

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Rossmann Sales Forecasting", layout="wide")

st.title("🛒 Rossmann Store Sales Forecasting")
st.caption(
    "Forecasts served by a FastAPI backend using models selected via "
    "rolling-window backtesting (seasonal-naive, SARIMA, Prophet, or a pooled LSTM)."
)


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_models():
    resp = requests.get(f"{API_BASE_URL}/models", timeout=15)
    resp.raise_for_status()
    return resp.json()["stores"]


@st.cache_data(ttl=60)
def fetch_forecast(store_id: int, horizon: int, model: str = None):
    params = {"store_id": store_id, "horizon": horizon}
    if model:
        params["model"] = model
    resp = requests.get(f"{API_BASE_URL}/forecast", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def fetch_history(store_id: int, days: int = 60):
    resp = requests.get(f"{API_BASE_URL}/history", params={"store_id": store_id, "days": days}, timeout=15)
    resp.raise_for_status()
    return resp.json()["history"]


def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
health = check_api_health()
if health is None:
    st.error(
        f"⚠️ Can't reach the API at `{API_BASE_URL}`. "
        "Make sure it's running (`uvicorn api.main:app --reload`) or check `API_BASE_URL`."
    )
    st.stop()
elif health["status"] != "ok":
    st.warning(f"API reports degraded status: {health}")

try:
    store_models = fetch_models()
except Exception as e:
    st.error(f"Failed to load available stores from the API: {e}")
    st.stop()

if not store_models:
    st.warning("No trained models found. Run notebooks 03, 04, and 05 first to populate model_registry.json.")
    st.stop()

store_lookup = {s["store_id"]: s for s in store_models}
store_ids = sorted(store_lookup.keys())

with st.sidebar:
    st.header("⚙️ Forecast Settings")
    selected_store = st.selectbox("Store", store_ids, format_func=lambda x: f"Store {x}")
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, step=7)
    compare_mode = st.checkbox("Compare against seasonal-naive baseline", value=True)

    st.divider()
    st.caption(
        f"**Note:** only {len(store_ids)} stores currently have trained "
        "SARIMA/Prophet/LSTM models (the sample used during development). "
        "All other stores would fall back to the seasonal-naive model in "
        "a full production rollout."
    )


# ------------------------------------------------------------------
# Fetch data for the selected store
# ------------------------------------------------------------------
selected_model_info = store_lookup[selected_store]

try:
    forecast_data = fetch_forecast(selected_store, horizon)
    history_data = fetch_history(selected_store, days=60)
except Exception as e:
    st.error(f"Failed to fetch forecast: {e}")
    st.stop()

forecast_df = pd.DataFrame(forecast_data["forecast"])
forecast_df["date"] = pd.to_datetime(forecast_df["date"])
history_df = pd.DataFrame(history_data)
history_df["date"] = pd.to_datetime(history_df["date"])

if compare_mode:
    try:
        naive_data = fetch_forecast(selected_store, horizon, model="naive")
        naive_df = pd.DataFrame(naive_data["forecast"])
        naive_df["date"] = pd.to_datetime(naive_df["date"])
    except Exception:
        naive_df = None
else:
    naive_df = None


# ------------------------------------------------------------------
# KPI row
# ------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

forecasted_total = forecast_df["predicted_sales"].sum()
recent_actual_total = history_df["sales"].tail(horizon).sum()
pct_change = ((forecasted_total - recent_actual_total) / recent_actual_total * 100) if recent_actual_total else 0

col1.metric(f"Forecasted next {horizon} days", f"₹{forecasted_total:,.0f}")
col2.metric(f"Actual last {horizon} days", f"₹{recent_actual_total:,.0f}", f"{pct_change:+.1f}%")
col3.metric("Model used", forecast_data["model_used"].upper())
col4.metric(
    "Backtest MAPE",
    f"{forecast_data['backtest_mape']:.2f}%" if forecast_data.get("backtest_mape") else "N/A",
)

st.divider()

# ------------------------------------------------------------------
# Main chart: history + forecast (+ optional naive comparison)
# ------------------------------------------------------------------
st.subheader(f"Store {selected_store} — Sales History & Forecast")

chart_df = pd.concat([
    history_df.assign(series="Actual (history)"),
    forecast_df[["date", "predicted_sales"]].rename(columns={"predicted_sales": "sales"}).assign(
        series=f"Forecast ({forecast_data['model_used'].upper()})"
    ),
])

if naive_df is not None:
    chart_df = pd.concat([
        chart_df,
        naive_df[["date", "predicted_sales"]].rename(columns={"predicted_sales": "sales"}).assign(
            series="Forecast (Seasonal-Naive baseline)"
        ),
    ])

pivot_df = chart_df.pivot(index="date", columns="series", values="sales")
st.line_chart(pivot_df)

# Confidence band as a separate area, since st.line_chart doesn't support bands directly
with st.expander("📊 Forecast detail table (incl. confidence interval)"):
    st.dataframe(forecast_df.set_index("date"), use_container_width=True)

st.divider()

# ------------------------------------------------------------------
# Model performance tab
# ------------------------------------------------------------------
st.subheader("📈 Model Performance Across Stores")
st.caption("Backtest results from rolling-window cross-validation (05_backtesting_evaluation.ipynb)")

perf_df = pd.DataFrame(store_models)
perf_df = perf_df.rename(columns={
    "store_id": "Store ID",
    "selected_model": "Selected Model",
    "backtest_mean_mape": "Backtest Mean MAPE (%)",
})
st.dataframe(
    perf_df[["Store ID", "Selected Model", "Backtest Mean MAPE (%)"]].set_index("Store ID"),
    use_container_width=True,
)

st.bar_chart(perf_df.set_index("Store ID")["Backtest Mean MAPE (%)"])

st.divider()
st.caption(
    "⚠️ **Known limitation:** future `promo`/`school_holiday` values are unknown at "
    "forecast time and currently default to 0 (no promo, not a holiday). "
    "If a promo is already scheduled for the forecast window, actual sales "
    "will likely exceed this forecast."
)
