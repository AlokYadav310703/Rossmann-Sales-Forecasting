"""
FastAPI application entrypoint.

Run locally with:
    uvicorn backend.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import forecast

app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="Serves demand forecasts per store using backtest-selected "
                 "SARIMA, Prophet, LSTM, or seasonal-naive models.",
    version="1.0.0",
)

# Allow the Streamlit frontend (running on a different port/host) to call this API.
# Restrict allow_origins to your actual frontend's URL before deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router)


@app.get("/")
def root():
    return {
        "message": "Rossmann Sales Forecasting API",
        "docs": "/docs",
        "endpoints": ["/health", "/models", "/forecast"],
    }
