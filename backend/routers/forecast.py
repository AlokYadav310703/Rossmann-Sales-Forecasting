"""
Forecast API routes.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.schemas import ForecastResponse, ModelsResponse, HealthResponse, HistoryResponse
from backend.services import forecast_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return forecast_service.check_health()


@router.get("/history", response_model=HistoryResponse)
def history(
    store_id: int = Query(..., description="Store ID"),
    days: int = Query(60, ge=1, le=365, description="Number of past days of actual sales to return"),
):
    try:
        return {"store_id": store_id, "history": forecast_service.get_history(store_id, days)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")


@router.get("/models", response_model=ModelsResponse)
def models():
    """Lists every store that has a trained model, its backtest-selected
    model type, and backtest MAPE -- used by the frontend to populate the
    store selector dropdown."""
    return {"stores": forecast_service.list_models()}


@router.get("/forecast", response_model=ForecastResponse)
def forecast(
    store_id: int = Query(..., description="Store ID to forecast"),
    horizon: int = Query(14, ge=1, le=60, description="Number of days ahead to forecast"),
    model: Optional[str] = Query(
        None,
        description="Force a specific model (naive/sarima/prophet/lstm). "
                     "Omit to use the backtest-selected model for this store.",
    ),
):
    try:
        return forecast_service.generate_forecast(store_id, horizon, model_override=model)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {e}")
