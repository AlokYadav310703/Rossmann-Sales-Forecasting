"""
Pydantic request/response models for the forecasting API.
"""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    date: date
    predicted_sales: float
    lower: Optional[float] = Field(None, description="Lower bound of the confidence interval")
    upper: Optional[float] = Field(None, description="Upper bound of the confidence interval")


class ForecastResponse(BaseModel):
    store_id: int
    model_used: str
    horizon_days: int
    backtest_mape: Optional[float] = Field(
        None, description="Mean MAPE for this model on this store from rolling-window backtesting"
    )
    forecast: List[ForecastPoint]


class ModelInfo(BaseModel):
    store_id: int
    selected_model: str
    backtest_mean_mape: Optional[float] = None
    available_models: List[str]


class ModelsResponse(BaseModel):
    stores: List[ModelInfo]


class HistoryPoint(BaseModel):
    date: date
    sales: float


class HistoryResponse(BaseModel):
    store_id: int
    history: List[HistoryPoint]


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    registry_loaded: bool
