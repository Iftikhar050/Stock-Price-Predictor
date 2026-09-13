# main.py
# ---------------------------------------------------------
# FastAPI Backend - Stock Predictor API
#
# Slim app entrypoint: CORS + lifespan (ML model loading) + router wiring.
# Endpoint implementations live in api/routers/*.py.
# ---------------------------------------------------------
import os
import sys
import logging
import joblib
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the root is in path for imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.psx_predictor.models.train_lstm import LSTMModel
from src.psx_predictor.api.state import ml_models, psx_cache
from src.psx_predictor.api.routers import predict, company, market, screener, macro

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
if not logger.handlers:
    ch = logging.StreamHandler()
    logger.addHandler(ch)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads both Random Forest and LSTM models into memory.
    """
    try:
        # Load RF
        ml_models["rf_predictor"] = joblib.load(os.path.join(MODELS_DIR, "baseline_rf_model.pkl"))
        print("[SUCCESS] Random Forest model loaded successfully.")

        # Load LR
        ml_models["lr_predictor"] = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        print("[SUCCESS] Linear Regression model loaded successfully.")

        # Load XGBoost
        ml_models["xgb_predictor"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
        print("[SUCCESS] XGBoost model loaded successfully.")

        # Load Scalers for LSTM
        ml_models["feature_scaler"] = joblib.load(os.path.join(MODELS_DIR, "feature_scaler_lstm.pkl"))
        ml_models["target_scaler"] = joblib.load(os.path.join(MODELS_DIR, "target_scaler_lstm.pkl"))
        ml_models["ticker_to_id"] = joblib.load(os.path.join(MODELS_DIR, "ticker_to_id.pkl"))
        ml_models["sector_to_id"] = joblib.load(os.path.join(MODELS_DIR, "sector_to_id.pkl"))

        # Load XGBoost Categories
        ml_models["xgb_ticker_categories"] = joblib.load(os.path.join(MODELS_DIR, "xgb_ticker_categories.pkl"))
        ml_models["xgb_sector_categories"] = joblib.load(os.path.join(MODELS_DIR, "xgb_sector_categories.pkl"))

        # Load LSTM
        # We dynamically get the input_dim from the loaded scaler and embeddings from state_dict
        device = torch.device("cpu")
        input_dim = ml_models["feature_scaler"].n_features_in_

        state_dict = torch.load(os.path.join(MODELS_DIR, "lstm_model.pth"), map_location=device, weights_only=True)
        num_tickers = state_dict['ticker_embed.weight'].shape[0]
        num_sectors = state_dict['sector_embed.weight'].shape[0]

        lstm = LSTMModel(input_dim=input_dim, num_tickers=num_tickers, num_sectors=num_sectors, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
        lstm.load_state_dict(state_dict)
        lstm.eval()
        ml_models["lstm_predictor"] = lstm
        print("[SUCCESS] LSTM model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
    yield
    ml_models.clear()


app = FastAPI(
    title="Stock Predictor API",
    description="Serves ML predictions for stock prices using Random Forest and LSTM.",
    version="1.0.0",
    lifespan=lifespan
)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_env.split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(company.router)
app.include_router(market.router)
app.include_router(screener.router)
app.include_router(macro.router)
