# main.py
# ---------------------------------------------------------
# FastAPI Backend - Stock Predictor API
# ---------------------------------------------------------
import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
from contextlib import asynccontextmanager
from functools import lru_cache
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the root is in path for imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.psx_predictor.models.train_lstm import LSTMModel

# Global dictionary to hold loaded ML models in memory
ml_models = {}

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
        print("✅ Random Forest model loaded successfully.")
        
        # Load LR
        ml_models["lr_predictor"] = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        print("✅ Linear Regression model loaded successfully.")

        # Load XGBoost
        ml_models["xgb_predictor"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
        print("✅ XGBoost model loaded successfully.")
        
        # Load Scalers for LSTM
        ml_models["feature_scaler"] = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.pkl"))
        ml_models["target_scaler"] = joblib.load(os.path.join(MODELS_DIR, "target_scaler.pkl"))
        
        # Load LSTM
        # We need to know input_dim. It's usually 19 based on the current feature set.
        device = torch.device("cpu")
        lstm = LSTMModel(input_dim=19, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
        lstm.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_model.pth"), map_location=device, weights_only=True))
        lstm.eval()
        ml_models["lstm_predictor"] = lstm
        print("✅ LSTM model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
    yield
    ml_models.clear()

app = FastAPI(
    title="Stock Predictor API",
    description="Serves ML predictions for stock prices using Random Forest and LSTM.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'PSO')", min_length=2)

class DataPoint(BaseModel):
    date: str
    close: float
    rf_pred: float | None = None
    lr_pred: float | None = None
    xgb_pred: float | None = None
    lstm_pred: float | None = None

class PredictionResponse(BaseModel):
    ticker: str
    latest_date: str
    current_price: float
    rf_predicted_price: float
    lr_predicted_price: float
    xgb_predicted_price: float
    lstm_predicted_price: float
    ensemble_min: float
    ensemble_max: float
    confidence_score: float
    historical_data: list[DataPoint] = Field(..., description="Last 30 days of closing prices for chart rendering")

@app.post("/api/predict", response_model=PredictionResponse)
async def get_prediction(payload: PredictionRequest):
    if "rf_predictor" not in ml_models or "lstm_predictor" not in ml_models or "lr_predictor" not in ml_models or "xgb_predictor" not in ml_models:
        raise HTTPException(status_code=503, detail="ML Models are currently unavailable.")
        
    ticker = payload.ticker.lower()
    data_path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Feature data for ticker '{ticker.upper()}' not found.")
        
    try:
        df = pd.read_csv(data_path)
        
        exclude_cols = ['ticker', 'date', 'created_at', 'target_close_t1']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 1. Random Forest Inference (Requires only the last day)
        latest_day_features = df[feature_cols].iloc[-1:]
        rf_prediction = ml_models["rf_predictor"].predict(latest_day_features)[0]
        
        # 1a. Linear Regression Inference
        lr_prediction = ml_models["lr_predictor"].predict(latest_day_features)[0]

        # 1b. XGBoost Inference
        xgb_prediction = ml_models["xgb_predictor"].predict(latest_day_features)[0]
        
        # 2. LSTM Inference (Requires sequence of last 30 days)
        lookback = 30
        if len(df) < lookback:
            raise HTTPException(status_code=400, detail="Not enough historical data for LSTM.")
            
        recent_30_features = df[feature_cols].tail(lookback).values
        scaled_features = ml_models["feature_scaler"].transform(recent_30_features)
        
        tensor_input = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0) # Shape: (1, 30, 19)
        with torch.no_grad():
            lstm_out_scaled = ml_models["lstm_predictor"](tensor_input)
            
        lstm_prediction = ml_models["target_scaler"].inverse_transform(lstm_out_scaled.numpy())[0][0]
        
        # Calculate Ensemble Metrics
        predictions = [rf_prediction, lr_prediction, xgb_prediction, lstm_prediction]
        ensemble_min = float(min(predictions))
        ensemble_max = float(max(predictions))
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        cv = std_pred / mean_pred if mean_pred != 0 else 0
        
        # Map CV to a 0-100% confidence score
        # cv=0.01 (1% variation) -> 90% confidence. cv=0.05 -> 50%
        confidence = float(max(0.0, min(98.0, 100.0 - (cv * 1000))))
        
        # --- Add Historical Predictions ---
        # Vectorized inference for traditional models
        df['rf_pred'] = pd.Series(ml_models["rf_predictor"].predict(df[feature_cols])).shift(1)
        df['lr_pred'] = pd.Series(ml_models["lr_predictor"].predict(df[feature_cols])).shift(1)
        df['xgb_pred'] = pd.Series(ml_models["xgb_predictor"].predict(df[feature_cols])).shift(1)
        
        # Batch sequence generation for LSTM
        all_features_scaled = ml_models["feature_scaler"].transform(df[feature_cols].values)
        lstm_sequences = []
        lstm_valid_indices = []
        for i in range(lookback, len(df)):
            lstm_sequences.append(all_features_scaled[i-lookback:i])
            lstm_valid_indices.append(i)
            
        df['lstm_pred'] = None
        if lstm_sequences:
            batch_tensor = torch.tensor(np.array(lstm_sequences), dtype=torch.float32)
            with torch.no_grad():
                batch_lstm_out = ml_models["lstm_predictor"](batch_tensor)
            batch_lstm_preds = ml_models["target_scaler"].inverse_transform(batch_lstm_out.numpy()).flatten()
            
            # Create a temporary series to hold predictions and shift by 1
            lstm_series = pd.Series(index=df.index, dtype=float)
            lstm_series.loc[lstm_valid_indices] = batch_lstm_preds
            df['lstm_pred'] = lstm_series.shift(1)
            
        # Select final columns and convert to dict
        recent_history_df = df.tail(1825)[['date', 'close', 'rf_pred', 'lr_pred', 'xgb_pred', 'lstm_pred']]
        # Replace NaNs with None for JSON serialization
        recent_history_df = recent_history_df.where(pd.notnull(recent_history_df), None)
        recent_history = recent_history_df.to_dict(orient='records')
        
        return PredictionResponse(
            ticker=payload.ticker.upper(),
            latest_date=str(df['date'].iloc[-1]),
            current_price=round(float(df['close'].iloc[-1]), 2),
            rf_predicted_price=round(float(rf_prediction), 2),
            lr_predicted_price=round(float(lr_prediction), 2),
            xgb_predicted_price=round(float(xgb_prediction), 2),
            lstm_predicted_price=round(float(lstm_prediction), 2),
            ensemble_min=round(ensemble_min, 2),
            ensemble_max=round(ensemble_max, 2),
            confidence_score=round(confidence, 1),
            historical_data=recent_history
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/reload_models")
async def reload_models():
    """Endpoint to hot-reload ML models from disk after retraining."""
    try:
        # Load RF
        ml_models["rf_predictor"] = joblib.load(os.path.join(MODELS_DIR, "baseline_rf_model.pkl"))
        
        # Load LR
        ml_models["lr_predictor"] = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        
        # Load XGBoost
        ml_models["xgb_predictor"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
        
        # Load Scalers for LSTM
        ml_models["feature_scaler"] = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.pkl"))
        ml_models["target_scaler"] = joblib.load(os.path.join(MODELS_DIR, "target_scaler.pkl"))
        
        # Load LSTM
        device = torch.device("cpu")
        lstm = LSTMModel(input_dim=19, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
        lstm.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_model.pth"), map_location=device, weights_only=True))
        lstm.eval()
        ml_models["lstm_predictor"] = lstm
        
        return {"status": "success", "message": "All AI models successfully reloaded into memory!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")

@lru_cache(maxsize=100)
def fetch_psx_company_profile(ticker: str):
    url = f"https://dps.psx.com.pk/company/{ticker.upper()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return None
        
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # 1. Name & Sector
    name_el = soup.select_one('.quote__name')
    sector_el = soup.select_one('.quote__sector')
    name = name_el.text.strip() if name_el else ticker.upper()
    sector = sector_el.text.strip() if sector_el else "Unknown Sector"
    
    # 2. Description
    desc_el = soup.select_one('.profile__item--decription p')
    desc = desc_el.text.strip() if desc_el else "No description available."
    
    # 3. People
    people = []
    trs = soup.select('.profile__item--people tr')
    for tr in trs:
        tds = tr.find_all('td')
        if len(tds) == 2:
            people.append({
                "name": tds[0].text.strip(),
                "role": tds[1].text.strip()
            })
            
    # 4. Other Details (Address, Web, Auditor, etc)
    details = {}
    items = soup.select('.profile__item')
    for item in items:
        heads = item.select('.item__head')
        for head in heads:
            key = head.text.strip()
            # The value is usually in the next <p> tag
            val_node = head.find_next_sibling('p')
            if val_node:
                details[key] = val_node.text.strip()
                
    return {
        "name": name,
        "sector": sector,
        "description": desc,
        "people": people,
        "details": details
    }

@app.get("/api/company/{ticker}")
async def get_company_profile(ticker: str):
    profile = fetch_psx_company_profile(ticker)
    if not profile:
        raise HTTPException(status_code=404, detail="Company profile not found on PSX.")
    return profile

class RealtimePriceResponse(BaseModel):
    ticker: str
    price: float
    change: float
    change_percent: float

@app.get("/api/realtime/{ticker}", response_model=RealtimePriceResponse)
async def get_realtime_price(ticker: str):
    url = f"https://dps.psx.com.pk/company/{ticker.upper()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=404, detail="Company not found on PSX.")
        
    soup = BeautifulSoup(r.text, 'html.parser')
    
    try:
        price_el = soup.select_one('.quote__close')
        price_str = price_el.text.strip().replace('Rs.', '').replace(',', '')
        price = float(price_str)
        
        change_el = soup.select_one('.change__value')
        change_str = change_el.text.strip().replace(',', '')
        change = float(change_str)
        
        percent_el = soup.select_one('.change__percent')
        percent_str = percent_el.text.strip().replace('(', '').replace(')', '').replace('%', '')
        percent = float(percent_str)
        
        return RealtimePriceResponse(
            ticker=ticker.upper(),
            price=price,
            change=change,
            change_percent=percent
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to parse real-time price.")

@app.get("/api/market_performers")
async def get_market_performers():
    tickers = ['PSO', 'FFC', 'NBP', 'MEBL', 'OGDC', 'LUCK']
    results = []
    
    for t in tickers:
        try:
            # 1. Scrape real-time price
            url = f"https://dps.psx.com.pk/company/{t}"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=5)
            
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                
                price_el = soup.select_one('.quote__close')
                price = float(price_el.text.strip().replace('Rs.', '').replace(',', '')) if price_el else 0.0
                
                change_el = soup.select_one('.change__value')
                change = float(change_el.text.strip().replace(',', '')) if change_el else 0.0
                
                percent_el = soup.select_one('.change__percent')
                percent = float(percent_el.text.strip().replace('(', '').replace(')', '').replace('%', '')) if percent_el else 0.0
            else:
                price, change, percent = 0.0, 0.0, 0.0
                
            # 2. Get volume from the latest local dataset
            data_path = os.path.join(PROCESSED_DIR, f"{t.lower()}_features.csv")
            volume = 0
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                if not df.empty:
                    volume = int(df['volume'].iloc[-1])
                    
            results.append({
                "symbol": t,
                "price": price,
                "change": change,
                "change_percent": percent,
                "volume": volume
            })
        except Exception as e:
            print(f"Error fetching performer data for {t}: {e}")
            
    # Sort into three categories
    top_active = sorted(results, key=lambda x: x['volume'], reverse=True)
    top_advancers = sorted(results, key=lambda x: x['change_percent'], reverse=True)
    top_decliners = sorted(results, key=lambda x: x['change_percent'])
    
    return {
        "top_active": top_active,
        "top_advancers": top_advancers,
        "top_decliners": top_decliners
    }
