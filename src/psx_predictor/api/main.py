# main.py
# ---------------------------------------------------------
# FastAPI Backend - Stock Predictor API
# ---------------------------------------------------------
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import pandas as pd
import numpy as np
import torch
from contextlib import asynccontextmanager
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Ensure the root is in path for imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.psx_predictor.models.train_lstm import LSTMModel
from src.psx_predictor.config import VALID_TICKERS
from sqlalchemy import text
from src.psx_predictor.db.connection import engine

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
if not logger.handlers:
    ch = logging.StreamHandler()
    logger.addHandler(ch)

# Global dictionaries
ml_models = {}
psx_cache = {}
CACHE_TTL = 15

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

class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'PSO')", min_length=2)

class DataPoint(BaseModel):
    date: str
    close: float
    rf_pred: float | None = None
    lr_pred: float | None = None
    xgb_pred: float | None = None
    lstm_pred: float | None = None

class DividendInfo(BaseModel):
    amount: float
    ex_date: str
    type: str

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
    model_agreement_score: float
    historical_data: list[DataPoint] = Field(..., description="Last 30 days of closing prices for chart rendering")
    latest_sentiment: float = 0.0
    recent_news: list[dict] = Field(default_factory=list)
    latest_dividend: DividendInfo | None = None

@app.post("/api/predict", response_model=PredictionResponse)
async def get_prediction(payload: PredictionRequest):
    ticker = payload.ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    if "rf_predictor" not in ml_models or "lstm_predictor" not in ml_models or "lr_predictor" not in ml_models or "xgb_predictor" not in ml_models:
        raise HTTPException(status_code=503, detail="ML Models are currently unavailable.")
        
    ticker = ticker.lower()
    data_path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Feature data for ticker '{ticker.upper()}' not found.")
        
    try:
        file_mtime = os.path.getmtime(data_path)
        cache_key = f"predict_{ticker}"
        if cache_key in psx_cache:
            cached_mtime, cached_result = psx_cache[cache_key]
            if cached_mtime == file_mtime:
                return cached_result
                
        # Run synchronous blocking logic in threadpool to avoid blocking event loop
        def _compute():
            df = pd.read_csv(data_path)
            if df.empty:
                raise HTTPException(status_code=404, detail=f"Insufficient historical data to compute predictions for {ticker.upper()}.")
            
            exclude_cols = ['ticker', 'date', 'created_at', 'target_return_t1', 'close']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # The models now predict percentage return.
            # We need to convert it back to an actual price by multiplying by the current close price.
            current_close_price = df['close'].iloc[-1]
        
            # 1. Random Forest Inference (Requires only the last day)
            latest_day_features = df[feature_cols].iloc[-1:]
            rf_return = ml_models["rf_predictor"].predict(latest_day_features)[0]
            rf_prediction = current_close_price * (1 + rf_return)
        
            # 1a. Linear Regression Inference
            lr_return = ml_models["lr_predictor"].predict(latest_day_features)[0]
            lr_prediction = current_close_price * (1 + lr_return)

            # 1b. XGBoost Inference
            latest_day_features_xgb = latest_day_features.copy()
            latest_day_features_xgb.insert(0, 'ticker', ticker.upper())
        
            query = text("SELECT sector FROM stock_metadata WHERE ticker = :ticker")
            with engine.connect() as conn:
                sector = conn.execute(query, {"ticker": ticker.upper()}).scalar() or "Unknown"
            latest_day_features_xgb['sector'] = sector
        
            ticker_categories = ml_models["xgb_ticker_categories"]
            sector_categories = ml_models["xgb_sector_categories"]
            latest_day_features_xgb['ticker'] = pd.Categorical(latest_day_features_xgb['ticker'], categories=ticker_categories)
            latest_day_features_xgb['sector'] = pd.Categorical(latest_day_features_xgb['sector'], categories=sector_categories)
        
            xgb_return = ml_models["xgb_predictor"].predict(latest_day_features_xgb)[0]
            xgb_prediction = current_close_price * (1 + xgb_return)
        
            # 2. LSTM Inference (Requires sequence of last 30 days)
            lookback = 30
            if len(df) < lookback:
                raise HTTPException(status_code=400, detail="Not enough historical data for LSTM.")
            
            recent_30_features = df[feature_cols].tail(lookback).values
            scaled_features = ml_models["feature_scaler"].transform(recent_30_features)
        
            tensor_input = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0) # Shape: (1, 30, 19)
        
            ticker_id = ml_models["ticker_to_id"].get(ticker.upper(), 0)
            sector_id = ml_models["sector_to_id"].get(sector, 0)
        
            x_cat_np = np.zeros((1, lookback, 2), dtype=int)
            x_cat_np[0, :, 0] = ticker_id
            x_cat_np[0, :, 1] = sector_id
            tensor_cat = torch.tensor(x_cat_np, dtype=torch.long)
        
            with torch.no_grad():
                lstm_out_scaled = ml_models["lstm_predictor"](tensor_input, tensor_cat)
            
            lstm_return = ml_models["target_scaler"].inverse_transform(lstm_out_scaled.numpy())[0][0]
            lstm_prediction = current_close_price * (1 + lstm_return)
        
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
            # We only need predictions for the last 1825 days (5 years)
            history_len = 1825
            subset_start = max(0, len(df) - history_len - lookback)
            df_sub = df.iloc[subset_start:].copy()
        
            # Vectorized inference for traditional models
            rf_preds_return = pd.Series(ml_models["rf_predictor"].predict(df_sub[feature_cols]), index=df_sub.index)
            df_sub['rf_pred'] = (df_sub['close'] * (1 + rf_preds_return)).shift(1)
        
            lr_preds_return = pd.Series(ml_models["lr_predictor"].predict(df_sub[feature_cols]), index=df_sub.index)
            df_sub['lr_pred'] = (df_sub['close'] * (1 + lr_preds_return)).shift(1)
        
            df_xgb = df_sub[feature_cols].copy()
            df_xgb.insert(0, 'ticker', ticker.upper())
            df_xgb['sector'] = sector
            df_xgb['ticker'] = pd.Categorical(df_xgb['ticker'], categories=ticker_categories)
            df_xgb['sector'] = pd.Categorical(df_xgb['sector'], categories=sector_categories)
        
            xgb_preds_return = pd.Series(ml_models["xgb_predictor"].predict(df_xgb), index=df_xgb.index)
            df_sub['xgb_pred'] = (df_sub['close'] * (1 + xgb_preds_return)).shift(1)
        
            # Batch sequence generation for LSTM
            all_features_scaled = ml_models["feature_scaler"].transform(df_sub[feature_cols].values)
            lstm_sequences = []
            lstm_valid_indices = []
            for i in range(lookback, len(df_sub)):
                lstm_sequences.append(all_features_scaled[i-lookback:i])
                lstm_valid_indices.append(df_sub.index[i])
            
            df_sub['lstm_pred'] = None
            if lstm_sequences:
                batch_tensor = torch.tensor(np.array(lstm_sequences), dtype=torch.float32)
            
                batch_size = batch_tensor.size(0)
                x_cat_batch_np = np.zeros((batch_size, lookback, 2), dtype=int)
                x_cat_batch_np[:, :, 0] = ticker_id
                x_cat_batch_np[:, :, 1] = sector_id
                batch_tensor_cat = torch.tensor(x_cat_batch_np, dtype=torch.long)
            
                with torch.no_grad():
                    batch_lstm_out = ml_models["lstm_predictor"](batch_tensor, batch_tensor_cat)
                batch_lstm_preds_return = ml_models["target_scaler"].inverse_transform(batch_lstm_out.numpy()).flatten()
            
                # Create a temporary series to hold predictions and shift by 1
                lstm_series_return = pd.Series(index=df_sub.index, dtype=float)
                lstm_series_return.loc[lstm_valid_indices] = batch_lstm_preds_return
                df_sub['lstm_pred'] = (df_sub['close'] * (1 + lstm_series_return)).shift(1)
            
            # Select final columns and convert to dict
            recent_history_df = df_sub.tail(1825)[['date', 'close', 'rf_pred', 'lr_pred', 'xgb_pred', 'lstm_pred']]
            # Replace NaNs with None for JSON serialization
            recent_history_df = recent_history_df.where(pd.notnull(recent_history_df), None)
            recent_history = recent_history_df.to_dict(orient='records')
        
            # Get latest sentiment
            latest_sentiment = float(df['sentiment_score'].iloc[-1]) if 'sentiment_score' in df.columns else 0.0
        
            # Query recent news from DB
            recent_news = []
            try:
                query = text("""
                    SELECT headline, source, published_at, url, sentiment_score 
                    FROM stock_news 
                    WHERE ticker = :ticker 
                    ORDER BY published_at DESC 
                    LIMIT 5
                """)
                with engine.connect() as conn:
                    news_result = conn.execute(query, {"ticker": payload.ticker.upper()}).fetchall()
                    for row in news_result:
                        recent_news.append({
                            "headline": row.headline,
                            "source": row.source,
                            "published_at": str(row.published_at),
                            "url": row.url,
                            "sentiment_score": float(row.sentiment_score) if row.sentiment_score is not None else 0.0
                        })
            except Exception as e:
                logger.error(f"Failed to fetch recent news: {e}")
            
            # Fetch latest dividend
            latest_dividend = None
            dividend_query = text("SELECT dividend_amount, ex_dividend_date, dividend_type FROM stock_dividends WHERE ticker = :ticker ORDER BY ex_dividend_date DESC LIMIT 1")
            with engine.connect() as conn:
                div_res = conn.execute(dividend_query, {"ticker": payload.ticker.upper()}).fetchone()
                if div_res:
                    latest_dividend = DividendInfo(
                        amount=float(div_res[0]),
                        ex_date=str(div_res[1]),
                        type=str(div_res[2])
                    )
        
            result = PredictionResponse(
                ticker=payload.ticker.upper(),
                latest_date=str(df['date'].iloc[-1]),
                current_price=round(float(df['close'].iloc[-1]), 2),
                rf_predicted_price=round(float(rf_prediction), 2),
                lr_predicted_price=round(float(lr_prediction), 2),
                xgb_predicted_price=round(float(xgb_prediction), 2),
                lstm_predicted_price=round(float(lstm_prediction), 2),
                ensemble_min=round(ensemble_min, 2),
                ensemble_max=round(ensemble_max, 2),
                model_agreement_score=round(confidence, 1),
                historical_data=recent_history,
                latest_sentiment=round(latest_sentiment, 2),
                recent_news=recent_news,
                latest_dividend=latest_dividend
            )
            return result

        result = await run_in_threadpool(_compute)
        psx_cache[cache_key] = (file_mtime, result)
        return result
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        import traceback
        traceback.print_exc()
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while processing prediction.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Server misconfiguration.")
    if api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API Key.")
    return api_key

@app.post("/api/reload_models")
async def reload_models(api_key: str = Depends(verify_api_key)):
    """Endpoint to hot-reload ML models from disk after retraining."""
    try:
        # Load RF
        ml_models["rf_predictor"] = joblib.load(os.path.join(MODELS_DIR, "baseline_rf_model.pkl"))
        
        # Load LR
        ml_models["lr_predictor"] = joblib.load(os.path.join(MODELS_DIR, "lr_model.pkl"))
        
        # Load XGBoost
        ml_models["xgb_predictor"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
        
        # Load Scalers for LSTM
        ml_models["feature_scaler"] = joblib.load(os.path.join(MODELS_DIR, "feature_scaler_lstm.pkl"))
        ml_models["target_scaler"] = joblib.load(os.path.join(MODELS_DIR, "target_scaler_lstm.pkl"))
        ml_models["ticker_to_id"] = joblib.load(os.path.join(MODELS_DIR, "ticker_to_id.pkl"))
        ml_models["sector_to_id"] = joblib.load(os.path.join(MODELS_DIR, "sector_to_id.pkl"))
        
        # Load XGBoost Categories
        ml_models["xgb_ticker_categories"] = joblib.load(os.path.join(MODELS_DIR, "xgb_ticker_categories.pkl"))
        ml_models["xgb_sector_categories"] = joblib.load(os.path.join(MODELS_DIR, "xgb_sector_categories.pkl"))
        
        # Load LSTM dynamically using feature_scaler's input dim and state_dict embeddings
        device = torch.device("cpu")
        input_dim = ml_models["feature_scaler"].n_features_in_
        
        state_dict = torch.load(os.path.join(MODELS_DIR, "lstm_model.pth"), map_location=device, weights_only=True)
        num_tickers = state_dict['ticker_embed.weight'].shape[0]
        num_sectors = state_dict['sector_embed.weight'].shape[0]
        
        lstm = LSTMModel(input_dim=input_dim, num_tickers=num_tickers, num_sectors=num_sectors, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
        lstm.load_state_dict(state_dict)
        lstm.eval()
        ml_models["lstm_predictor"] = lstm
        
        return {"status": "success", "message": "All AI models successfully reloaded into memory!"}
    except Exception as e:
        logger.error(f"Failed to reload models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while reloading models.")

def fetch_psx_company_profile(ticker: str):
    cache_key = f"profile_{ticker.upper()}"
    if cache_key in psx_cache:
        cached_time, data = psx_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL:
            return data

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
                
    result = {
        "name": name,
        "sector": sector,
        "description": desc,
        "people": people,
        "details": details
    }
    psx_cache[cache_key] = (time.time(), result)
    return result

class TickerInfo(BaseModel):
    ticker: str
    name: str
    sector: str

@app.get("/api/tickers", response_model=list[TickerInfo])
async def get_all_tickers():
    cache_key = "all_tickers"
    if cache_key in psx_cache:
        cached_time, data = psx_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL * 10: # Cache for longer since it rarely changes
            return data
            
    tickers_list = []
    try:
        query = text("SELECT ticker, company_name, sector FROM stock_metadata WHERE is_active = true ORDER BY ticker ASC")
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
            for row in result:
                tickers_list.append(TickerInfo(
                    ticker=row.ticker,
                    name=row.company_name,
                    sector=row.sector
                ))
        psx_cache[cache_key] = (time.time(), tickers_list)
        return tickers_list
    except Exception as e:
        logger.error(f"Failed to fetch tickers: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching tickers.")

@app.get("/api/company/{ticker}")
async def get_company_profile(ticker: str):
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    profile = await run_in_threadpool(fetch_psx_company_profile, ticker)
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
    ticker = ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    cache_key = f"realtime_{ticker}"
    if cache_key in psx_cache:
        cached_time, data = psx_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL:
            return data

    url = f"https://dps.psx.com.pk/company/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    def fetch():
        return requests.get(url, headers=headers, timeout=5)
        
    r = await run_in_threadpool(fetch)
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
        
        result = RealtimePriceResponse(
            ticker=ticker,
            price=price,
            change=change,
            change_percent=percent
        )
        psx_cache[cache_key] = (time.time(), result)
        return result
    except Exception as e:
        logger.error(f"Failed to parse real-time price for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while parsing real-time price.")

def _fetch_single_performer(t):
    try:
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
            
        # Since volume is no longer in the ML feature set, query it directly from the database
        volume = 0
        try:
            vol_query = text("SELECT volume FROM stock_eod_data WHERE ticker = :ticker ORDER BY date DESC LIMIT 1")
            with engine.connect() as conn:
                vol_res = conn.execute(vol_query, {"ticker": t.upper()}).fetchone()
                if vol_res:
                    volume = int(vol_res[0])
        except Exception as db_e:
            logger.error(f"Failed to fetch volume from DB for {t}: {db_e}")
                
        return {
            "symbol": t,
            "price": price,
            "change": change,
            "change_percent": percent,
            "volume": volume
        }
    except Exception as e:
        logger.error(f"Error fetching performer data for {t}: {e}")
        return None

def fetch_performer_data(tickers):
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(_fetch_single_performer, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            res = future.result()
            if res:
                results.append(res)
    return results

@app.get("/api/market_performers")
async def get_market_performers():
    cache_key = "market_performers"
    if cache_key in psx_cache:
        cached_time, data = psx_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL:
            return data

    tickers = list(VALID_TICKERS)
    results = await run_in_threadpool(fetch_performer_data, tickers)
    
    # Sort into three categories
    top_active = sorted(results, key=lambda x: x['volume'], reverse=True)
    top_advancers = sorted(results, key=lambda x: x['change_percent'], reverse=True)
    top_decliners = sorted(results, key=lambda x: x['change_percent'])
    
    result = {
        "top_active": top_active,
        "top_advancers": top_advancers,
        "top_decliners": top_decliners
    }
    psx_cache[cache_key] = (time.time(), result)
    return result
