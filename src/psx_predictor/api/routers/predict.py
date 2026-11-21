# predict.py
# ---------------------------------------------------------
# ML prediction endpoints — /api/predict, /api/reload_models
# Moved out of api/main.py verbatim as part of the router split; no
# behavior change from the original implementation.
# ---------------------------------------------------------
import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.psx_predictor.models.train_lstm import LSTMModel
from src.psx_predictor.models.feature_wrapper import get_feature_columns
from src.psx_predictor.config import VALID_TICKERS
from sqlalchemy import text
from src.psx_predictor.db.connection import engine
from src.psx_predictor.api.state import ml_models, psx_cache
from src.psx_predictor.api.master_data import master_path, read_master

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.handlers:
    ch = logging.StreamHandler()
    logger.addHandler(ch)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
HOLDOUT_PATH = os.path.join(MODELS_DIR, "holdout_accuracy.json")
_holdout_cache = {"mtime": None, "data": {}}


def _holdout_accuracy() -> dict:
    """Per-ticker directional accuracy on each model family's held-out test
    period, written at training time (models/utils.save_holdout_accuracy).
    Re-read whenever the file changes."""
    try:
        mtime = os.path.getmtime(HOLDOUT_PATH)
    except OSError:
        return {}
    if mtime != _holdout_cache["mtime"]:
        import json
        with open(HOLDOUT_PATH) as f:
            _holdout_cache.update(mtime=mtime, data=json.load(f))
    return _holdout_cache["data"]


def _fmt_day(iso: str) -> str:
    return pd.Timestamp(iso).strftime("%d %b %Y").lstrip("0")


def _unseen_accuracy(ticker: str, spec: dict | None) -> tuple[dict, dict, dict]:
    """(accuracy %, days, basis) per model key, measured only on days the
    model never trained on. Models without such a measurement are left out
    rather than re-scored on their own training days, which is in-sample and
    reads far higher (e.g. 85% vs ~49% on unseen days)."""
    holdout = _holdout_accuracy()
    acc, days, basis = {}, {}, {}

    def from_holdout(key, family, label):
        entry = holdout.get(family, {})
        row = entry.get("tickers", {}).get(ticker)
        if row:
            acc[key], days[key] = row["accuracy"], row["days"]
            basis[key] = (f"{label}held-out test, {_fmt_day(entry['test_start'])} – "
                          f"{_fmt_day(entry['test_end'])}")

    from_holdout("rf", "rf", "")
    from_holdout("xgb", "xgb", "")
    if spec is not None:
        wf = (spec.get("walk_forward_summary") or {}).get(spec.get("model_type"), {})
        if wf.get("mean_da") is not None:
            acc["specialized"] = round(float(wf["mean_da"]), 1)
            basis["specialized"] = f"walk-forward test over {int(wf.get('n_windows', 0))} periods"
    else:
        from_holdout("specialized", "sector", "sector model, ")
    return acc, days, basis


router = APIRouter()


class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'PSO')", min_length=2)


class DataPoint(BaseModel):
    date: str
    close: float
    rf_pred: float | None = None
    lr_pred: float | None = None
    xgb_pred: float | None = None
    lstm_pred: float | None = None
    specialized_pred: float | None = None


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
    specialized_predicted_price: float | None = Field(
        None, description="MEBL/PSO-only: ticker-specific 1-day model trained on cleaned data + extra volatility features. Null for every other ticker."
    )
    specialized_model_type: str | None = Field(
        None, description="Which model family the specialized prediction came from (e.g. 'lightgbm', 'xgboost')."
    )
    backtest_directional_accuracy: dict[str, float] | None = Field(
        None, description="Directional accuracy (%) per model on days that model never trained on "
                           "(its held-out test or walk-forward test). Models with no such measurement "
                           "are omitted. Keys: rf, lr, xgb, lstm, specialized."
    )
    backtest_sample_size: dict[str, int] | None = Field(
        None, description="Number of unseen days each backtest_directional_accuracy figure covers, where known."
    )
    backtest_basis: dict[str, str] | None = Field(
        None, description="Which test each backtest_directional_accuracy figure comes from."
    )


@router.post("/api/predict", response_model=PredictionResponse)
async def get_prediction(payload: PredictionRequest):
    ticker = payload.ticker.upper()
    if ticker not in VALID_TICKERS:
        raise HTTPException(status_code=400, detail=f"Invalid ticker. Must be one of {VALID_TICKERS}")

    if "rf_predictor" not in ml_models or "lstm_predictor" not in ml_models or "lr_predictor" not in ml_models or "xgb_predictor" not in ml_models:
        raise HTTPException(status_code=503, detail="ML Models are currently unavailable.")

    data_path = master_path(ticker)
    ticker = ticker.lower()

    if data_path is None:
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
            df = read_master(ticker, exclude=['raw_text_announcements', 'raw_news_headline_daily'])
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail=f"Insufficient historical data to compute predictions for {ticker.upper()}.")

            # Text/categorical columns (e.g. sentiment_coverage_era's "unavailable"/"live"
            # labels) must never reach the model inputs. RF/LR/XGB already filter by name
            # via reindex(columns=model.feature_names_in_), but the LSTM scaler has no
            # feature_names_in_ (fit on a bare ndarray) and falls back to a positional
            # slice below - if a text column lands in that slice it crashes MinMaxScaler.
            # get_feature_columns() is shared with training (feature_wrapper.prepare_data)
            # so the two selection lists can't drift apart again.
            exclude_cols = ['ticker', 'date', 'created_at', 'target_return_t1', 'close']
            feature_cols = get_feature_columns(df, exclude_cols)

            # The models now predict percentage return.
            # We need to convert it back to an actual price by multiplying by the current close price.
            current_close_price = df['close'].iloc[-1]

            # 1. Random Forest Inference (Requires only the last day)
            latest_day_features = df[feature_cols].iloc[-1:]
            rf_model = ml_models["rf_predictor"]
            if hasattr(rf_model, 'feature_names_in_'):
                rf_features = latest_day_features.reindex(columns=rf_model.feature_names_in_, fill_value=0.0)
            else:
                rf_features = latest_day_features
            rf_return = rf_model.predict(rf_features)[0]
            rf_prediction = current_close_price * (1 + rf_return)

            # 1a. Linear Regression Inference
            lr_model = ml_models["lr_predictor"]
            if hasattr(lr_model, 'feature_names_in_'):
                lr_features = latest_day_features.reindex(columns=lr_model.feature_names_in_, fill_value=0.0)
            else:
                lr_features = latest_day_features
            # Ridge can't take NaN; train_regression.py fills gaps with 0 too.
            lr_features = lr_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            lr_return = lr_model.predict(lr_features)[0]
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

            xgb_model = ml_models["xgb_predictor"]
            if hasattr(xgb_model, 'feature_names_in_'):
                latest_day_features_xgb = latest_day_features_xgb.reindex(columns=xgb_model.feature_names_in_, fill_value=0.0)

            latest_day_features_xgb['ticker'] = pd.Categorical(latest_day_features_xgb['ticker'], categories=ticker_categories)
            latest_day_features_xgb['sector'] = pd.Categorical(latest_day_features_xgb['sector'], categories=sector_categories)

            xgb_return = xgb_model.predict(latest_day_features_xgb)[0]
            xgb_prediction = current_close_price * (1 + xgb_return)

            # 2. LSTM Inference (Requires sequence of last 30 days)
            lookback = 30
            if len(df) < lookback:
                raise HTTPException(status_code=400, detail="Not enough historical data for LSTM.")

            scaler = ml_models["feature_scaler"]
            expected_scaler_n = getattr(scaler, 'n_features_in_', len(feature_cols))
            scaler_cols = getattr(scaler, 'feature_names_in_', None)

            recent_30_df = df[feature_cols].tail(lookback)
            if scaler_cols is not None:
                # Same 0-fill the LSTM was trained with (train_lstm.py).
                recent_30_df = (recent_30_df.reindex(columns=scaler_cols, fill_value=0.0)
                                .replace([np.inf, -np.inf], np.nan).fillna(0.0))
            elif expected_scaler_n < len(feature_cols):
                recent_30_df = recent_30_df.iloc[:, :expected_scaler_n]

            recent_30_features = recent_30_df.values
            scaled_features = scaler.transform(recent_30_features)

            tensor_input = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 30, 19)

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

            # 1c. Specialized / Sector-Specific Model (MEBL/PSO or Sector Models)
            specialized_predicted_price = None
            specialized_model_type = None
            specialized_key = f"{ticker}_specialized"

            SECTOR_MAPPING = {
                "banking": ["ABL", "AKBL", "BAHL", "BAFL", "BOP", "FABL", "HBL", "HMB", "MCB", "NBP", "SCBPL", "UBL"], # MEBL uses own specialized model
                "cement": ["BWCL", "CHCC", "DGKC", "FCCL", "KOHC", "LUCK", "MLCF", "POWER", "PIOC"],
                "oil_exploration": ["MARI", "OGDC", "POL", "PPL"],
                "oil_marketing": ["APL", "WAFI", "SNGP", "SSGC"], # PSO uses own specialized model
                "technology": ["AIRLINK", "AVN", "HUMNL", "PTC", "SYS", "TRG"],
                "automobile": ["ATLH", "GAL", "GHNI", "HCAR", "INDU", "MTL", "SAZEW", "THALL"],
                "fertilizer": ["EFERT", "FATIMA", "FFC"],
                "pharmaceuticals": ["ABOT", "AGP", "CPHL", "HALEON", "GLAXO", "HINOON", "SEARL"],
                "power": ["HUBC", "KEL", "KAPCO"],
                "food": ["COLG", "FFL", "MUREB", "NATF", "NESTLE", "RMPL", "UPFL"],
                "investment_banks": ["AHCL", "ENGROH", "PSX"],
                "refinery": ["ATRL", "CNERGY", "NRL", "PRL"],
                "textile": ["ILP", "KTML", "MEHT", "GADT", "YOUW", "IBFL", "GATM", "NML"],
                "cable_electrical": ["PAEL"],
                "chemical": ["LOTCHEM", "LCI", "EPCL"],
                "closed_end_funds": ["HGFA"],
                "engineering": ["INIL", "ISL", "ASTL", "MUGHAL"],
                "glass_ceramics": ["GHGL", "TGL"],
                "insurance": ["AICL"],
                "leasing": ["PGLC"],
                "leather_tanneries": ["SRVI"],
                "misc_other": ["PABC", "PSEL", "SHFA"],
                "modarabas": ["FHAM"],
                "paper_packaging": ["PKGS"],
                "property": ["JVDC"],
                "reit": ["DCR"],
                "sugar": ["JDWS"],
                "tobacco": ["PAKT"],
                "transport": ["PIBTL"],
                "woollen": ["BNWM"],
            }

            if specialized_key in ml_models:
                spec = ml_models[specialized_key]
                spec_features = latest_day_features.reindex(columns=spec['feature_cols'], fill_value=0.0)
                spec_return = float(spec['model'].predict(spec_features)[0])
                specialized_predicted_price = current_close_price * (1 + spec_return)
                specialized_model_type = spec.get('model_type')
            else:
                # Find matching sector key for ticker
                matched_sector = None
                for s_key, t_list in SECTOR_MAPPING.items():
                    if ticker.upper() in t_list:
                        matched_sector = s_key
                        break

                if matched_sector and f"{matched_sector}_model" in ml_models:
                    try:
                        s_scalers = ml_models.get(f"{matched_sector}_scalers", {})
                        s_scaler = s_scalers.get(ticker.upper())
                        s_features_list = ml_models.get(f"{matched_sector}_feature_cols", [])
                        s_model = ml_models.get(f"{matched_sector}_model")

                        if s_scaler is not None and s_model is not None and s_features_list:
                            s_sub = df.reindex(columns=s_features_list).iloc[-1:].fillna(0.0).replace([np.inf, -np.inf], 0.0)
                            s_scaled = pd.DataFrame(s_scaler.transform(s_sub), columns=s_features_list)

                            # Attach one-hot ticker columns for that sector based on scalers registry
                            for st_t in s_scalers.keys():
                                s_scaled[f"is_ticker_{st_t}"] = 1.0 if st_t == ticker.upper() else 0.0

                            if hasattr(s_model, 'feature_names_in_'):
                                s_scaled = s_scaled.reindex(columns=s_model.feature_names_in_, fill_value=0.0)

                            s_return = float(s_model.predict(s_scaled)[0])
                            specialized_predicted_price = current_close_price * (1 + s_return)
                            specialized_model_type = f"{matched_sector.replace('_', ' ').title()} Sector Model"
                    except Exception as s_err:
                        logger.warning(f"Sector model prediction failed for {ticker.upper()} ({matched_sector}): {s_err}")

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
            if hasattr(rf_model, 'feature_names_in_'):
                df_sub_rf = df_sub[feature_cols].reindex(columns=rf_model.feature_names_in_, fill_value=0.0)
            else:
                df_sub_rf = df_sub[feature_cols]
            rf_preds_return = pd.Series(rf_model.predict(df_sub_rf), index=df_sub.index)
            df_sub['rf_pred'] = (df_sub['close'] * (1 + rf_preds_return)).shift(1)

            if hasattr(lr_model, 'feature_names_in_'):
                df_sub_lr = df_sub[feature_cols].reindex(columns=lr_model.feature_names_in_, fill_value=0.0)
            else:
                df_sub_lr = df_sub[feature_cols]
            df_sub_lr = df_sub_lr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            lr_preds_return = pd.Series(lr_model.predict(df_sub_lr), index=df_sub.index)
            df_sub['lr_pred'] = (df_sub['close'] * (1 + lr_preds_return)).shift(1)

            df_xgb = df_sub[feature_cols].copy()
            df_xgb.insert(0, 'ticker', ticker.upper())
            df_xgb['sector'] = sector
            if hasattr(xgb_model, 'feature_names_in_'):
                df_xgb = df_xgb.reindex(columns=xgb_model.feature_names_in_, fill_value=0.0)
            df_xgb['ticker'] = pd.Categorical(df_xgb['ticker'], categories=ticker_categories)
            df_xgb['sector'] = pd.Categorical(df_xgb['sector'], categories=sector_categories)

            xgb_preds_return = pd.Series(xgb_model.predict(df_xgb), index=df_xgb.index)
            df_sub['xgb_pred'] = (df_sub['close'] * (1 + xgb_preds_return)).shift(1)

            # Batch sequence generation for LSTM
            df_sub_scaler = df_sub[feature_cols]
            if scaler_cols is not None:
                df_sub_scaler = (df_sub_scaler.reindex(columns=scaler_cols, fill_value=0.0)
                                 .replace([np.inf, -np.inf], np.nan).fillna(0.0))
            elif expected_scaler_n < len(feature_cols):
                df_sub_scaler = df_sub_scaler.iloc[:, :expected_scaler_n]
            all_features_scaled = scaler.transform(df_sub_scaler.values)
            lstm_sequences = []
            lstm_valid_indices = []
            for i in range(lookback, len(df_sub)):
                lstm_sequences.append(all_features_scaled[i - lookback:i])
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

            # Vectorized specialized-model backtest (MEBL/PSO or Sector Models), same
            # shift(1) convention as every other model above: row i holds the
            # prediction *for* day i, computed from day i-1's features.
            history_cols = ['date', 'close', 'rf_pred', 'lr_pred', 'xgb_pred', 'lstm_pred']
            if specialized_key in ml_models:
                spec_df = df_sub[feature_cols].reindex(columns=spec['feature_cols'], fill_value=0.0)
                spec_preds_return = pd.Series(spec['model'].predict(spec_df), index=df_sub.index)
                df_sub['specialized_pred'] = (df_sub['close'] * (1 + spec_preds_return)).shift(1)
                history_cols.append('specialized_pred')
            elif matched_sector and f"{matched_sector}_model" in ml_models:
                try:
                    s_scalers = ml_models.get(f"{matched_sector}_scalers", {})
                    s_scaler = s_scalers.get(ticker.upper())
                    s_features_list = ml_models.get(f"{matched_sector}_feature_cols", [])
                    s_model = ml_models.get(f"{matched_sector}_model")

                    if s_scaler is not None and s_model is not None and s_features_list:
                        df_sec = df_sub.reindex(columns=s_features_list).fillna(0.0).replace([np.inf, -np.inf], 0.0)
                        df_sec_scaled = pd.DataFrame(s_scaler.transform(df_sec), columns=s_features_list, index=df_sub.index)

                        for st_t in s_scalers.keys():
                            df_sec_scaled[f"is_ticker_{st_t}"] = 1.0 if st_t == ticker.upper() else 0.0

                        if hasattr(s_model, 'feature_names_in_'):
                            df_sec_scaled = df_sec_scaled.reindex(columns=s_model.feature_names_in_, fill_value=0.0)

                        sec_preds_return = pd.Series(s_model.predict(df_sec_scaled), index=df_sub.index)
                        df_sub['specialized_pred'] = (df_sub['close'] * (1 + sec_preds_return)).shift(1)
                        history_cols.append('specialized_pred')
                except Exception as sec_backtest_err:
                    logger.warning(f"Sector backtest prediction failed for {ticker.upper()} ({matched_sector}): {sec_backtest_err}")

            # Directional accuracy on days each model never trained on. Scoring
            # the models over df_sub's history instead is in-sample - they were
            # trained on those same days - and overstates them badly.
            backtest_dir_acc, backtest_sample_size, backtest_basis = _unseen_accuracy(
                ticker.upper(), ml_models.get(specialized_key))

            # Select final columns and convert to dict
            recent_history_df = df_sub.tail(1825)[history_cols]
            # Replace NaNs with None for JSON serialization
            recent_history_df = recent_history_df.where(pd.notnull(recent_history_df), None)
            recent_history = recent_history_df.to_dict(orient='records')

            # Get latest sentiment. Sentiment columns are intentionally excluded from
            # the master CSV (DEAD_COLUMNS in build_features.py - coverage is too
            # sparse to train on), so this was always reading a column that doesn't
            # exist and silently returning 0.0. Pull the most recent real aggregate
            # from stock_news_sentiment directly instead.
            latest_sentiment = 0.0
            try:
                sent_query = text("""
                    SELECT sentiment_score FROM stock_news_sentiment
                    WHERE ticker = :ticker
                    ORDER BY date DESC
                    LIMIT 1
                """)
                with engine.connect() as conn:
                    sent_row = conn.execute(sent_query, {"ticker": ticker.upper()}).fetchone()
                if sent_row is not None:
                    latest_sentiment = float(sent_row[0])
            except Exception as sent_err:
                logger.warning(f"Could not fetch latest sentiment for {ticker.upper()}: {sent_err}")

            # Query recent news from DB
            recent_news = []
            try:
                query = text("""
                    SELECT headline, source, published_at, url, sentiment_score
                    FROM stock_news
                    WHERE ticker = :ticker
                    ORDER BY published_at DESC, id DESC
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
                latest_dividend=latest_dividend,
                specialized_predicted_price=round(float(specialized_predicted_price), 2) if specialized_predicted_price is not None else None,
                specialized_model_type=specialized_model_type,
                backtest_directional_accuracy=backtest_dir_acc or None,
                backtest_sample_size=backtest_sample_size or None,
                backtest_basis=backtest_basis or None,
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


@router.post("/api/reload_models")
async def reload_models(api_key: str = Depends(verify_api_key)):
    """Endpoint to hot-reload ML models from disk after retraining."""
    try:
        psx_cache.clear()
        logger.info("Cleared API cache for fresh prediction computation.")
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

        # Per-ticker specialized models (MEBL, PSO) - see main.py lifespan for context.
        for t in ("mebl", "pso"):
            try:
                ml_models[f"{t}_specialized"] = joblib.load(os.path.join(MODELS_DIR, f"{t}_1day_best_model.pkl"))
            except Exception as e:
                logger.error(f"Failed to reload {t.upper()} specialized model: {e}")

        # Sector-Specific Models (Dynamic loading for all PSX sectors)
        sectors = [
            "banking", "cement", "oil_exploration", "oil_marketing", "technology", "automobile",
            "fertilizer", "pharmaceuticals", "power", "food", "investment_banks", "refinery", "textile",
            "cable_electrical", "chemical", "closed_end_funds", "engineering", "glass_ceramics",
            "insurance", "leasing", "leather_tanneries", "misc_other", "modarabas", "paper_packaging",
            "property", "reit", "sugar", "tobacco", "transport", "woollen",
        ]
        for s_key in sectors:
            try:
                sec_dir = os.path.join(MODELS_DIR, f"{s_key}_sector")
                m_path = os.path.join(sec_dir, f"{s_key}_model.pkl")
                sc_path = os.path.join(sec_dir, f"{s_key}_scalers.pkl")
                fc_path = os.path.join(sec_dir, f"{s_key}_feature_cols.pkl")

                if os.path.exists(m_path) and os.path.exists(sc_path) and os.path.exists(fc_path):
                    ml_models[f"{s_key}_model"] = joblib.load(m_path)
                    ml_models[f"{s_key}_scalers"] = joblib.load(sc_path)
                    ml_models[f"{s_key}_feature_cols"] = joblib.load(fc_path)
            except Exception as e:
                logger.error(f"Failed to reload sector model '{s_key}': {e}")

        return {"status": "success", "message": "All AI models & sector models successfully reloaded into memory!"}
    except Exception as e:
        logger.error(f"Failed to reload models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while reloading models.")
