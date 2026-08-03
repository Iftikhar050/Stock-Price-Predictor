import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports", "figures")

TICKERS = ['PSO', 'FFC', 'NBP', 'MEBL', 'OGDC', 'LUCK']

def prepare_data(ticker: str):
    """Loads the engineered features and creates the target variable."""
    file_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_features.csv")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}. Run build_features.py first.")
        raise FileNotFoundError(file_path)
        
    df = pd.read_csv(file_path)
    
    # 1. Create Target Variable (Next Day's Close Price)
    df['target_close_t1'] = df['close'].shift(-1)
    df.dropna(subset=['target_close_t1'], inplace=True)
    
    # 2. Select Features (X) and Target (y)
    exclude_cols = ['ticker', 'date', 'created_at', 'target_close_t1']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target_close_t1']
    dates = df['date']
    
    return X, y, dates

def train_and_evaluate():
    """Builds and evaluates the XGBoost model."""
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    dates_test_pso, y_test_pso, preds_pso = None, None, None
    
    for ticker in TICKERS:
        try:
            X, y, dates = prepare_data(ticker)
        except Exception:
            continue
            
        # Time-Series Split (80% Train, 20% Test) without shuffling
        split_idx = int(len(X) * 0.8)
        
        X_train_list.append(X.iloc[:split_idx])
        X_test_list.append(X.iloc[split_idx:])
        y_train_list.append(y.iloc[:split_idx])
        y_test_list.append(y.iloc[split_idx:])
        
        if ticker == 'PSO':
            dates_test_pso = dates.iloc[split_idx:]
            y_test_pso = y.iloc[split_idx:]
            X_test_pso = X.iloc[split_idx:]
            
    X_train = pd.concat(X_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)
    
    logger.info(f"Global Training set: {len(X_train)} samples")
    logger.info(f"Global Testing set: {len(X_test)} samples")
    
    # Train XGBoost
    logger.info("Training XGBRegressor...")
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    
    # Predictions
    logger.info("Evaluating model on global test set...")
    predictions = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    
    logger.info(f"--- Global Model Performance on Test Set ---")
    logger.info(f"MAE:  Rs. {mae:.2f}")
    logger.info(f"RMSE: Rs. {rmse:.2f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    # Save Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Saved trained model artifact to {model_path}")
    
    # Plotting Actual vs Predicted
    if dates_test_pso is not None:
        preds_pso = model.predict(X_test_pso)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        plt.figure(figsize=(14, 6))
        plt.plot(pd.to_datetime(dates_test_pso), y_test_pso.values, label='Actual Close Price', color='blue', alpha=0.7)
        plt.plot(pd.to_datetime(dates_test_pso), preds_pso, label='Predicted Close Price', color='purple', alpha=0.7)
        plt.title(f"PSO - XGBoost Price Prediction (Test Set)")
        plt.xlabel('Date')
        plt.ylabel('Closing Price (Rs.)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(REPORTS_DIR, "xgboost_predictions.png")
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Saved prediction visualization to {plot_path}")

if __name__ == '__main__':
    train_and_evaluate()
