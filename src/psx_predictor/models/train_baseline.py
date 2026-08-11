import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
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
    
    # 1. Create Target Variable (Next Day's Return)
    df['target_return_t1'] = (df['close'].shift(-1) - df['close']) / df['close']
    df.dropna(subset=['target_return_t1'], inplace=True)
    
    # 2. Select Features (X) and Target (y)
    exclude_cols = ['ticker', 'date', 'created_at', 'target_return_t1', 'close']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['target_return_t1']
    dates = df['date']
    current_close = df['close']
    
    return X, y, dates, current_close

def train_and_evaluate():
    """Builds and evaluates the baseline Random Forest model."""
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    close_test_list = []
    dates_test_pso, y_test_pso, preds_pso, close_test_pso = None, None, None, None
    
    for ticker in TICKERS:
        try:
            X, y, dates, current_close = prepare_data(ticker)
        except Exception:
            continue
            
        # Time-Series Split (80% Train, 20% Test) without shuffling
        split_idx = int(len(X) * 0.8)
        
        X_train_list.append(X.iloc[:split_idx])
        X_test_list.append(X.iloc[split_idx:])
        y_train_list.append(y.iloc[:split_idx])
        y_test_list.append(y.iloc[split_idx:])
        close_test_list.append(current_close.iloc[split_idx:])
        
        if ticker == 'PSO':
            dates_test_pso = dates.iloc[split_idx:]
            y_test_pso = y.iloc[split_idx:]
            X_test_pso = X.iloc[split_idx:]
            close_test_pso = current_close.iloc[split_idx:]
            
    X_train = pd.concat(X_train_list, ignore_index=True)
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)
    close_test_all = pd.concat(close_test_list, ignore_index=True)
    
    logger.info(f"Global Training set: {len(X_train)} samples")
    logger.info(f"Global Testing set: {len(X_test)} samples")
    
    # Train Baseline Random Forest
    logger.info("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    logger.info("Evaluating model on global test set...")
    predictions_return = model.predict(X_test)
    
    predicted_prices = close_test_all * (1 + predictions_return)
    actual_prices = close_test_all * (1 + y_test)
    
    # Metrics
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
    
    logger.info(f"--- Global Model Performance on Test Set ---")
    logger.info(f"MAE:  Rs. {mae:.2f}")
    logger.info(f"RMSE: Rs. {rmse:.2f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    # Save Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "baseline_rf_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Saved trained model artifact to {model_path}")
    
    # Plotting Actual vs Predicted (Only for PSO to keep chart clean)
    if dates_test_pso is not None:
        preds_pso_return = model.predict(X_test_pso)
        preds_pso_price = close_test_pso * (1 + preds_pso_return)
        actual_pso_price = close_test_pso * (1 + y_test_pso)
        
        os.makedirs(REPORTS_DIR, exist_ok=True)
        plt.figure(figsize=(14, 6))
        plt.plot(pd.to_datetime(dates_test_pso), actual_pso_price.values, label='Actual Close Price', color='blue', alpha=0.7)
        plt.plot(pd.to_datetime(dates_test_pso), preds_pso_price.values, label='Predicted Close Price', color='red', alpha=0.7)
        plt.title(f"PSO - Baseline Random Forest Price Prediction (Test Set)")
        plt.xlabel('Date')
        plt.ylabel('Closing Price (Rs.)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(REPORTS_DIR, "baseline_predictions.png")
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Saved prediction visualization to {plot_path}")

if __name__ == '__main__':
    train_and_evaluate()
