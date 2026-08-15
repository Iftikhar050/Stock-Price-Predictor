import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text
from src.psx_predictor.db.connection import engine
from src.psx_predictor.models.utils import choose_global_cutoff

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

# 1. Custom PyTorch Dataset for Sliding Windows
class StockDataset(Dataset):
    def __init__(self, X_num, X_cat, y, lookback=30):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y
        self.lookback = lookback

    def __len__(self):
        return len(self.X_num) - self.lookback

    def __getitem__(self, idx):
        seq_x_num = self.X_num[idx : idx + self.lookback]
        seq_x_cat = self.X_cat[idx : idx + self.lookback]
        seq_y = self.y[idx + self.lookback - 1]
        
        return (torch.tensor(seq_x_num, dtype=torch.float32), 
                torch.tensor(seq_x_cat, dtype=torch.long)), \
               torch.tensor(seq_y, dtype=torch.float32)

# 2. LSTM Neural Network Architecture
class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_tickers, num_sectors, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Categorical Embeddings
        self.ticker_embed = nn.Embedding(num_tickers, 16)
        self.sector_embed = nn.Embedding(num_sectors, 8)
        
        # Total input to LSTM = numeric features + 16 (ticker) + 8 (sector)
        lstm_input_dim = input_dim + 16 + 8
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_num, x_cat):
        # x_cat is [batch, lookback, 2] where 0 is ticker_id, 1 is sector_id
        t_emb = self.ticker_embed(x_cat[:, :, 0])
        s_emb = self.sector_embed(x_cat[:, :, 1])
        
        # Concat along feature dimension
        x = torch.cat([x_num, t_emb, s_emb], dim=-1)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

from src.psx_predictor.db.repository import get_active_tickers
TICKERS = get_active_tickers()

def prepare_and_scale_data(lookback=30):
    cutoff_str, valid_tickers = choose_global_cutoff(test_trading_days=250, min_train_trading_days=500)
    cutoff_date = pd.to_datetime(cutoff_str)
    
    query = text("SELECT ticker, sector FROM stock_metadata")
    with engine.connect() as conn:
        res = conn.execute(query).fetchall()
    ticker_sectors = {row[0]: row[1] for row in res}
    
    ticker_to_id = {t: i for i, t in enumerate(valid_tickers)}
    unique_sectors = list(set(ticker_sectors.values()))
    sector_to_id = {s: i for i, s in enumerate(unique_sectors)}
    
    X_num_list, X_cat_list, y_list = [], [], []
    feature_cols_len = 0
    
    for ticker in valid_tickers:
        file_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_features.csv")
        try:
            df = pd.read_csv(file_path)
        except Exception:
            logger.warning(f"Feature file not found for {ticker}, skipping.")
            continue
            
        df['target_return_t1'] = (df['close'].shift(-1) - df['close']) / df['close']
        df.dropna(subset=['target_return_t1'], inplace=True)
        
        exclude_cols = ['ticker', 'date', 'created_at', 'target_return_t1', 'close']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols_len = len(feature_cols)
        
        dates = pd.to_datetime(df['date'])
        df['sector'] = ticker_sectors.get(ticker, "")
        
        df['ticker_id'] = ticker_to_id[ticker]
        df['sector_id'] = df['sector'].map(sector_to_id).fillna(0).astype(int)
        
        X_num = df[feature_cols].values
        X_cat = df[['ticker_id', 'sector_id']].values
        y_raw = df[['target_return_t1']].values
        
        train_mask = dates <= cutoff_date
        test_mask = dates > cutoff_date
        
        X_num_list.append((X_num[train_mask], X_num[test_mask]))
        X_cat_list.append((X_cat[train_mask], X_cat[test_mask]))
        y_list.append((y_raw[train_mask], y_raw[test_mask]))
        
    X_num_train_all = np.vstack([x[0] for x in X_num_list])
    y_train_all = np.vstack([y[0] for y in y_list])
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    feature_scaler.fit(X_num_train_all)
    target_scaler.fit(y_train_all)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(feature_scaler, os.path.join(MODELS_DIR, "feature_scaler_lstm.pkl"))
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, "target_scaler_lstm.pkl"))
    
    train_datasets, test_datasets = [], []
    
    for i in range(len(X_num_list)):
        X_num_tr, X_num_te = X_num_list[i]
        X_cat_tr, X_cat_te = X_cat_list[i]
        y_tr, y_te = y_list[i]
        
        if len(X_num_tr) > lookback:
            X_num_tr_sc = feature_scaler.transform(X_num_tr)
            y_tr_sc = target_scaler.transform(y_tr)
            train_datasets.append(StockDataset(X_num_tr_sc, X_cat_tr, y_tr_sc, lookback))
            
        if len(X_num_te) > lookback:
            X_num_te_sc = feature_scaler.transform(X_num_te)
            y_te_sc = target_scaler.transform(y_te)
            test_datasets.append(StockDataset(X_num_te_sc, X_cat_te, y_te_sc, lookback))
            
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)
    
    # 80/20 split on the train_dataset for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, target_scaler, feature_cols_len, len(valid_tickers), len(unique_sectors)

def train_lstm_pipeline():
    logger.info("Starting Deep Learning LSTM Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    lookback = 30
    train_loader, val_loader, test_loader, target_scaler, input_dim, num_tickers, num_sectors = prepare_and_scale_data(lookback=lookback)
    
    # 2. Initialize Model, Loss, and Optimizer
    model = LSTMModel(input_dim=input_dim, num_tickers=num_tickers, num_sectors=num_sectors, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop with Early Stopping
    epochs = 100
    early_stopping = EarlyStopping(patience=10, min_delta=1e-5)
    best_val_loss = float('inf')
    model_path = os.path.join(MODELS_DIR, "lstm_model.pth")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_inputs, y_batch in train_loader:
            X_num, X_cat = X_inputs
            X_num, X_cat, y_batch = X_num.to(device), X_cat.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_num, X_cat)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_inputs, y_batch in val_loader:
                X_num, X_cat = X_inputs
                X_num, X_cat, y_batch = X_num.to(device), X_cat.to(device), y_batch.to(device)
                outputs = model(X_num, X_cat)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
    # 5. Evaluate on Test Set
    logger.info("Training complete. Evaluating on Test Set...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    predictions, actuals = [], []
    with torch.no_grad():
        for X_inputs, y_batch in test_loader:
            X_num, X_cat = X_inputs
            X_num, X_cat = X_num.to(device), X_cat.to(device)
            outputs = model(X_num, X_cat)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
            
    # Inverse transform to get real Return values
    predictions_real = target_scaler.inverse_transform(predictions)
    actuals_real = target_scaler.inverse_transform(actuals)
    
    mae = mean_absolute_error(actuals_real, predictions_real)
    rmse = np.sqrt(mean_squared_error(actuals_real, predictions_real))
    
    logger.info(f"--- LSTM Model Performance (Test Set) ---")
    logger.info(f"MAE:  {mae * 100:.2f}%")
    logger.info(f"RMSE: {rmse * 100:.2f}%")
    logger.info(f"Model weights saved to {model_path}")

if __name__ == '__main__':
    train_lstm_pipeline()
