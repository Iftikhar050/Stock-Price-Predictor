import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import RobustScaler

# Scalers are fit on named columns (so predict.py can align by name) but
# transform plain arrays here, which sklearn warns about on every call.
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text
from src.psx_predictor.db.connection import engine
from src.psx_predictor.models.utils import choose_global_cutoff, refit_on_all_data
from src.psx_predictor.models.feature_wrapper import get_feature_columns

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

    # Tickers' master CSVs don't share one column set (each has its own
    # search_trend_<ticker> etc.), so inputs are one fixed, named list: the
    # union of every ticker's numeric feature columns, missing ones as 0. The
    # scaler is fit on named columns, which predict.py lines up by name - the
    # same 0-fill the API applies. get_feature_columns() is the selection
    # predict.py uses, which also keeps text columns out.
    exclude_cols = ['ticker', 'date', 'created_at', 'target_return_t1', 'close']
    frames = {}
    feature_cols = []
    for ticker in valid_tickers:
        file_path = os.path.join(PROCESSED_DIR, f"{ticker.upper()}_master.csv")
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception:
            logger.warning(f"Feature file not found for {ticker}, skipping.")
            continue
        df['target_return_t1'] = (df['close'].shift(-1) - df['close']) / df['close']
        df.dropna(subset=['target_return_t1'], inplace=True)
        cols = get_feature_columns(df, exclude_cols)
        feature_cols += [c for c in cols if c not in set(feature_cols)]
        frames[ticker] = df[['date', 'target_return_t1'] + cols].copy()
    feature_cols_len = len(feature_cols)

    for ticker, df in frames.items():
        dates = pd.to_datetime(df['date'])
        df['sector'] = ticker_sectors.get(ticker, "")
        
        df['ticker_id'] = ticker_to_id[ticker]
        df['sector_id'] = df['sector'].map(sector_to_id).fillna(0).astype(int)
        
        X_num = (df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
                 .fillna(0.0).to_numpy(dtype=np.float32))
        X_cat = df[['ticker_id', 'sector_id']].values
        y_raw = df[['target_return_t1']].values
        
        train_mask = dates <= cutoff_date
        test_mask = dates > cutoff_date
        
        X_num_list.append((X_num[train_mask], X_num[test_mask]))
        X_cat_list.append((X_cat[train_mask], X_cat[test_mask]))
        y_list.append((y_raw[train_mask], y_raw[test_mask]))
        
    X_num_train_all = np.vstack([x[0] for x in X_num_list])
    y_train_all = np.vstack([y[0] for y in y_list])
    
    # RobustScaler (median/IQR) instead of MinMaxScaler: a handful of extreme values
    # (fat-tailed PSX returns, even post data-cleaning) would otherwise set the min/max
    # and compress every other, well-behaved value into a tiny slice of [0, 1].
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    feature_scaler.fit(pd.DataFrame(X_num_train_all, columns=feature_cols))
    target_scaler.fit(y_train_all)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(feature_scaler, os.path.join(MODELS_DIR, "feature_scaler_lstm.pkl"))
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, "target_scaler_lstm.pkl"))
    joblib.dump(ticker_to_id, os.path.join(MODELS_DIR, "ticker_to_id.pkl"))
    joblib.dump(sector_to_id, os.path.join(MODELS_DIR, "sector_to_id.pkl"))
    
    train_datasets, val_datasets, test_datasets = [], [], []

    # Chronological per-ticker split, done BEFORE windowing: the last ~20% of each
    # ticker's own training-period rows (in date order) becomes validation, the
    # earlier ~80% becomes training. Splitting before StockDataset builds its sliding
    # windows guarantees no window straddles the boundary - a random_split on the
    # already-windowed data (the previous behavior) could put two overlapping
    # lookback-30 windows, sharing up to 29 days of the same rows, on opposite sides.
    for i in range(len(X_num_list)):
        X_num_tr, X_num_te = X_num_list[i]
        X_cat_tr, X_cat_te = X_cat_list[i]
        y_tr, y_te = y_list[i]

        if len(X_num_tr) > lookback:
            val_cut = int(len(X_num_tr) * 0.8)
            X_num_sub, X_num_val = X_num_tr[:val_cut], X_num_tr[val_cut:]
            X_cat_sub, X_cat_val = X_cat_tr[:val_cut], X_cat_tr[val_cut:]
            y_sub, y_val = y_tr[:val_cut], y_tr[val_cut:]

            X_num_sub_sc = feature_scaler.transform(X_num_sub)
            y_sub_sc = target_scaler.transform(y_sub)
            if len(X_num_sub) > lookback:
                train_datasets.append(StockDataset(X_num_sub_sc, X_cat_sub, y_sub_sc, lookback))

            if len(X_num_val) > lookback:
                X_num_val_sc = feature_scaler.transform(X_num_val)
                y_val_sc = target_scaler.transform(y_val)
                val_datasets.append(StockDataset(X_num_val_sc, X_cat_val, y_val_sc, lookback))

        if len(X_num_te) > lookback:
            X_num_te_sc = feature_scaler.transform(X_num_te)
            y_te_sc = target_scaler.transform(y_te)
            test_datasets.append(StockDataset(X_num_te_sc, X_cat_te, y_te_sc, lookback))

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    raw_splits = (X_num_list, X_cat_list, y_list, feature_cols)
    return train_loader, val_loader, test_loader, target_scaler, feature_cols_len, len(valid_tickers), len(unique_sectors), raw_splits

def train_lstm_pipeline():
    logger.info("Starting Deep Learning LSTM Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    lookback = 30
    train_loader, val_loader, test_loader, target_scaler, input_dim, num_tickers, num_sectors, raw_splits = prepare_and_scale_data(lookback=lookback)
    
    # 2. Initialize Model, Loss, and Optimizer
    model = LSTMModel(input_dim=input_dim, num_tickers=num_tickers, num_sectors=num_sectors, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop with Early Stopping
    epochs = 100
    early_stopping = EarlyStopping(patience=10, min_delta=1e-5)
    best_val_loss = float('inf')
    best_epoch = 1
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
            best_epoch = epoch + 1
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

    if refit_on_all_data():
        refit_lstm_on_all_data(raw_splits, best_epoch, lookback, input_dim, num_tickers, num_sectors, device, model_path)


def refit_lstm_on_all_data(raw_splits, epochs, lookback, input_dim, num_tickers, num_sectors, device, model_path):
    """The model evaluated above never saw the test period. Train a fresh one
    on every row - scalers re-fit on all rows too - for the epoch count that
    gave the best validation loss, since there's no held-out data left to
    early-stop on. Overwrites the saved weights and scalers."""
    X_num_list, X_cat_list, y_list, feature_cols = raw_splits
    X_num_full = [np.vstack(pair) for pair in X_num_list]
    X_cat_full = [np.vstack(pair) for pair in X_cat_list]
    y_full = [np.vstack(pair) for pair in y_list]

    feature_scaler = RobustScaler().fit(pd.DataFrame(np.vstack(X_num_full), columns=feature_cols))
    target_scaler = RobustScaler().fit(np.vstack(y_full))

    datasets = [
        StockDataset(feature_scaler.transform(xn), xc, target_scaler.transform(yy), lookback)
        for xn, xc, yy in zip(X_num_full, X_cat_full, y_full) if len(xn) > lookback
    ]
    loader = DataLoader(ConcatDataset(datasets), batch_size=64, shuffle=True)
    logger.info(f"Refitting LSTM on all {sum(len(d) for d in datasets)} windows (train + test) for {epochs} epochs...")

    model = LSTMModel(input_dim=input_dim, num_tickers=num_tickers, num_sectors=num_sectors, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (X_num, X_cat), y_batch in loader:
            X_num, X_cat, y_batch = X_num.to(device), X_cat.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_num, X_cat), y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            logger.info(f"Refit epoch [{epoch+1}/{epochs}] | Train Loss: {total / len(loader):.5f}")

    torch.save(model.state_dict(), model_path)
    joblib.dump(feature_scaler, os.path.join(MODELS_DIR, "feature_scaler_lstm.pkl"))
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, "target_scaler_lstm.pkl"))
    logger.info(f"Saved all-data LSTM weights to {model_path} and refit scalers.")

if __name__ == '__main__':
    train_lstm_pipeline()
