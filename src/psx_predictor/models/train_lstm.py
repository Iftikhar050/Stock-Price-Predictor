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
    def __init__(self, X, y, lookback=30):
        self.X = X
        self.y = y
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        # Sequence of shape (lookback, num_features)
        seq_x = self.X[idx : idx + self.lookback]
        # Target corresponding to the last day in the sequence
        # (Since target is already t+1 in the dataframe, the target at lookback-1 is the correct label)
        seq_y = self.y[idx + self.lookback - 1]
        
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

# 2. LSTM Neural Network Architecture
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
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

TICKERS = ['PSO', 'FFC', 'NBP', 'MEBL', 'OGDC', 'LUCK']

def prepare_and_scale_data(lookback=30):
    X_raw_list, y_raw_list = [], []
    feature_cols_len = 19
    
    for ticker in TICKERS:
        file_path = os.path.join(PROCESSED_DIR, f"{ticker.lower()}_features.csv")
        try:
            df = pd.read_csv(file_path)
        except Exception:
            logger.warning(f"Feature file not found for {ticker}, skipping.")
            continue
            
        df['target_return_t1'] = (df['close'].shift(-1) - df['close']) / df['close']
        df.dropna(subset=['target_return_t1'], inplace=True)
        
        exclude_cols = ['ticker', 'date', 'created_at', 'target_return_t1']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols_len = len(feature_cols)
        
        X_raw = df[feature_cols].values
        y_raw = df[['target_return_t1']].values
        
        n = len(X_raw)
        if n < lookback * 3:
            continue
            
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        
        X_raw_list.append((X_raw[:train_end], X_raw[train_end:val_end], X_raw[val_end:]))
        y_raw_list.append((y_raw[:train_end], y_raw[train_end:val_end], y_raw[val_end:]))
        
    X_train_all = np.vstack([x[0] for x in X_raw_list])
    y_train_all = np.vstack([y[0] for y in y_raw_list])
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    feature_scaler.fit(X_train_all)
    target_scaler.fit(y_train_all)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(feature_scaler, os.path.join(MODELS_DIR, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(MODELS_DIR, "target_scaler.pkl"))
    
    train_datasets, val_datasets, test_datasets = [], [], []
    
    for i in range(len(X_raw_list)):
        X_tr_sc = feature_scaler.transform(X_raw_list[i][0])
        X_va_sc = feature_scaler.transform(X_raw_list[i][1])
        X_te_sc = feature_scaler.transform(X_raw_list[i][2])
        
        y_tr_sc = target_scaler.transform(y_raw_list[i][0])
        y_va_sc = target_scaler.transform(y_raw_list[i][1])
        y_te_sc = target_scaler.transform(y_raw_list[i][2])
        
        if len(X_tr_sc) > lookback:
            train_datasets.append(StockDataset(X_tr_sc, y_tr_sc, lookback))
        if len(X_va_sc) > lookback:
            val_datasets.append(StockDataset(X_va_sc, y_va_sc, lookback))
        if len(X_te_sc) > lookback:
            test_datasets.append(StockDataset(X_te_sc, y_te_sc, lookback))
            
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    test_dataset = ConcatDataset(test_datasets)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, target_scaler, feature_cols_len

def train_lstm_pipeline():
    logger.info("Starting Deep Learning LSTM Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    lookback = 30
    train_loader, val_loader, test_loader, target_scaler, input_dim = prepare_and_scale_data(lookback=lookback)
    
    # 2. Initialize Model, Loss, and Optimizer
    model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
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
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
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
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
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
