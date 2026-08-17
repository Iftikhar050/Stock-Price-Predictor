'''Factory functions for creating fresh model instances.
Each factory returns an *untrained* model ready for fit().
Used by the walk‑forward evaluation pipeline.
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

def baseline_factory():
    """Return a new RandomForestRegressor instance."""
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

def ridge_factory():
    """Return a new Ridge regression instance."""
    return Ridge(alpha=1.0)

def xgboost_factory():
    """Return a new XGBRegressor instance."""
    return XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
    )

if torch is not None:
    class LSTMModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out).squeeze(-1)

    def lstm_factory(input_dim: int):
        """Create a fresh LSTM model. Caller must supply the feature dimension.
        The dedicated LSTM training script handles its own evaluation.
        """
        return LSTMModel(input_dim)
else:
    # Placeholder factory when torch is unavailable
    def lstm_factory(input_dim: int):
        raise ImportError("torch is not installed; LSTM model cannot be created.")
