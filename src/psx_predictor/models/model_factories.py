'''Factory functions for creating fresh model instances.
Each factory returns an *untrained* model ready for fit().
Used by the walk‑forward evaluation pipeline.
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
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

def svr_rbf_factory(epsilon: float = 0.001):
    """Return a new SVR (RBF kernel) instance.

    Requires standardized input (StandardScaler) - unlike the tree models,
    SVR is not scale-invariant. `epsilon` defaults far below sklearn's 0.1
    default: a next-day return is ~1-2% in magnitude, and the epsilon-
    insensitive tube swallows nearly every training point if epsilon is
    comparable to or larger than the target's own scale, collapsing the fit
    to just the bias term (found and fixed earlier via the MEBL/PSO report
    bake-off - both RBF and linear kernels converged to near-identical,
    near-zero predictions until epsilon was shrunk to match the target scale).
    """
    return SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=epsilon)

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

def lightgbm_factory():
    """Return a new LGBMRegressor instance.

    Second tree-ensemble opinion alongside XGBoost, per the comparative-study
    literature (SVM-RBF/RF being the accurate-but-slow end, trees being the
    right family for wide tabular data like this dataset's 270+ columns).
    LightGBM's native categorical support means `sector` doesn't need to be
    cast to a pandas category first the way XGBoost's `enable_categorical`
    path requires - pass the sector column's name via `categorical_feature`
    at fit time if using this directly outside of prepare_data()'s df.
    """
    if LGBMRegressor is None:
        raise ImportError("lightgbm is not installed; run `pip install lightgbm`.")
    return LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
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
