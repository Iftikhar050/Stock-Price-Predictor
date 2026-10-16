import numpy as np
import pytest
import torch

from src.psx_predictor.models.train_lstm import EarlyStopping, LSTMModel, StockDataset


class TestEarlyStopping:
    def test_first_call_just_records_baseline(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        assert es.best_loss == 1.0
        assert es.counter == 0
        assert es.early_stop is False

    def test_improving_loss_resets_counter(self):
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)  # worse
        assert es.counter == 1
        es(0.5)  # better -> resets
        assert es.counter == 0
        assert es.best_loss == 0.5

    def test_stops_after_patience_consecutive_non_improvements(self):
        es = EarlyStopping(patience=2)
        es(1.0)
        es(1.1)
        assert es.early_stop is False
        es(1.2)
        assert es.early_stop is True

    def test_min_delta_requires_meaningful_improvement(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es(1.0)
        es(0.95)  # improved, but less than min_delta -> counts as non-improvement
        assert es.counter == 1


class TestStockDataset:
    def test_length_accounts_for_lookback_window(self):
        n, lookback = 50, 10
        ds = StockDataset(np.zeros((n, 3)), np.zeros((n, 2), dtype=int), np.zeros(n), lookback=lookback)
        assert len(ds) == n - lookback

    def test_getitem_returns_correctly_shaped_window(self):
        n, lookback, n_features = 50, 10, 3
        X_num = np.arange(n * n_features, dtype=float).reshape(n, n_features)
        X_cat = np.zeros((n, 2), dtype=int)
        y = np.arange(n, dtype=float)

        ds = StockDataset(X_num, X_cat, y, lookback=lookback)
        (seq_num, seq_cat), target = ds[0]

        assert seq_num.shape == (lookback, n_features)
        assert seq_cat.shape == (lookback, 2)
        assert target.item() == pytest.approx(y[lookback - 1])

    def test_getitem_window_slides_with_index(self):
        n, lookback = 20, 5
        X_num = np.arange(n).reshape(n, 1).astype(float)
        X_cat = np.zeros((n, 2), dtype=int)
        y = np.arange(n, dtype=float)

        ds = StockDataset(X_num, X_cat, y, lookback=lookback)
        (seq_0, _), _ = ds[0]
        (seq_3, _), _ = ds[3]

        assert seq_0[0].item() == 0
        assert seq_3[0].item() == 3  # window start shifted by the same index


class TestLSTMModel:
    def test_forward_output_shape(self):
        batch, lookback, input_dim = 4, 7, 6
        model = LSTMModel(input_dim=input_dim, num_tickers=10, num_sectors=5, hidden_dim=16, num_layers=1)
        x_num = torch.randn(batch, lookback, input_dim)
        x_cat = torch.zeros(batch, lookback, 2, dtype=torch.long)

        out = model(x_num, x_cat)
        assert out.shape == (batch, 1)

    def test_different_ticker_ids_produce_different_output(self):
        """Confirms the ticker embedding actually participates in the
        forward pass, not just along for the ride unused."""
        torch.manual_seed(0)
        model = LSTMModel(input_dim=4, num_tickers=10, num_sectors=5, hidden_dim=8, num_layers=1)
        model.eval()

        x_num = torch.randn(1, 5, 4)
        x_cat_a = torch.zeros(1, 5, 2, dtype=torch.long)
        x_cat_b = x_cat_a.clone()
        x_cat_b[:, :, 0] = 3  # different ticker id

        with torch.no_grad():
            out_a = model(x_num, x_cat_a)
            out_b = model(x_num, x_cat_b)

        assert not torch.allclose(out_a, out_b)
