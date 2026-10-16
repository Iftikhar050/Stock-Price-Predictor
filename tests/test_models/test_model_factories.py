import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.psx_predictor.models.model_factories import (
    baseline_factory,
    lightgbm_factory,
    lstm_factory,
    ridge_factory,
    svr_rbf_factory,
    xgboost_factory,
)


def _toy_data(n=30, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.normal(scale=0.01, size=n)
    return X, y


class TestBaselineFactory:
    def test_returns_random_forest_with_expected_hyperparams(self):
        model = baseline_factory()
        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 100
        assert model.max_depth == 10
        assert model.random_state == 42

    def test_each_call_returns_an_independent_instance(self):
        a, b = baseline_factory(), baseline_factory()
        assert a is not b

    def test_fits_and_predicts_on_toy_data(self):
        X, y = _toy_data()
        model = baseline_factory()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape


class TestRidgeFactory:
    def test_returns_ridge_with_default_alpha(self):
        model = ridge_factory()
        assert isinstance(model, Ridge)
        assert model.alpha == 1.0

    def test_fits_and_predicts(self):
        X, y = _toy_data()
        model = ridge_factory()
        model.fit(X, y)
        assert model.predict(X).shape == y.shape


class TestSvrRbfFactory:
    def test_default_epsilon_is_far_below_sklearn_default(self):
        """The whole point of this factory - sklearn's SVR default epsilon
        (0.1) swallows the target's entire ~1-2% scale and collapses the fit
        to the bias term; this factory's default must stay well below that."""
        model = svr_rbf_factory()
        assert model.epsilon == 0.001
        assert model.epsilon < 0.1

    def test_custom_epsilon_is_respected(self):
        model = svr_rbf_factory(epsilon=0.02)
        assert model.epsilon == 0.02

    def test_returns_rbf_kernel_svr(self):
        model = svr_rbf_factory()
        assert isinstance(model, SVR)
        assert model.kernel == "rbf"


class TestXgboostFactory:
    def test_returns_xgbregressor_with_expected_hyperparams(self):
        model = xgboost_factory()
        assert isinstance(model, XGBRegressor)
        assert model.n_estimators == 200
        assert model.max_depth == 6
        assert model.learning_rate == 0.05

    def test_fits_and_predicts(self):
        X, y = _toy_data()
        model = xgboost_factory()
        model.fit(X, y)
        assert model.predict(X).shape == y.shape


class TestLightgbmFactory:
    def test_returns_configured_lgbm_regressor(self):
        model = lightgbm_factory()
        assert model.n_estimators == 200
        assert model.max_depth == 6

    def test_fits_and_predicts(self):
        X, y = _toy_data()
        model = lightgbm_factory()
        model.fit(X, y)
        assert model.predict(X).shape == y.shape

    def test_raises_clear_error_when_lightgbm_unavailable(self, mocker):
        mocker.patch("src.psx_predictor.models.model_factories.LGBMRegressor", None)
        with pytest.raises(ImportError, match="lightgbm"):
            lightgbm_factory()


class TestLstmFactory:
    def test_returns_model_with_correct_input_dim(self):
        import torch

        model = lstm_factory(input_dim=10)
        batch, seq_len = 4, 5
        x = torch.randn(batch, seq_len, 10)
        out = model(x)
        assert out.shape == (batch,)

