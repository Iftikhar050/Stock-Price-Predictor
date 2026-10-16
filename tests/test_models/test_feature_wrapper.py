import numpy as np
import pandas as pd
import pytest

from src.psx_predictor.models.feature_wrapper import (
    bin_volatility_regime,
    get_feature_columns,
    make_dead_zone_labels,
    prepare_data,
)


def _write_master_csv(tmp_path, mocker, ticker="TEST", n=30, volume=None, close=None):
    mocker.patch("src.psx_predictor.models.feature_wrapper.BASE_FEATURES_DIR", str(tmp_path))
    close = close if close is not None else [100.0 + i for i in range(n)]
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B").astype(str),
        "ticker": [ticker] * n,
        "close": close,
        "volume": volume if volume is not None else [1000] * n,
        "feat_a": np.arange(n, dtype=float),
        "feat_b": np.arange(n, dtype=float) * 2,
        "text_col": ["unavailable"] * n,  # must be excluded - non-numeric
    })
    df.to_csv(tmp_path / f"{ticker}_master.csv", index=False)
    return df


class TestGetFeatureColumns:
    def test_excludes_non_numeric_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "ticker": ["PSO", "PSO"]})
        cols = get_feature_columns(df)
        assert "b" not in cols
        assert "a" in cols

    def test_keeps_ticker_even_though_non_numeric(self):
        df = pd.DataFrame({"a": [1, 2], "ticker": ["PSO", "PSO"]})
        cols = get_feature_columns(df)
        assert "ticker" in cols

    def test_respects_explicit_exclude_cols(self):
        df = pd.DataFrame({"a": [1, 2], "close": [10.0, 11.0]})
        cols = get_feature_columns(df, exclude_cols=["close"])
        assert "close" not in cols
        assert "a" in cols


class TestMakeDeadZoneLabels:
    def test_classifies_up_flat_down(self):
        returns = pd.Series([0.02, 0.001, -0.02, -0.001, 0.0])
        labels = make_dead_zone_labels(returns, dead_zone=0.005)
        assert list(labels) == [1, 0, -1, 0, 0]


class TestBinVolatilityRegime:
    def test_fits_edges_on_train_and_applies_to_test(self):
        train_vol = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        test_vol = pd.Series([0.5, 3.5, 100.0])  # below, mid, far-above train range
        labels, edges = bin_volatility_regime(train_vol, test_vol, n_bins=3)
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        # Extremes never fall outside the bin range thanks to the +/-inf edges.
        assert not labels.isna().any()
        assert labels.iloc[0] == 0   # below all train values -> lowest bin
        assert labels.iloc[2] == 2   # far above all train values -> highest bin


class TestPrepareData:
    def test_raises_for_missing_ticker_file(self, tmp_path, mocker):
        mocker.patch("src.psx_predictor.models.feature_wrapper.BASE_FEATURES_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            prepare_data("NOPE")

    def test_target_is_forward_return_over_horizon(self, tmp_path, mocker):
        close = [100.0, 110.0, 121.0, 100.0]  # +10%, +10%, -17.355...%
        _write_master_csv(tmp_path, mocker, ticker="TEST", n=4, close=close)
        X, y, dates, current_close = prepare_data("TEST", horizon=1)
        # Last row has no t+1 to compute a target from, so it's dropped.
        assert len(y) == 3
        assert y.iloc[0] == pytest.approx((110.0 - 100.0) / 100.0)
        assert y.iloc[1] == pytest.approx((121.0 - 110.0) / 110.0)

    def test_text_columns_are_excluded_from_features(self, tmp_path, mocker):
        _write_master_csv(tmp_path, mocker, ticker="TEST", n=10)
        X, y, dates, current_close = prepare_data("TEST")
        assert "text_col" not in X.columns
        assert "feat_a" in X.columns

    def test_exclude_known_bad_dates_drops_rows_spanning_zero_volume(self, tmp_path, mocker):
        n = 10
        volume = [1000] * n
        volume[5] = 0  # a stale/non-trading day at index 5
        _write_master_csv(tmp_path, mocker, ticker="TEST", n=n, volume=volume)

        # horizon=1: a row's target is tainted if EITHER its own day or the
        # next day is the stale one, i.e. rows at index 4 and 5 get dropped.
        X, y, dates, current_close = prepare_data("TEST", horizon=1, exclude_known_bad_dates=True)
        kept_count_with_filter = len(y)

        X2, y2, _, _ = prepare_data("TEST", horizon=1, exclude_known_bad_dates=False)
        kept_count_without_filter = len(y2)

        assert kept_count_with_filter == kept_count_without_filter - 2

    def test_dead_zone_attaches_target_class(self, tmp_path, mocker):
        _write_master_csv(tmp_path, mocker, ticker="TEST", n=10)
        result = prepare_data("TEST", dead_zone=0.005)
        assert len(result) == 5
        X, y, dates, current_close, target_class = result
        assert set(target_class.unique()).issubset({-1, 0, 1})

    def test_inf_and_nan_feature_values_are_cleaned(self, tmp_path, mocker):
        mocker.patch("src.psx_predictor.models.feature_wrapper.BASE_FEATURES_DIR", str(tmp_path))
        n = 5
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B").astype(str),
            "ticker": ["TEST"] * n,
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            "volume": [1000] * n,
            "feat_a": [1.0, np.inf, -np.inf, np.nan, 5.0],
        })
        df.to_csv(tmp_path / "TEST_master.csv", index=False)

        X, y, dates, current_close = prepare_data("TEST")
        assert np.isfinite(X["feat_a"]).all()
        assert not X["feat_a"].isna().any()
