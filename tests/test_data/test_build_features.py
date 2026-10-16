import numpy as np
import pandas as pd
import pytest

from src.psx_predictor.data.build_features import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_lag_features,
    calculate_macd,
    calculate_obv,
    calculate_realized_volatility,
    calculate_relative_volume,
    calculate_rsi,
    calculate_sma,
    calculate_vwap,
    extract_time_features,
)


def _ohlcv(n=60, trend=0.0, seed=42):
    """n rows of synthetic daily OHLCV with a configurable linear drift, so
    indicator direction (e.g. RSI > 50 on an uptrend) is predictable."""
    rng = np.random.default_rng(seed)
    base = 100.0 + np.cumsum(trend + rng.normal(0, 0.3, n))
    close = base
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    open_ = close - rng.normal(0, 0.2, n)
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


class TestSMA:
    def test_adds_raw_sma_and_distance_columns(self):
        df = calculate_sma(_ohlcv(), windows=[5])
        assert "sma_5" in df.columns and "sma_5_dist" in df.columns

    def test_sma_matches_manual_rolling_mean(self):
        df = _ohlcv(n=10)
        out = calculate_sma(df.copy(), windows=[3])
        expected = df["close"].rolling(3).mean()
        pd.testing.assert_series_equal(out["sma_3"], expected, check_names=False)

    def test_distance_is_zero_when_price_equals_sma(self):
        df = pd.DataFrame({"close": [100.0] * 10})
        out = calculate_sma(df, windows=[3])
        assert out["sma_3_dist"].iloc[-1] == pytest.approx(0.0)


class TestRSI:
    def test_rsi_is_bounded_0_100(self):
        df = calculate_rsi(_ohlcv(n=100), window=14)
        valid = df["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_high_on_sustained_uptrend(self):
        df = calculate_rsi(_ohlcv(n=60, trend=0.8), window=14)
        assert df["rsi_14"].iloc[-1] > 70

    def test_rsi_low_on_sustained_downtrend(self):
        df = calculate_rsi(_ohlcv(n=60, trend=-0.8), window=14)
        assert df["rsi_14"].iloc[-1] < 30


class TestMACD:
    def test_adds_macd_columns(self):
        df = calculate_macd(_ohlcv())
        assert {"macd", "macd_signal", "macd_hist"}.issubset(df.columns)

    def test_histogram_is_macd_minus_signal(self):
        df = calculate_macd(_ohlcv())
        pd.testing.assert_series_equal(
            df["macd_hist"], df["macd"] - df["macd_signal"], check_names=False
        )


class TestLagFeatures:
    def test_daily_return_matches_pct_change(self):
        df = _ohlcv(n=20)
        out = calculate_lag_features(df.copy())
        pd.testing.assert_series_equal(out["daily_return"], df["close"].pct_change(), check_names=False)

    def test_lag_1_is_shifted_daily_return(self):
        df = calculate_lag_features(_ohlcv(n=20), lags=[1])
        pd.testing.assert_series_equal(
            df["return_lag_1"], df["daily_return"].shift(1), check_names=False
        )


class TestBollingerBands:
    def test_upper_band_above_lower_band(self):
        df = calculate_bollinger_bands(_ohlcv(n=50))
        valid = df.dropna(subset=["bollinger_hband", "bollinger_lband"])
        assert (valid["bollinger_hband"] >= valid["bollinger_lband"]).all()

    def test_middle_band_is_rolling_mean(self):
        df = _ohlcv(n=30)
        out = calculate_bollinger_bands(df.copy(), window=20)
        expected = df["close"].rolling(20).mean()
        pd.testing.assert_series_equal(out["bollinger_mavg"], expected, check_names=False)

    def test_zero_std_window_gives_zero_width_not_nan(self):
        df = pd.DataFrame({"close": [100.0] * 25})
        out = calculate_bollinger_bands(df, window=20)
        assert out["bollinger_width"].iloc[-1] == 0.0


class TestVWAP:
    def test_vwap_within_price_range_for_constant_volume(self):
        df = _ohlcv(n=30)
        out = calculate_vwap(df.copy())
        # Compute the rolling bounds on the full (pre-dropna) series so the
        # window alignment matches vwap's own 14-day window at each row -
        # slicing to valid rows first before rolling would shift the window.
        rolling_low = df["low"].rolling(14).min()
        rolling_high = df["high"].rolling(14).max()
        mask = out["vwap"].notna()
        assert (out.loc[mask, "vwap"] >= rolling_low[mask]).all()
        assert (out.loc[mask, "vwap"] <= rolling_high[mask]).all()

    def test_zero_volume_window_does_not_raise_or_divide_by_zero(self):
        df = _ohlcv(n=20)
        df["volume"] = 0.0
        out = calculate_vwap(df)
        assert out["vwap"].isna().all()  # zero-volume window -> no meaningful VWAP


class TestExtractTimeFeatures:
    def test_dow_sin_cos_are_unit_circle_bounded(self):
        df = extract_time_features(_ohlcv(n=10))
        assert (df["dow_sin"].abs() <= 1.0).all()
        assert (df["dow_cos"].abs() <= 1.0).all()

    def test_same_weekday_gives_same_encoding(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-08"])})  # both Mondays
        out = extract_time_features(df)
        assert out["dow_sin"].iloc[0] == pytest.approx(out["dow_sin"].iloc[1])
        assert out["dow_cos"].iloc[0] == pytest.approx(out["dow_cos"].iloc[1])


class TestOBV:
    def test_obv_is_zero_mean_normalized(self):
        df = calculate_obv(_ohlcv(n=100), window=50)
        valid = df["obv"].dropna()
        assert not valid.empty
        assert valid.abs().max() < 100  # z-scored, not a raw cumulative sum


class TestRelativeVolume:
    def test_relative_volume_is_one_for_constant_volume(self):
        df = pd.DataFrame({"volume": [1000.0] * 30})
        out = calculate_relative_volume(df, window=10)
        assert out["relative_volume"].iloc[-1] == pytest.approx(1.0)

    def test_handles_zero_volume_average_without_division_error(self):
        df = pd.DataFrame({"volume": [0.0] * 30})
        out = calculate_relative_volume(df, window=10)
        assert not out["relative_volume"].isna().all()


class TestRealizedVolatility:
    def test_adds_columns_for_each_window(self):
        df = calculate_realized_volatility(_ohlcv(n=40), windows=[10, 20])
        assert "return_vol_10" in df.columns and "return_vol_20" in df.columns

    def test_zero_price_movement_gives_zero_volatility(self):
        df = pd.DataFrame({"close": [100.0] * 30})
        out = calculate_realized_volatility(df, windows=[10])
        assert out["return_vol_10"].iloc[-1] == pytest.approx(0.0)


class TestATR:
    def test_atr_is_non_negative(self):
        df = calculate_atr(_ohlcv(n=40))
        valid = df["atr"].dropna()
        assert (valid >= 0).all()

    def test_atr_zero_for_flat_ohlc(self):
        n = 30
        df = pd.DataFrame({"open": [100.0] * n, "high": [100.0] * n, "low": [100.0] * n, "close": [100.0] * n})
        out = calculate_atr(df, window=14)
        assert out["atr"].iloc[-1] == pytest.approx(0.0)


class TestADX:
    def test_adx_is_bounded_0_100(self):
        df = calculate_adx(_ohlcv(n=60))
        valid = df["adx"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_plus_di_dominates_on_strong_uptrend(self):
        df = calculate_adx(_ohlcv(n=60, trend=1.0))
        tail = df.dropna(subset=["plus_di", "minus_di"]).tail(10)
        assert (tail["plus_di"] > tail["minus_di"]).mean() > 0.5
