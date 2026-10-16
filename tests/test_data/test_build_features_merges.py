import numpy as np
import pandas as pd
import pytest

from src.psx_predictor.data.build_features import (
    calculate_computed_pdf_features,
    load_data,
    merge_circuit_breakers,
    merge_cross_sectional_return_rank,
    merge_dividends,
    merge_market_index,
    merge_sentiment,
)


def _price_df(n=30, close=None):
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "close": close if close is not None else [100.0 + i for i in range(n)],
    })


def _ohlcv_df(n=30, close=None):
    """calculate_computed_pdf_features needs full OHLCV plus daily_return
    (accessed unconditionally for the fear/greed proxies near the end of the
    function, unlike most of its other optional-column sections), unlike the
    simpler merge_* functions above which only touch date/close."""
    close = close if close is not None else [100.0 + i for i in range(n)]
    close_series = pd.Series(close, dtype=float)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": close, "high": [c + 1.0 for c in close], "low": [c - 1.0 for c in close],
        "close": close, "volume": [1000.0] * n,
        "daily_return": close_series.pct_change().fillna(0.0).to_numpy(),
    })


def _mock_engine(mocker):
    mock_conn = mocker.MagicMock()
    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False
    mocker.patch("src.psx_predictor.data.build_features.engine", mock_engine)
    return mock_engine


class TestLoadData:
    def test_returns_empty_df_without_writing_file_when_no_rows(self, mocker, tmp_path):
        _mock_engine(mocker)
        mocker.patch("pandas.read_sql", return_value=pd.DataFrame())
        mocker.patch("src.psx_predictor.data.build_features.PROCESSED_DIR", str(tmp_path))

        df = load_data("NOPE")

        assert df.empty
        assert not (tmp_path / "nope_cleaned.csv").exists()

    def test_sorts_chronologically_and_writes_cleaned_csv(self, mocker, tmp_path):
        _mock_engine(mocker)
        unsorted = pd.DataFrame({
            "ticker": ["PSO", "PSO"],
            "date": ["2024-01-02", "2024-01-01"],
            "close": [101.0, 100.0],
        })
        mocker.patch("pandas.read_sql", return_value=unsorted)
        mocker.patch("src.psx_predictor.data.build_features.PROCESSED_DIR", str(tmp_path))

        df = load_data("PSO")

        assert list(df["close"]) == [100.0, 101.0]  # now chronological
        assert (tmp_path / "pso_cleaned.csv").exists()


class TestMergeSentiment:
    def test_fills_zero_when_no_sentiment_data(self, mocker):
        _mock_engine(mocker)
        mocker.patch("pandas.read_sql", return_value=pd.DataFrame())
        df = merge_sentiment(_price_df(n=5), "PSO")
        assert (df["sentiment_score"] == 0.0).all()

    def test_applies_three_day_decay_for_missing_days(self, mocker):
        """Fixed 2026-10-13: apply_decay() was defined but never actually
        called, so sentiment_score was left as raw NaN on non-news days
        instead of being decayed as the docstring promises - harmless in
        production only because sentiment_score was (and still is) dropped
        via DEAD_COLUMNS before the final master CSV, but dead code
        nonetheless. Now verifies the real decay math."""
        _mock_engine(mocker)
        df = _price_df(n=5)
        sentiment_df = pd.DataFrame({
            "date": [df["date"].iloc[0]],  # only day 0 has real sentiment
            "sentiment_score": [0.8],
        })
        pucars_empty = pd.DataFrame()
        mocker.patch("pandas.read_sql", side_effect=[sentiment_df, pucars_empty])

        out = merge_sentiment(df, "PSO")

        assert out["sentiment_score"].iloc[0] == pytest.approx(0.8)
        assert out["sentiment_score"].iloc[1] == pytest.approx(0.8 * 0.5)    # lag-1 decay
        assert out["sentiment_score"].iloc[2] == pytest.approx(0.8 * 0.25)   # lag-2 decay
        assert out["sentiment_score"].iloc[3] == pytest.approx(0.8 * 0.125)  # lag-3 decay
        assert out["sentiment_score"].iloc[4] == 0.0  # beyond the decay window

    def test_pucars_sentiment_overrides_stock_news_sentiment_when_nonzero(self, mocker):
        _mock_engine(mocker)
        df = _price_df(n=3)
        sentiment_df = pd.DataFrame({"date": df["date"], "sentiment_score": [0.1, 0.1, 0.1]})
        pucars_df = pd.DataFrame({"date": [df["date"].iloc[1]], "pucars_sentiment": [0.9]})
        mocker.patch("pandas.read_sql", side_effect=[sentiment_df, pucars_df])

        out = merge_sentiment(df, "PSO")

        assert out["sentiment_score"].iloc[0] == pytest.approx(0.1)
        assert out["sentiment_score"].iloc[1] == pytest.approx(0.9)  # overridden by PUCARS
        assert "pucars_sentiment" not in out.columns  # intermediate column dropped


class TestMergeDividends:
    def test_defaults_when_no_dividend_history(self, mocker):
        _mock_engine(mocker)
        mocker.patch("pandas.read_sql", return_value=pd.DataFrame())
        df = merge_dividends(_price_df(n=5), "PSO")
        assert (df["days_since_dividend"] == 9999).all()
        assert (df["dividend_yield"] == 0.0).all()

    def test_days_since_dividend_counts_forward_from_ex_date(self, mocker):
        _mock_engine(mocker)
        df = _price_df(n=10)
        div_df = pd.DataFrame({"date": [df["date"].iloc[2]], "dividend_amount": [5.0]})
        mocker.patch("pandas.read_sql", return_value=div_df)

        out = merge_dividends(df, "PSO")

        assert out["days_since_dividend"].iloc[2] == 0
        # index 2 is a Thursday ex-date; index 3 is the next business day (Friday) -> 1 calendar day later
        assert out["days_since_dividend"].iloc[3] == (df["date"].iloc[3] - df["date"].iloc[2]).days
        assert out["days_since_dividend"].iloc[0] == 365  # before any dividend -> capped default

    def test_dividend_yield_uses_last_dividend_over_close(self, mocker):
        _mock_engine(mocker)
        df = _price_df(n=5, close=[100.0] * 5)
        div_df = pd.DataFrame({"date": [df["date"].iloc[1]], "dividend_amount": [2.0]})
        mocker.patch("pandas.read_sql", return_value=div_df)

        out = merge_dividends(df, "PSO")

        assert out["dividend_yield"].iloc[1] == pytest.approx(2.0 / 100.0)
        assert out["dividend_yield"].iloc[4] == pytest.approx(2.0 / 100.0)  # forward-filled


class TestMergeCircuitBreakers:
    def test_no_events_gives_all_zero_flags(self, mocker):
        _mock_engine(mocker)
        mocker.patch("pandas.read_sql", return_value=pd.DataFrame())
        df = merge_circuit_breakers(_price_df(n=5), "PSO")
        assert (df["hit_upper_circuit"] == 0).all()
        assert (df["hit_lower_circuit"] == 0).all()

    def test_flags_correct_day_and_direction(self, mocker):
        _mock_engine(mocker)
        df = _price_df(n=5)
        cb_df = pd.DataFrame({
            "date": [df["date"].iloc[1], df["date"].iloc[3]],
            "direction": ["upper", "lower"],
        })
        mocker.patch("pandas.read_sql", return_value=cb_df)

        out = merge_circuit_breakers(df, "PSO")

        assert out["hit_upper_circuit"].tolist() == [0, 1, 0, 0, 0]
        assert out["hit_lower_circuit"].tolist() == [0, 0, 0, 1, 0]


class TestMergeCrossSectionalReturnRank:
    def test_defaults_to_midpoint_when_ticker_not_in_universe(self, mocker):
        _mock_engine(mocker)
        mocker.patch("pandas.read_sql", return_value=pd.DataFrame())
        df = merge_cross_sectional_return_rank(_price_df(n=5), "PSO")
        assert (df["return_rank_vs_sector_20d"] == 0.5).all()
        assert (df["return_rank_vs_market_20d"] == 0.5).all()

    def test_ranks_ticker_against_sector_and_market_peers(self, mocker):
        _mock_engine(mocker)
        dates = pd.date_range("2024-01-01", periods=25, freq="B")
        rows = []
        # PSO (Energy) is the strongest performer; HBL (Banking) and OGDC
        # (Energy) are weaker - PSO should rank highest in both groups.
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": "PSO", "close": 100.0 * (1.05 ** i), "sector": "Energy"})
            rows.append({"date": d, "ticker": "OGDC", "close": 100.0 * (1.01 ** i), "sector": "Energy"})
            rows.append({"date": d, "ticker": "HBL", "close": 100.0 * (1.001 ** i), "sector": "Banking"})
        mocker.patch("pandas.read_sql", return_value=pd.DataFrame(rows))

        df = merge_cross_sectional_return_rank(_price_df(n=25).assign(date=dates), "PSO", window=20)

        last = df.iloc[-1]
        assert last["return_rank_vs_market_20d"] == pytest.approx(1.0)
        assert last["return_rank_vs_sector_20d"] == pytest.approx(1.0)


class TestMergeMarketIndex:
    def test_no_other_tickers_gives_zeroed_market_features(self, mocker):
        _mock_engine(mocker)
        mocker.patch("pandas.read_sql", return_value=pd.DataFrame())
        df = merge_market_index(_price_df(n=5), "PSO")
        assert (df["market_return"] == 0.0).all()
        assert (df["sector_return"] == 0.0).all()

    def test_falls_back_to_market_return_when_no_sector_peers(self, mocker):
        _mock_engine(mocker)
        df = _price_df(n=10)
        idx_df = pd.DataFrame({"date": df["date"], "market_close": [200.0 + i for i in range(10)]})

        # 1st call: market index, 2nd: breadth (raises inside try/except so make it empty), 3rd: sector peers (empty)
        mocker.patch("pandas.read_sql", side_effect=[idx_df, pd.DataFrame(), pd.DataFrame()])

        out = merge_market_index(df, "PSO")

        pd.testing.assert_series_equal(
            out["sector_return"], out["market_return"], check_names=False
        )


class TestCalculateComputedPdfFeatures:
    def test_momentum_returns_match_pct_change(self):
        df = _ohlcv_df(n=250, close=[100.0 * (1.001 ** i) for i in range(250)])
        out = calculate_computed_pdf_features(df.copy())
        expected_3d = df["close"].pct_change(periods=3).fillna(0.0)
        pd.testing.assert_series_equal(out["return_3d"], expected_3d, check_names=False)

    def test_max_drawdown_is_zero_on_monotonic_uptrend(self):
        df = _ohlcv_df(n=260, close=[100.0 + i for i in range(260)])
        out = calculate_computed_pdf_features(df.copy())
        assert (out["max_drawdown_252d"] <= 1e-9).all()

    def test_max_drawdown_is_negative_after_a_decline(self):
        close = [100.0] * 30 + [50.0] * 30  # halves in value
        df = _ohlcv_df(n=60, close=close)
        out = calculate_computed_pdf_features(df.copy())
        assert out["max_drawdown_252d"].iloc[-1] == pytest.approx(-0.5)

    def test_beta_defaults_to_one_without_market_return(self):
        df = _ohlcv_df(n=30)
        out = calculate_computed_pdf_features(df.copy())
        assert (out["beta_60d"] == 1.0).all()
        assert out["corr_stock_kse100"].iloc[-1] == 0.0

    def test_flat_market_series_gives_zero_correlation_not_inf(self):
        """A zero-variance rolling window produces +/-inf from pandas' own
        correlation math, not NaN - the fillna(0.0) alone wouldn't catch
        that without the explicit replace([inf,-inf], nan) step first."""
        n = 80
        df = _ohlcv_df(n=n, close=[100.0 + i * 0.5 for i in range(n)])
        df["daily_return"] = df["close"].pct_change()
        df["market_return"] = 0.0  # perfectly flat -> undefined correlation

        out = calculate_computed_pdf_features(df)
        assert not np.isinf(out["corr_stock_kse100"]).any()
        assert (out["corr_stock_kse100"] == 0.0).all()
