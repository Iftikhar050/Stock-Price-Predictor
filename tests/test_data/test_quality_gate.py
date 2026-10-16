import pandas as pd
import pytest

from src.psx_predictor.data.quality_gate import (
    check_adjusted_close,
    check_constant_zero_columns,
    check_ohlc_integrity,
    check_placeholder_columns,
    check_synthetic_flags,
    check_zero_volume_days,
    run_quality_checks,
)


def _price_df(n=400, **extra_cols):
    """n rows of plausible daily OHLCV, plus whatever extra columns a test needs."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "open": [100.0 + i * 0.1 for i in range(n)],
        "high": [101.0 + i * 0.1 for i in range(n)],
        "low": [99.0 + i * 0.1 for i in range(n)],
        "close": [100.5 + i * 0.1 for i in range(n)],
        "volume": [1000] * n,
    })
    for col, values in extra_cols.items():
        df[col] = values
    return df


def test_run_quality_checks_empty_df_fails():
    report = run_quality_checks(pd.DataFrame(), ticker="EMPTY")
    assert not report.passed
    assert "DataFrame is empty." in report.failures


def test_run_quality_checks_clean_data_passes():
    df = _price_df(n=400, revenue=[1000.0 + i for i in range(400)])
    report = run_quality_checks(df, ticker="CLEAN")
    assert report.passed
    assert report.total_rows == 400


def test_check_placeholder_columns_flags_frozen_fundamental():
    # ~1.6 years of data, revenue never changes -> should be flagged as
    # likely placeholder/frozen, matching the real PSO/MEBL defect this
    # check was written to catch.
    df = _price_df(n=400, revenue=[50000.0] * 400)
    failures, warnings = check_placeholder_columns(df, years_covered=400 / 252.0)
    assert not failures
    assert any("revenue" in w and "placeholder/frozen" in w for w in warnings)


def test_check_placeholder_columns_passes_when_value_moves_quarterly():
    n = 400
    years = n / 252.0
    # Revenue steps every ~63 trading days (~quarterly) - enough changes to
    # clear the 30%-tolerance threshold for 3 changes/year.
    revenue = [50000.0 + 1000.0 * (i // 63) for i in range(n)]
    failures, warnings = check_placeholder_columns(df := _price_df(n=n, revenue=revenue), years_covered=years)
    assert not any("revenue" in w for w in warnings)


def test_check_constant_zero_columns_flags_all_zero_signal_column():
    df = _price_df(n=100, sentiment_score=[0.0] * 100)
    failures, warnings = check_constant_zero_columns(df)
    assert not failures
    assert any("sentiment_score" in w and "no signal" in w for w in warnings)


def test_check_constant_zero_columns_passes_with_some_signal():
    values = [0.0] * 90 + [0.5] * 10
    df = _price_df(n=100, sentiment_score=values)
    _, warnings = check_constant_zero_columns(df)
    assert not any("sentiment_score" in w for w in warnings)


def test_check_synthetic_flags_warns_above_threshold():
    df = _price_df(n=100, pakistan_activity_is_synthetic=[1] * 90 + [0] * 10)
    _, warnings = check_synthetic_flags(df, max_synthetic_pct=0.80)
    assert any("pakistan_activity_is_synthetic" in w for w in warnings)


def test_check_synthetic_flags_silent_below_threshold():
    df = _price_df(n=100, pakistan_activity_is_synthetic=[1] * 50 + [0] * 50)
    _, warnings = check_synthetic_flags(df, max_synthetic_pct=0.80)
    assert warnings == []


def test_check_adjusted_close_warns_when_mostly_null():
    df = _price_df(n=100, adjusted_close=[None] * 99 + [100.0])
    _, warnings = check_adjusted_close(df)
    assert any("adjusted_close" in w for w in warnings)


def test_check_adjusted_close_silent_when_populated():
    df = _price_df(n=100, adjusted_close=[100.0] * 100)
    _, warnings = check_adjusted_close(df)
    assert warnings == []


def test_check_ohlc_integrity_flags_close_outside_band():
    df = _price_df(n=10)
    # Force one row's close far above its high - the exact "defect #10" shape.
    df.loc[0, "close"] = df.loc[0, "high"] + 50.0
    _, warnings = check_ohlc_integrity(df)
    assert any("OHLC integrity" in w for w in warnings)


def test_check_ohlc_integrity_silent_when_consistent():
    df = _price_df(n=10)
    _, warnings = check_ohlc_integrity(df)
    assert warnings == []


def test_check_zero_volume_days_flags_excessive_suspension():
    df = _price_df(n=100)
    df.loc[:30, "volume"] = 0  # 31% zero-volume days, above the 15% threshold
    _, warnings = check_zero_volume_days(df)
    assert any("Zero-volume days" in w for w in warnings)


def test_check_zero_volume_days_silent_within_threshold():
    df = _price_df(n=100)
    df.loc[:5, "volume"] = 0  # 6%, within the 15% threshold
    _, warnings = check_zero_volume_days(df)
    assert warnings == []


def test_run_quality_checks_strict_raises_on_failure(mocker):
    # EXPECTED_VARIANCE only has "warn" severities today, so force a failure
    # path by making a hard-"fail" entry active for this test.
    mocker.patch(
        "src.psx_predictor.data.quality_gate.EXPECTED_VARIANCE",
        [("revenue", 3.0, "fail")],
    )
    df = _price_df(n=400, revenue=[50000.0] * 400)
    with pytest.raises(RuntimeError, match="Quality gate FAILED"):
        run_quality_checks(df, ticker="STRICT", strict=True)
