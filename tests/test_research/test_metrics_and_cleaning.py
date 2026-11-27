import numpy as np
import pandas as pd
import pytest

from src.psx_predictor.research import data, metrics


def test_max_drawdown_and_return_known_series():
    nav = pd.Series([100, 120, 90, 135], index=pd.bdate_range("2021-01-04", periods=4))
    m = metrics.perf(nav)
    assert m["max_drawdown_pct"] == pytest.approx(-25.0)
    assert m["net_return_pct"] == pytest.approx(35.0)


def test_deflated_sharpe_falls_as_more_variants_are_tried():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0008, 0.01, 1500))
    few = metrics.deflated_sharpe(r, [metrics.sharpe_daily(r), 0.0, 0.01])
    many = metrics.deflated_sharpe(r, [metrics.sharpe_daily(r)] + list(rng.normal(0, 0.03, 200)))
    assert many < few


def test_calibration_buckets_count_everything():
    p = np.array([0.42, 0.52, 0.58, 0.61, 0.7])
    y = np.array([0, 1, 0, 1, 1])
    rows = metrics.calibration_table(p, y)
    assert sum(r["n"] for r in rows) == 5


def test_impossible_print_is_neutralised_and_untradeable(synthetic_df):
    df, meta, cfg = synthetic_df
    df = df.copy()
    d0 = df.date.unique()[300]
    i = df.index[(df.ticker == "AAA") & (df.date == d0)][0]
    df.loc[i, "close"] *= 5            # spike...
    p = data.build_panel(df, meta, cfg)
    assert p.bad.loc[d0, "AAA"]
    assert not p.can_buy.loc[d0, "AAA"]
    nxt = p.close.index[p.close.index.get_loc(d0) + 1]
    assert p.bad.loc[nxt, "AAA"]       # ...and its reversal
    assert p.ret["AAA"].abs().max() < 0.11


def test_pit_universe_uses_only_past_month_ends(synthetic_df):
    df, meta, cfg = synthetic_df
    p = data.build_panel(df, meta, cfg)
    m = data.pit_universe(p, top_n=2, min_history=60)
    assert m.dtypes.eq(bool).all()
    assert (m.sum(axis=1) <= 2).all()
    # first trading day of a month carries the decision of the previous month end
    idx = p.close.index
    first_days = pd.Series(idx, index=idx).groupby(idx.to_period("M")).min()
    d = first_days.iloc[6]
    prev_me = idx[idx.get_loc(d) - 1]
    assert m.loc[d].equals(m.loc[prev_me:].iloc[1])


def test_calendar_drops_holiday_rows(synthetic_df):
    df, meta, cfg = synthetic_df
    dates = pd.DatetimeIndex(sorted(df.date.unique()))
    cal = dates.delete([10, 20])
    p = data.build_panel(df, meta, cfg, calendar=cal)
    assert len(p.close) == len(dates) - 2 and len(p.holidays) == 2
