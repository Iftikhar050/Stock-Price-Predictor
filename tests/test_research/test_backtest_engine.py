"""Known-answer cases for the research backtest engine."""
import numpy as np
import pandas as pd
import pytest

from src.psx_predictor.research import backtest
from tests.test_research.conftest import EXEC, ZERO_COSTS, bt_inputs


def run(prices, targets_by_idx, costs=ZERO_COSTS, execc=EXEC, **kw):
    dates, tick, ret, cb, cs, adv = bt_inputs(prices, **kw)
    tg = {pd.Timestamp(dates[i]): pd.Series(w, index=tick) for i, w in targets_by_idx.items()}
    return backtest.run(dates, tick, ret, cb, cs, adv, tg, costs, execc, dates[0], dates[-1])


def test_buy_and_hold_fills_at_next_close_without_costs():
    res = run([100, 110, 121, 133.1], {0: [1.0]})
    # signal on day 0 fills at day 1's close (110): 100 * 133.1 / 110
    assert res.nav.iloc[-1] == pytest.approx(121.0)


def test_round_trip_cost_arithmetic():
    c = {**ZERO_COSTS, "commission": 0.001, "sales_tax_on_commission": 0.5,
         "exchange_clearing_cdc_secp": 0.0005, "cvt_on_buys": 0.0002, "slippage_by_adv": [[0, 0.001]]}
    res = run([100, 100, 100, 100, 100], {0: [1.0], 2: [0.0]}, costs=c)
    base = 0.001 * 1.5 + 0.0005
    h = 100 / (1 + base + 0.001 + 0.0002)
    assert res.nav.iloc[-1] == pytest.approx(h * (1 - base - 0.001))
    assert len(res.trades) == 1 and res.trades.hold_days.iloc[0] == 2


def test_untradeable_day_defers_the_fill():
    blocked = [[True], [False], [True], [True]]
    res = run([100, 100, 120, 120], {0: [1.0]}, can_buy=blocked)
    assert res.nav.iloc[-1] == pytest.approx(100.0)       # bought at 120, missed the jump
    res = run([100, 100, 120, 120], {0: [1.0]})
    assert res.nav.iloc[-1] == pytest.approx(120.0)


def test_liquidity_cap_limits_daily_fill():
    execc = {**EXEC, "account_pkr": 10_000_000, "max_adv_fraction": 0.10}
    res = run([100] * 4, {0: [1.0]}, execc=execc, adv=[[1_000_000]] * 4)
    # 10% of PKR 1m ADV = PKR 100k per day = 1% of a PKR 10m account
    assert res.weights.iloc[1, 0] == pytest.approx(0.01, rel=1e-6)
    assert res.weights.iloc[2, 0] == pytest.approx(0.02, rel=1e-6)


def test_never_levered_with_costs():
    c = {**ZERO_COSTS, "commission": 0.003, "slippage_by_adv": [[0, 0.005]]}
    rng = np.random.default_rng(1)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, (60, 3)), axis=0)
    tg = {i: list(rng.dirichlet(np.ones(3))) for i in range(0, 60, 5)}
    res = run(px, tg, costs=c)
    assert (res.weights.sum(axis=1) <= 1 + 1e-9).all()


def test_one_day_peek_is_not_rewarded():
    """A signal that 'knows' day t+1's move is only filled at t+1's close, so it
    earns day t+2's (independent) move: no edge should survive."""
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.02, 800)
    px = 100 * np.cumprod(1 + r)
    tg = {i: [1.0 if r[i + 1] > 0 else 0.0] for i in range(len(r) - 1)}
    res = run(px, tg)
    daily = res.nav.pct_change().dropna()
    oracle = np.prod(1 + np.where(r[1:] > 0, r[1:], 0))
    assert res.nav.iloc[-1] / 100 < 0.05 * oracle
    assert abs(daily.mean() / daily.std() * np.sqrt(250)) < 1.5
