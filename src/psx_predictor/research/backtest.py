"""Daily long-only portfolio backtest with PSX frictions.

Execution model (stated in the report):
  * A target is decided from data up to the close of day t and filled at the close
    of day t+1. Any part that cannot fill (no volume, artefact print, locked at the
    circuit limit, above the liquidity cap) is retried on following days until it
    fills or a newer target replaces it.
  * Per-side cost = commission x (1 + sales tax) + exchange/clearing/CDC/SECP fees
    + slippage by liquidity tier, + CVT on purchases.
  * A day's trade in one name is capped at max_adv_fraction of its 20-day median
    traded value.
  * T+2: both legs settle two days later, so a sale's proceeds can fund a purchase
    made the same day (they settle together) but the account can never go below
    zero cash - i.e. no leverage and no buying with money not yet owned.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def slippage_matrix(adv, tiers):
    out = np.full(adv.shape, tiers[-1][1], dtype=float)
    a = np.nan_to_num(adv, nan=0.0)
    for min_adv, slip in sorted(tiers, key=lambda x: x[0]):
        out = np.where(a >= min_adv, slip, out)
    return out


@dataclass
class Result:
    nav: pd.Series
    weights: pd.DataFrame            # end-of-day weights
    trades: pd.DataFrame             # one row per round trip
    turnover: pd.Series              # traded value / NAV per day (one side)
    costs: pd.Series
    info: dict = field(default_factory=dict)


def run(dates, tickers, ret, can_buy, can_sell, adv, targets, costs_cfg, exec_cfg,
        start, end, initial=100.0):
    """targets: dict {signal_date: pd.Series(weights by ticker)}; weights sum <= 1."""
    T, N = len(dates), len(tickers)
    R = np.nan_to_num(np.asarray(ret, dtype=float), nan=0.0)
    CB, CS = np.asarray(can_buy, dtype=bool), np.asarray(can_sell, dtype=bool)
    ADV = np.nan_to_num(np.asarray(adv, dtype=float), nan=0.0)
    SLIP = slippage_matrix(ADV, costs_cfg["slippage_by_adv"])
    base = costs_cfg["commission"] * (1 + costs_cfg["sales_tax_on_commission"]) \
        + costs_cfg["exchange_clearing_cdc_secp"]
    cvt = costs_cfg["cvt_on_buys"]
    # liquidity caps are in PKR; scale the account so caps bind realistically
    scale = exec_cfg["account_pkr"] / initial
    cap_frac = exec_cfg["max_adv_fraction"]

    idx = {d: i for i, d in enumerate(dates)}
    sig = {}
    for d, w in targets.items():
        if d in idx:
            sig[idx[d]] = w.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    t0 = int(np.searchsorted(dates, np.datetime64(start)))
    t1 = int(np.searchsorted(dates, np.datetime64(end), side="right")) - 1

    h = np.zeros(N)
    cash = initial
    pending = None
    navs, wts, tos, cst = [], [], [], []
    open_t = np.full(N, -1)
    pnl_net = np.zeros(N)
    pnl_gross = np.zeros(N)
    trades = []
    for t in range(t0, t1 + 1):
        gain = h * R[t]
        h = h + gain
        pnl_net += gain
        pnl_gross += gain
        nav = cash + h.sum()
        traded_val = 0.0
        day_cost = 0.0
        if (t - 1) in sig:
            pending = sig[t - 1]
        if pending is not None and nav > 0:
            cap = cap_frac * ADV[t] / scale
            delta = pending * nav - h
            # sells first
            sell = np.where((delta < 0) & CS[t], np.minimum(-delta, cap), 0.0)
            # a full exit is allowed to exceed the cap only up to the position size
            sell = np.minimum(sell, h)
            c_s = sell * (base + SLIP[t])
            h -= sell
            cash += sell.sum() - c_s.sum()
            pnl_net -= c_s
            # then buys, limited by cash (no leverage)
            want = np.where((delta > 0) & CB[t], np.minimum(delta, cap), 0.0)
            unit = 1 + base + SLIP[t] + cvt
            need = (want * unit).sum()
            if need > cash > 0:
                want *= cash / need
            elif cash <= 0:
                want[:] = 0
            c_b = want * (unit - 1)
            h += want
            cash -= want.sum() + c_b.sum()
            pnl_net -= c_b
            traded_val = sell.sum() + want.sum()
            day_cost = c_s.sum() + c_b.sum()
            nav = cash + h.sum()
            resid = np.abs(pending * nav - h).sum()
            if resid < 0.005 * nav:
                pending = None
        # round-trip bookkeeping
        held = h > 1e-9 * max(nav, 1e-9)
        opened = held & (open_t < 0)
        open_t[opened] = t
        closed = (~held) & (open_t >= 0)
        for i in np.where(closed)[0]:
            trades.append((tickers[i], dates[open_t[i]], dates[t], t - open_t[i],
                           pnl_net[i], pnl_gross[i]))
            open_t[i] = -1
        pnl_net[~held] = 0.0
        pnl_gross[~held] = 0.0
        navs.append(nav)
        wts.append(h / nav if nav > 0 else h * 0)
        tos.append(traded_val / 2 / nav if nav > 0 else 0)
        cst.append(day_cost)
    for i in np.where(open_t >= 0)[0]:   # mark still-open trades at the end
        trades.append((tickers[i], dates[open_t[i]], dates[t1], t1 - open_t[i],
                       pnl_net[i], pnl_gross[i]))
    rng = pd.DatetimeIndex(dates[t0:t1 + 1])
    tr = pd.DataFrame(trades, columns=["ticker", "entry", "exit", "hold_days", "pnl_net", "pnl_gross"])
    return Result(pd.Series(navs, index=rng), pd.DataFrame(wts, index=rng, columns=tickers),
                  tr, pd.Series(tos, index=rng), pd.Series(cst, index=rng))
