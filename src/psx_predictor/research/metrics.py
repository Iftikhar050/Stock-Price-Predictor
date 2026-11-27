"""Performance metrics, deflated Sharpe ratio and calibration tables."""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

DAYS = 250   # PSX trades ~245-250 days a year


def perf(nav, trades=None, turnover=None):
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    if len(r) < 2:
        return {}
    years = len(r) / DAYS
    total = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1 if nav.iloc[-1] > 0 else -1.0
    sd = r.std()
    down = r[r < 0].std()
    dd = (nav / nav.cummax() - 1).min()
    m = {
        "net_return_pct": 100 * total,
        "cagr_pct": 100 * cagr,
        "sharpe": float(r.mean() / sd * np.sqrt(DAYS)) if sd > 0 else 0.0,
        "sortino": float(r.mean() / down * np.sqrt(DAYS)) if down and down > 0 else 0.0,
        "max_drawdown_pct": 100 * dd,
        "volatility_pct": 100 * sd * np.sqrt(DAYS),
    }
    if trades is not None:
        n = len(trades)
        wins = trades.loc[trades.pnl_net > 0, "pnl_net"].sum()
        losses = -trades.loc[trades.pnl_net < 0, "pnl_net"].sum()
        m.update({
            "trades": int(n),
            "win_rate_pct": 100 * float((trades.pnl_net > 0).mean()) if n else None,
            "profit_factor": float(wins / losses) if losses > 0 else None,
            "avg_hold_days": float(trades.hold_days.mean()) if n else None,
            # share of round trips whose price move (before costs) was up
            "directional_accuracy_pct": 100 * float((trades.pnl_gross > 0).mean()) if n else None,
        })
    if turnover is not None and len(turnover):
        m["turnover_annual"] = float(turnover.sum() / years)
    return m


def after_cgt_cagr(nav, rate):
    """Approximate: tax at `rate` on each calendar year's positive gain."""
    yearly = nav.groupby(nav.index.year).agg(["first", "last"])
    g = yearly["last"] / yearly["first"] - 1
    g = np.where(g > 0, g * (1 - rate), g)
    years = len(nav) / DAYS
    return 100 * (np.prod(1 + g) ** (1 / years) - 1)


def sharpe_daily(r):
    r = pd.Series(r).dropna()
    return r.mean() / r.std() if r.std() > 0 else 0.0


def deflated_sharpe(r, trial_sharpes_daily):
    """Bailey & Lopez de Prado (2014). r = daily returns of the strategy;
    trial_sharpes_daily = daily Sharpe of every variant tried (incl. this one).
    Returns the probability that the true Sharpe is above what the best of
    N unskilled trials would show by chance."""
    r = pd.Series(r).dropna()
    T = len(r)
    sr = sharpe_daily(r)
    n = len(trial_sharpes_daily)
    v = np.var(trial_sharpes_daily, ddof=1) if n > 1 else 0.0
    g = 0.5772156649
    if n > 1 and v > 0:
        sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e)))
    else:
        sr0 = 0.0
    sk, ku = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def calibration_table(prob, outcome, edges=(0.0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 1.0)):
    df = pd.DataFrame({"p": prob, "y": outcome}).dropna()
    df["b"] = pd.cut(df.p, edges, include_lowest=True)
    rows = []
    for b, g in df.groupby("b", observed=True):
        rows.append({"bucket": f"{b.left:.2f}-{b.right:.2f}", "predicted": round(float(g.p.mean()), 3),
                     "actual": round(float(g.y.mean()), 3), "n": int(len(g))})
    return rows


def breakdown(nav, periods, regime):
    """Metrics per sub-period and per market regime (regime: Series of labels)."""
    r = nav.pct_change().dropna()
    out = {"periods": {}, "regimes": {}}
    for name, (a, b) in periods.items():
        s = nav.loc[a:b]
        if len(s) > 20:
            p = perf(s)
            out["periods"][name] = {k: round(p[k], 2) for k in ("net_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct")}
    lab = regime.reindex(r.index)
    for name, g in r.groupby(lab):
        if len(g) > 20:
            out["regimes"][str(name)] = {"days": int(len(g)), "ann_return_pct": round(100 * g.mean() * DAYS, 2),
                                         "sharpe": round(float(g.mean() / g.std() * np.sqrt(DAYS)), 2) if g.std() > 0 else 0}
    return out
