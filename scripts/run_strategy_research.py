"""
Strategy research run: builds every candidate strategy, backtests it with PSX
frictions, corrects for the number of variants tried, and writes
frontend/results.json + reports/research/*.

  python scripts/run_strategy_research.py                 # holdout stays hidden
  python scripts/run_strategy_research.py --reveal-holdout  # the one final run

Read-only against the database. Settings: config/research.yaml.
"""
import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
warnings.filterwarnings("ignore")

from src.psx_predictor.db.connection import engine  # noqa: E402
from src.psx_predictor.research import backtest, data, metrics, models, signals  # noqa: E402

OUT_JSON = ROOT / "frontend" / "results.json"
OUT_DIR = ROOT / "reports" / "research"
CACHE = ROOT / "data" / "research_cache"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def subset(panel, cols):
    f = {k: getattr(panel, k) for k in ("close", "ret", "raw_ret", "volume", "adv", "bad",
                                         "can_buy", "can_sell", "listed")}
    return replace(panel, **{k: v[cols] for k, v in f.items()}, sector=panel.sector[cols])


def lgbm(cfg, seed, cls=False):
    import lightgbm as lgb
    p = dict(cfg["models"]["lgbm"], random_state=seed)
    return (lambda: lgb.LGBMClassifier(**p)) if cls else (lambda: lgb.LGBMRegressor(**p))


def rebalance_dates(dates, start, every):
    s0 = int(np.searchsorted(dates, np.datetime64(start))) - 1   # fill lands on `start`
    return [pd.Timestamp(dates[i]) for i in range(max(s0, 0), len(dates), every)]


def portfolio_targets(score, eligible, rdates, n, buffer=None, weight="equal", vol=None,
                      exposure=None):
    targets, held = {}, []
    for d in rdates:
        if d not in score.index:
            continue
        row = score.loc[d].where(eligible.loc[d])
        held = signals.select(row, n, held if buffer else None, buffer)
        if not held:
            targets[d] = pd.Series(dtype=float)
            continue
        if weight == "invvol" and vol is not None:
            iv = 1 / vol.loc[d, held].replace(0, np.nan)
            w = (iv / iv.sum()).fillna(0)
        else:
            w = pd.Series(1 / len(held), index=held)
        e = 1.0 if exposure is None else float(exposure.get(d, 1.0))
        targets[d] = w * e
    return targets


def run_engine(panel, targets, cfg, start, end):
    return backtest.run(panel.close.index.values, list(panel.close.columns), panel.ret.values,
                        panel.can_buy.values, panel.can_sell.values, panel.adv.values,
                        targets, cfg["costs"], cfg["execution"], start, end)


def downsample(s, max_points=300):
    s = s.dropna()
    k = max(1, math.ceil(len(s) / max_points))
    pts = s.iloc[::k]
    if pts.index[-1] != s.index[-1]:
        pts = pd.concat([pts, s.iloc[[-1]]])
    base = s.iloc[0]
    return [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v / base * 100), 2)} for d, v in pts.items()]


def r2(x, d=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reveal-holdout", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "config" / "research.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    reveal = args.reveal_holdout or cfg["periods"]["reveal_holdout"]
    seed = cfg["seed"]
    np.random.seed(seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    oos, hold, end = (pd.Timestamp(cfg["periods"]["oos_start"]), pd.Timestamp(cfg["periods"]["holdout_start"]),
                      pd.Timestamp(cfg["data"]["end"]))
    dev_end = hold - pd.Timedelta(days=1)
    report_end = end if reveal else dev_end

    # ---------------- data ----------------
    log("loading prices")
    df, meta = data.load_eod(engine, cfg["data"]["start"], cfg["data"]["end"])
    kse = data.load_kse100(cfg["data"]["kse100_csv"])
    full = data.build_panel(df, meta, cfg, calendar=kse.index)
    log(f"panel {full.close.shape}, holidays dropped {len(full.holidays)}, "
        f"artefact prints {int(full.bad.values.sum())}")
    pit = data.pit_universe(full, cfg["universe"]["pit_top_n"], cfg["universe"]["min_history_days"])
    pit_cols = list(pit.columns[pit.loc[oos - pd.Timedelta(days=400):].any()])
    cols = sorted(set(full.active) | set(pit_cols))
    P = subset(full, cols)
    dates = P.close.index.values
    feats = signals.features(P, kse)
    hist_ok = P.listed.cumsum() >= cfg["universe"]["min_history_days"]
    liquid = P.adv >= cfg["universe"]["min_adv_pkr"]
    elig_active = liquid & hist_ok & pd.DataFrame(P.close.columns.isin(full.active)[None, :].repeat(len(dates), 0),
                                                   index=P.close.index, columns=P.close.columns)
    elig_pit = liquid & hist_ok & pit.reindex(index=P.close.index, columns=cols).fillna(False).astype(bool)
    A = subset(P, full.active)
    elig_A = elig_active[full.active]

    regime_trend = (kse / kse.rolling(200, min_periods=160).mean() - 1).reindex(P.close.index).ffill()
    exposure_flag = regime_trend >= 0
    kvol = np.log(kse).diff().rolling(60).std().reindex(P.close.index).ffill()
    regime_vol = pd.Series(np.where(kvol > kvol.expanding(250).median(), "volatile", "calm"), index=P.close.index)
    regime_dir = pd.Series(np.where(exposure_flag, "uptrend", "downtrend"), index=P.close.index)

    strategies, trials = [], []

    def evaluate(sid, name, desc, model, horizon, res, kind, bench_nav, candidate=True, extra=None):
        nav = res.nav
        dev = metrics.perf(nav.loc[:dev_end], res.trades[res.trades.entry <= dev_end], res.turnover.loc[:dev_end])
        full_m = metrics.perf(nav.loc[:report_end], res.trades[res.trades.entry <= report_end],
                              res.turnover.loc[:report_end])
        hold_m = metrics.perf(nav.loc[hold:end], res.trades[res.trades.entry >= hold],
                              res.turnover.loc[hold:end]) if reveal else None
        b = bench_nav.loc[nav.index[0]:report_end].dropna()
        periods = {"2015-2017": ("2015-01-01", "2017-12-31"), "2018-2020": ("2018-01-01", "2020-12-31"),
                   "2021-2024": ("2021-01-01", str(dev_end.date()))}
        if reveal:
            periods["holdout"] = (str(hold.date()), str(end.date()))
        bd = metrics.breakdown(nav.loc[:report_end], periods, regime_dir)
        bd["vol_regimes"] = metrics.breakdown(nav.loc[:report_end], {}, regime_vol)["regimes"]
        entry = {
            "id": sid, "name": name, "description": desc, "model": model, "horizon_days": horizon,
            "kind": kind, "candidate": candidate, "recommended": False,
            "metrics": {k: r2(full_m.get(k), 2 if k in ("sharpe", "profit_factor", "sortino") else 1)
                        for k in ("net_return_pct", "cagr_pct", "sharpe", "sortino", "max_drawdown_pct",
                                  "win_rate_pct", "profit_factor", "trades", "avg_hold_days",
                                  "directional_accuracy_pct", "turnover_annual")},
            "dev_metrics": {k: r2(v) for k, v in dev.items()},
            "holdout_metrics": {k: r2(v) for k, v in hold_m.items()} if hold_m else None,
            "cagr_after_cgt_pct": r2(metrics.after_cgt_cagr(nav.loc[:report_end], cfg["costs"]["cgt_rate"])),
            "benchmark_return_pct": r2(100 * (b.iloc[-1] / b.iloc[0] - 1), 1) if len(b) else None,
            "equity": downsample(nav.loc[:report_end]),
            "benchmark": downsample(b) if len(b) else [],
            "breakdown": bd,
        }
        if entry["metrics"]["trades"] is not None:
            entry["metrics"]["trades"] = int(entry["metrics"]["trades"])
        if extra:
            entry.update(extra)
        strategies.append(entry)
        if candidate:
            trials.append({"id": sid, "name": name, "dev_daily": nav.loc[:dev_end].pct_change().dropna()})
        log(f"{sid:<10} {name:<48} dev CAGR {dev.get('cagr_pct', 0):6.1f}%  Sharpe {dev.get('sharpe', 0):5.2f}")
        return entry

    kse_nav = kse.reindex(P.close.index).ffill()

    # ---------------- baselines ----------------
    log("baselines")
    r20 = rebalance_dates(dates, oos, 20)
    ew = {d: pd.Series(1 / elig_A.loc[d].sum(), index=elig_A.columns[elig_A.loc[d]]) for d in r20 if elig_A.loc[d].any()}
    ew_res = run_engine(A, ew, cfg, oos, end)
    evaluate("b_ew", "Equal-weight universe (buy-and-hold proxy)",
             "All liquid stocks in the app's 105-name universe, equal weights, rebalanced every 20 trading days. "
             "Includes dividends (adjusted prices) and costs.", "none", 20, ew_res, "baseline", kse_nav, candidate=False)
    kres = backtest.Result(kse_nav.loc[oos:end], pd.DataFrame(), pd.DataFrame(columns=["entry", "pnl_net", "pnl_gross", "hold_days"]),
                           pd.Series(0.0, index=kse_nav.loc[oos:end].index), pd.Series(0.0, index=kse_nav.loc[oos:end].index))
    evaluate("b_kse100", "KSE-100 index (price, no dividends)",
             "The KSE-100 index itself, no costs. A price index: it excludes dividends, which the strategies earn.",
             "none", 0, kres, "baseline", kse_nav, candidate=False)

    # ---------------- S3: factor ranking ----------------
    log("S3 factor ranking")
    fac = ["mom_120_20", "dist_high_250", "lowvol_60"]
    score3 = signals.composite(feats, fac, elig_active)
    vol60 = feats["vol60"]
    s3_selections = {}
    for k in (5, 20):
        for n in (10, 20):
            for buf in (None, 2.0):
                sid = f"s3_k{k}_n{n}" + ("_buf" if buf else "")
                rd = rebalance_dates(dates, oos, k)
                tg = portfolio_targets(score3[full.active], elig_A, rd, n, buf)
                s3_selections[sid] = tg
                res = run_engine(A, tg, cfg, oos, end)
                evaluate(sid, f"Factor rank: top {n}, rebalance {k}d" + (", buffer" if buf else ""),
                         f"Hold the top {n} liquid stocks by a fixed blend of 6-month momentum (skipping the last month), "
                         f"closeness to the 52-week high and low volatility; rebalance every {k} trading days"
                         + ("; keep a holding while it stays in the top " + str(int(2 * n)) if buf else "") + ".",
                         "rule (no fitting)", k, res, "factor", kse_nav)
    for f1 in fac:
        sc = signals.composite(feats, [f1], elig_active)
        tg = portfolio_targets(sc[full.active], elig_A, r20, 20, 2.0)
        res = run_engine(A, tg, cfg, oos, end)
        evaluate(f"s3_single_{f1}", f"Single factor: {f1}, top 20, 20d, buffer",
                 f"Same as the factor blend but ranking on {f1} alone.", "rule (no fitting)", 20, res, "factor", kse_nav)
    PRIMARY = "s3_k20_n20_buf"   # chosen before any results: diversified, low turnover

    # PIT universe robustness for the primary factor strategy
    tg = portfolio_targets(score3.where(elig_pit), elig_pit, r20, 20, 2.0)
    res = run_engine(P, tg, cfg, oos, end)
    evaluate("s3_pit", "Factor rank (primary) on point-in-time universe",
             "The primary factor strategy (top 20, 20d, buffer) on a universe rebuilt each month from the 100 most traded "
             "stocks at that time, including names later delisted. Checks survivorship bias.",
             "rule (no fitting)", 20, res, "factor", kse_nav)

    # ---------------- ML: pooled walk-forward models ----------------
    stock_feats = ["r5", "r20", "r60", "mom_120_20", "mom_250_20", "vol20", "vol60", "vol_ratio",
                   "dist_high_250", "dist_low_250", "ma20_gap", "ma50_gap", "ma200_gap", "log_turnover",
                   "turnover_shock", "zero_vol_share", "beta60", "resid_mom_60"]
    every, emb = cfg["models"]["refit_every_days"], cfg["models"]["embargo_days"]
    mask = elig_A
    Xraw = models.to_long({k: feats[k][full.active] for k in stock_feats + signals.MARKET_FEATURES},
                          stock_feats + signals.MARKET_FEATURES, mask)
    ranked = {k: signals.cs_rank(feats[k][full.active].where(mask)) for k in stock_feats}
    Xrank = models.to_long(ranked, stock_feats, mask)
    sect = A.sector.astype("category").cat.codes
    Xraw["sector"] = sect.reindex(Xraw.index.get_level_values(1)).to_numpy()
    Xrank["sector"] = sect.reindex(Xrank.index.get_level_values(1)).to_numpy()

    def cached(name, fn):
        path = CACHE / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path).iloc[:, 0]
        s = fn()
        s.to_frame("p").to_parquet(path)
        return s

    fwd = {h: signals.forward_return(A, h) for h in (1, 5, 10, 20)}
    preds_reg, preds_cls, preds_rank = {}, {}, {}
    for h in (1, 5, 10, 20):
        y = fwd[h].stack(future_stack=True)
        log(f"walk-forward regressor h={h}")
        preds_reg[h] = cached(f"reg_h{h}", lambda: models.walk_forward(Xraw, y, dates, oos, every, h, emb, lgbm(cfg, seed)))
    for h in (5, 20):
        y = fwd[h].stack(future_stack=True)
        log(f"walk-forward classifier h={h}")
        preds_cls[h] = cached(f"cls_h{h}", lambda: models.walk_forward(Xraw, (y >= 0).astype(float).where(y.notna()), dates, oos,
                                                                          every, h, emb, lgbm(cfg, seed, cls=True), proba=True))
        ex = fwd[h].sub(fwd[h].where(mask).mean(axis=1), axis=0).stack(future_stack=True)
        log(f"walk-forward ranker h={h}")
        preds_rank[h] = cached(f"rank_h{h}", lambda: models.walk_forward(Xrank, ex, dates, oos, every, h, emb, lgbm(cfg, seed)))

    # ---------------- S4: ML ranking ----------------
    log("S4/S5/S6/S7")
    for h in (5, 20):
        sc = preds_rank[h].unstack().reindex(index=P.close.index, columns=full.active)
        for n in (10, 20):
            tg = portfolio_targets(sc, elig_A, rebalance_dates(dates, oos, h), n, 2.0)
            res = run_engine(A, tg, cfg, oos, end)
            evaluate(f"s4_h{h}_n{n}", f"ML rank (LightGBM): top {n}, {h}d horizon",
                     f"One LightGBM model across all stocks predicts each stock's {h}-day return relative to the others "
                     f"from ranked price/volume features; hold the top {n}, rebalance every {h} days, keep holdings in the top {2*n}.",
                     "LightGBM (pooled, walk-forward)", h, res, "ml_rank", kse_nav)
    S4_PRIMARY = ("s4_h20_n20", preds_rank[20].unstack().reindex(index=P.close.index, columns=full.active))

    # ---------------- S5: regime filter, S6: inverse-vol sizing ----------------
    for base_id, sc, label in ((PRIMARY, score3[full.active], "factor"), (S4_PRIMARY[0], S4_PRIMARY[1], "ML rank")):
        for e_off in (0.0, 0.5):
            expo = {d: (1.0 if exposure_flag.get(d, True) else e_off) for d in r20}
            tg = portfolio_targets(sc, elig_A, r20, 20, 2.0, exposure=expo)
            res = run_engine(A, tg, cfg, oos, end)
            evaluate(f"s5_{base_id}_x{int(e_off*100)}", f"{label} + market filter ({int(e_off*100)}% when KSE-100 < 200d avg)",
                     f"The {label} strategy (top 20, 20d), but exposure drops to {int(e_off*100)}% while the KSE-100 is below its "
                     "200-day average.", "rule + regime filter" if label == "factor" else "LightGBM + regime filter",
                     20, res, "overlay", kse_nav)
        tg = portfolio_targets(sc, elig_A, r20, 20, 2.0, weight="invvol", vol=vol60[full.active])
        res = run_engine(A, tg, cfg, oos, end)
        evaluate(f"s6_{base_id}_invvol", f"{label} + inverse-volatility weights",
                 f"The {label} strategy (top 20, 20d) with position sizes inversely proportional to 60-day volatility.",
                 "rule" if label == "factor" else "LightGBM", 20, res, "overlay", kse_nav)

    # ---------------- S7: confidence-gated direction model ----------------
    for h in (5, 20):
        pr = preds_cls[h].unstack().reindex(index=P.close.index, columns=full.active)
        for thr in (0.55, 0.60):
            tg = {}
            for d in rebalance_dates(dates, oos, h):
                if d not in pr.index:
                    continue
                row = pr.loc[d].where(elig_A.loc[d])
                pick = row[row > thr].sort_values(ascending=False).index[:10]
                tg[d] = pd.Series(0.1, index=pick)
            res = run_engine(A, tg, cfg, oos, end)
            evaluate(f"s7_h{h}_p{int(thr*100)}", f"Direction model, confident only (p>{thr:.2f}), {h}d",
                     f"A LightGBM classifier across all stocks estimates the probability each stock rises over {h} days; buy "
                     f"up to 10 stocks (10% each) whose probability exceeds {thr:.2f}, otherwise hold cash.",
                     "LightGBM classifier (pooled, walk-forward)", h, res, "ml_direction", kse_nav)

    # ---------------- S1/S2: current PSO/MEBL models ----------------
    import joblib
    from sklearn.base import clone
    from src.psx_predictor.models.feature_wrapper import prepare_data
    single_bench = {}
    for t in ("PSO", "MEBL"):
        b = joblib.load(ROOT / "models" / f"{t.lower()}_1day_best_model.pkl")
        X, _, d_, _ = prepare_data(t, exclude_known_bad_dates=True)
        X = X.reindex(columns=b["feature_cols"]).fillna(0.0)
        X.index = pd.DatetimeIndex(pd.to_datetime(pd.Series(d_)).values)
        X = X[~X.index.duplicated()]
        X.index = pd.MultiIndex.from_arrays([X.index, [t] * len(X)])
        y = fwd[1][t] if t in fwd[1] else signals.forward_return(subset(full, [t]), 1)[t]
        y = y.stack(future_stack=True) if isinstance(y, pd.DataFrame) else \
            pd.Series(y.values, index=pd.MultiIndex.from_arrays([y.index, [t] * len(y)]))
        Xd = X.loc[X.index.get_level_values(0).isin(P.close.index)]
        log(f"walk-forward current {t} model ({b['model_type']})")
        pred = cached(f"cur_{t}", lambda: models.walk_forward(Xd, y, dates, oos, every, 1, emb, lambda: clone(b["model"])))
        pred = pred.droplevel(1).reindex(P.close.index)
        T1 = subset(full, [t])
        bh = run_engine(T1, {pd.Timestamp(dates[int(np.searchsorted(dates, np.datetime64(oos))) - 1]): pd.Series(1.0, index=[t])},
                        cfg, oos, end)
        single_bench[t] = bh.nav
        evaluate(f"b_bh_{t}", f"Buy and hold {t}", f"Buy {t} once and hold it (dividends included).", "none", 0, bh,
                 "baseline", bh.nav, candidate=False)
        persist = {d: pd.Series(1.0 if feats["r1"].loc[d, t] > 0 else 0.0, index=[t]) for d in P.close.index[P.close.index >= oos - pd.Timedelta(days=5)]}
        res = run_engine(T1, persist, cfg, oos, end)
        evaluate(f"b_persist_{t}", f"Persistence {t} (long after an up day)",
                 f"Hold {t} the day after it rose, cash after it fell. Naive baseline for the daily models.", "none", 1, res,
                 "baseline", bh.nav, candidate=False)
        tg1 = {d: pd.Series(1.0 if p > 0 else 0.0, index=[t]) for d, p in pred.dropna().items()}
        res = run_engine(T1, tg1, cfg, oos, end)
        evaluate(f"s1_{t}", f"Current {t} model, trade every signal",
                 f"The app's {t} model ({b['model_type']}, same settings and features), refit quarterly on data available "
                 f"at the time, predicting the next tradable day. Hold {t} when it predicts a rise, cash otherwise.",
                 b["model_type"], 1, res, "current_model", bh.nav)
        q = pred.abs().dropna()
        thr = q.rolling(63, min_periods=40).quantile(0.67).shift(1)
        tg2 = {d: pd.Series(1.0 if (p > 0 and abs(p) > thr.get(d, np.inf)) else 0.0, index=[t]) for d, p in pred.dropna().items()}
        res = run_engine(T1, tg2, cfg, oos, end)
        evaluate(f"s2_{t}", f"Current {t} model, confident signals only",
                 f"As above, but only act when the predicted rise is in the top third of the model's recent predictions; "
                 "otherwise cash.", b["model_type"], 1, res, "current_model", bh.nav)

    # ---------------- random same-turnover baseline ----------------
    log("random baselines")
    rng = np.random.default_rng(seed)
    prim_t = s3_selections[PRIMARY]
    rand_navs = []
    for _ in range(cfg["random_baseline_seeds"]):
        held, tg = [], {}
        prev = []
        for d, w in prim_t.items():
            cur = list(w.index)
            changes = len(set(cur) - set(prev)) if prev else len(cur)
            pool = list(elig_A.columns[elig_A.loc[d]])
            held = [x for x in held if x in pool]
            drop = rng.choice(len(held), size=min(changes, len(held)), replace=False) if held and prev else []
            held = [x for i, x in enumerate(held) if i not in set(np.atleast_1d(drop))]
            cand = [x for x in pool if x not in held]
            add = list(rng.choice(cand, size=min(len(cur) - len(held), len(cand)), replace=False)) if cand else []
            held = held + add
            tg[d] = pd.Series(1 / len(held), index=held) if held else pd.Series(dtype=float)
            prev = cur
        rand_navs.append(run_engine(A, tg, cfg, oos, end))
    cagrs = [metrics.perf(r.nav.loc[:dev_end])["cagr_pct"] for r in rand_navs]
    med = rand_navs[int(np.argsort(cagrs)[len(cagrs) // 2])]
    evaluate("b_random", "Random picks, same turnover (median of 20)",
             "Random liquid stocks, replacing as many names at each rebalance as the primary factor strategy does. "
             f"Dev-period CAGR across 20 seeds: median {np.median(cagrs):.1f}%, 95th percentile {np.percentile(cagrs, 95):.1f}%.",
             "none", 20, med, "baseline", kse_nav, candidate=False,
             extra={"random_dev_cagr_p50": r2(np.median(cagrs)), "random_dev_cagr_p95": r2(np.percentile(cagrs, 95))})

    # ---------------- multiple testing + selection (dev period only) ----------------
    trial_sr = [metrics.sharpe_daily(t["dev_daily"]) for t in trials]
    ew_dev = metrics.perf(ew_res.nav.loc[:dev_end])
    rand_p95 = np.percentile(cagrs, 95)
    for s in strategies:
        if not s["candidate"]:
            continue
        tr = next(t for t in trials if t["id"] == s["id"])
        s["deflated_sharpe_dev"] = r2(metrics.deflated_sharpe(tr["dev_daily"], trial_sr), 3)
    cands = [s for s in strategies if s["candidate"]]
    passed = [s for s in cands if (s["deflated_sharpe_dev"] or 0) >= 0.95
              and s["dev_metrics"]["sharpe"] > ew_dev["sharpe"]
              and s["dev_metrics"]["cagr_pct"] > ew_dev["cagr_pct"]
              and s["dev_metrics"]["cagr_pct"] > rand_p95]
    passed.sort(key=lambda s: -s["deflated_sharpe_dev"])
    for s in passed[:2]:
        s["recommended"] = True

    # ---------------- model diagnostics ----------------
    hz = []
    for h in (1, 5, 10, 20):
        y = fwd[h].stack(future_stack=True)
        p = preds_reg[h]
        dfp = pd.DataFrame({"p": p, "y": y.reindex(p.index)}).dropna()
        dfp = dfp[dfp.index.get_level_values(0) <= report_end]
        up = dfp.y >= 0
        hit = ((dfp.p >= 0) == up)
        per_t = hit.groupby(level=1).mean() * 100
        hz.append({"horizon": f"{h}d", "directional_pct": r2(100 * hit.mean(), 1),
                   "always_up_pct": r2(100 * up.mean(), 1),
                   "mae_pct": r2(100 * (np.expm1(dfp.p) - np.expm1(dfp.y)).abs().mean(), 2),
                   "mae_zero_pct": r2(100 * np.expm1(dfp.y).abs().mean(), 2),
                   "rank_ic": r2(dfp.groupby(level=0).apply(lambda g: g.p.corr(g.y, method="spearman")).mean(), 3),
                   "per_ticker_p10_p50_p90": [r2(x, 1) for x in np.percentile(per_t, [10, 50, 90])],
                   "n": int(len(dfp))})
    cal = []
    for h in (5, 20):
        y = fwd[h].stack(future_stack=True)
        p = preds_cls[h]
        dfp = pd.DataFrame({"p": p, "y": (y.reindex(p.index) >= 0).astype(float).where(y.reindex(p.index).notna())}).dropna()
        dfp = dfp[dfp.index.get_level_values(0) <= report_end]
        for row in metrics.calibration_table(dfp.p, dfp.y):
            row["bucket"] = f"{h}d {row['bucket']}"
            cal.append(row)

    # ---------------- outputs ----------------
    pd.DataFrame([{"id": s["id"], "name": s["name"], "kind": s["kind"], "candidate": s["candidate"],
                   "recommended": s["recommended"], "deflated_sharpe_dev": s.get("deflated_sharpe_dev"),
                   **{f"dev_{k}": v for k, v in s["dev_metrics"].items()},
                   **({f"holdout_{k}": v for k, v in (s["holdout_metrics"] or {}).items()})}
                  for s in strategies]).to_csv(OUT_DIR / "trials.csv", index=False)
    result = {
        "generated_at": str(pd.Timestamp.today().date()),
        "universe": "KSE-100 (app's 105-stock universe; point-in-time check included)",
        "data_range": {"start": str(oos.date()), "end": str(report_end.date())},
        "holdout": {"start": str(hold.date()), "end": str(end.date()), "revealed": bool(reveal)},
        "headline": "",
        "assumptions": [],
        "strategies": strategies,
        "horizon_accuracy": hz,
        "calibration": cal,
        "trials_counted": len(trials),
        "findings": [], "risks": [], "next_steps": [],
        "data_cleaning": {"holidays_dropped": len(full.holidays), "artefact_prints": int(full.bad.values.sum())},
    }
    # Written conclusions (headline, assumptions, findings, risks, next steps) live in
    # reports/research/narrative.json so a re-run keeps them; they describe the
    # revealed results and are only merged when the holdout is shown.
    narrative = OUT_DIR / "narrative.json"
    if reveal and narrative.exists():
        result.update(json.loads(narrative.read_text(encoding="utf-8")))
    OUT_JSON.write_text(json.dumps(result, indent=1, default=str))
    log(f"wrote {OUT_JSON} ({len(strategies)} strategies, {len(trials)} counted trials); "
        f"recommended: {[s['id'] for s in strategies if s['recommended']] or 'none'}")


if __name__ == "__main__":
    main()
