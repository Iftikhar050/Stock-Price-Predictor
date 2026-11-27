"""Price/volume features computed only from data up to each day's close, and the
rule-based portfolio selection used by the factor strategies."""
import numpy as np
import pandas as pd


def features(panel, market):
    """Wide feature matrices {name: DataFrame}. `market` = KSE-100 close series."""
    p = panel.close
    lr = np.log1p(panel.ret)
    roll = lambda w: lr.rolling(w, min_periods=int(w * 0.8)).sum()
    f = {}
    f["r1"] = lr
    f["r5"] = roll(5)
    f["r20"] = roll(20)
    f["r60"] = roll(60)
    f["mom_120_20"] = roll(120) - roll(20)
    f["mom_250_20"] = roll(250) - roll(20)
    f["vol20"] = lr.rolling(20, min_periods=15).std()
    f["vol60"] = lr.rolling(60, min_periods=40).std()
    f["vol_ratio"] = f["vol20"] / f["vol60"]
    f["lowvol_60"] = -f["vol60"]
    f["dist_high_250"] = p / p.rolling(250, min_periods=200).max() - 1
    f["dist_low_250"] = p / p.rolling(250, min_periods=200).min() - 1
    for w in (20, 50, 200):
        f[f"ma{w}_gap"] = p / p.rolling(w, min_periods=int(w * 0.8)).mean() - 1
    tv = (p.ffill() * panel.volume.fillna(0))
    f["log_turnover"] = np.log1p(tv.rolling(20, min_periods=10).median())
    f["turnover_shock"] = np.log1p(tv.rolling(5, min_periods=3).mean()) - np.log1p(tv.rolling(60, min_periods=40).mean())
    f["zero_vol_share"] = (panel.volume.fillna(0) == 0).rolling(60, min_periods=40).mean()
    m = market.reindex(p.index).ffill()
    mr = np.log(m).diff()
    beta = lr.rolling(60, min_periods=40).cov(mr) / mr.rolling(60, min_periods=40).var().values[:, None]
    f["beta60"] = beta
    f["resid_mom_60"] = f["r60"] - beta.mul(mr.rolling(60, min_periods=40).sum(), axis=0)
    # market-level (same for every ticker on a day)
    mk = {"mkt_r5": mr.rolling(5).sum(), "mkt_r20": mr.rolling(20).sum(),
          "mkt_vol20": mr.rolling(20).std(), "mkt_ma200_gap": m / m.rolling(200, min_periods=160).mean() - 1}
    for k, s in mk.items():
        f[k] = pd.DataFrame(np.repeat(s.to_numpy()[:, None], p.shape[1], axis=1), index=p.index, columns=p.columns)
    return f


MARKET_FEATURES = ["mkt_r5", "mkt_r20", "mkt_vol20", "mkt_ma200_gap"]


def forward_return(panel, h):
    """Log return from the close of t+1 (the fill) to the close of t+1+h.
    (Artefact days carry a zero return, see data.build_panel.)"""
    lr = np.log1p(panel.ret)
    # S[t] = lr[t] + ... + lr[t+h-1] = log P[t+h-1] - log P[t-1]; we need S[t+2]
    return lr[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-2)


def cs_rank(df):
    return df.rank(axis=1, pct=True)


def composite(feats, names, eligible):
    ranks = [cs_rank(feats[n].where(eligible)) for n in names]
    return sum(ranks) / len(ranks)


def select(score_row, n, held=None, buffer=None):
    """Top-n names by score; with a buffer, current holdings are kept while they
    stay within the top buffer*n."""
    s = score_row.dropna().sort_values(ascending=False)
    if s.empty:
        return []
    if held is None or buffer is None:
        return list(s.index[:n])
    keep_zone = set(s.index[: int(round(buffer * n))])
    keep = [t for t in held if t in keep_zone]
    fill = [t for t in s.index if t not in keep][: max(0, n - len(keep))]
    return keep + fill
