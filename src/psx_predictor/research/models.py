"""Walk-forward (expanding window, purged + embargoed) model predictions.

A row dated t has features known at the close of t and a label covering the
close of t+1 (the fill) to the close of t+1+h. When predicting a test block that
starts at trading-day index s, training uses only rows with
t + 1 + h < s - embargo, so no training label overlaps the test period.
Hyperparameters are fixed in config (no tuning), so there is nothing to select
on test data."""
import numpy as np
import pandas as pd


def to_long(feats, names, mask):
    cols = {}
    for n in names:
        cols[n] = feats[n].where(mask).stack(future_stack=True)
    X = pd.DataFrame(cols)
    return X.dropna(how="all")


def blocks(dates, oos_start, every):
    s0 = int(np.searchsorted(dates, np.datetime64(oos_start)))
    return list(range(s0, len(dates), every))


def walk_forward(X, y, dates, oos_start, every, h, embargo, make_model, proba=False):
    """X, y indexed by (date, ticker). Returns OOS predictions (same index)."""
    pos = pd.Series(np.arange(len(dates)), index=pd.DatetimeIndex(dates))
    row_t = pos.reindex(X.index.get_level_values(0)).to_numpy()
    yv = y.reindex(X.index)
    out = []
    starts = blocks(dates, oos_start, every)
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(dates)
        tr = (row_t + 1 + h < s - embargo) & yv.notna().to_numpy()
        te = (row_t >= s) & (row_t < e)
        if tr.sum() < 500 or te.sum() == 0:
            continue
        m = make_model()
        m.fit(X.loc[tr], yv.loc[tr])
        p = m.predict_proba(X.loc[te])[:, 1] if proba else m.predict(X.loc[te])
        out.append(pd.Series(p, index=X.index[te]))
    return pd.concat(out) if out else pd.Series(dtype=float)
