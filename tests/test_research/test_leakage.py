"""Leakage checks: features use only the past, labels start at the fill, and the
walk-forward never trains on labels that overlap the test block."""
import numpy as np
import pandas as pd

from src.psx_predictor.research import data, models, signals


def _panel(df, meta, cfg):
    return data.build_panel(df, meta, cfg)


def test_features_unchanged_when_future_rows_removed(synthetic_df):
    df, meta, cfg = synthetic_df
    p_full = _panel(df, meta, cfg)
    cut = p_full.close.index[500]
    p_cut = _panel(df[df.date <= cut], meta, cfg)
    mkt = p_full.close.mean(axis=1)
    f_full = signals.features(p_full, mkt)
    f_cut = signals.features(p_cut, mkt[mkt.index <= cut])
    for k in ("r20", "mom_120_20", "dist_high_250", "vol60", "beta60", "ma200_gap", "mkt_ma200_gap"):
        a, b = f_full[k].loc[:cut], f_cut[k]
        pd.testing.assert_frame_equal(a, b, check_freq=False, obj=k)


def test_forward_return_starts_at_the_fill(synthetic_df):
    df, meta, cfg = synthetic_df
    p = _panel(df, meta, cfg)
    h, t = 5, 100
    fwd = signals.forward_return(p, h)
    c = p.close["AAA"]
    assert np.isclose(fwd["AAA"].iloc[t], np.log(c.iloc[t + 1 + h] / c.iloc[t + 1]))


class Recorder:
    seen = []

    def fit(self, X, y):
        Recorder.seen.append(X.index.get_level_values(0).max())
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_walk_forward_purges_overlapping_labels():
    dates = pd.bdate_range("2015-01-01", periods=600)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]])
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    y = pd.Series(1.0, index=idx)
    h, emb, every = 20, 5, 63
    Recorder.seen = []
    models.walk_forward(X, y, dates.values, dates[300], every, h, emb, Recorder)
    pos = {d: i for i, d in enumerate(dates)}
    starts = models.blocks(dates.values, dates[300], every)
    for s, last_train in zip(starts, Recorder.seen):
        assert pos[last_train] + 1 + h < s - emb


def test_noise_labels_give_no_out_of_sample_skill():
    import lightgbm as lgb
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2015-01-01", periods=700)
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(20)]])
    X = pd.DataFrame(rng.normal(size=(len(idx), 5)), index=idx, columns=list("abcde"))
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    p = models.walk_forward(X, y, dates.values, dates[300], 63, 5, 5,
                            lambda: lgb.LGBMRegressor(n_estimators=50, verbose=-1, random_state=0))
    ic = pd.DataFrame({"p": p, "y": y.reindex(p.index)}).groupby(level=0).apply(
        lambda g: g.p.corr(g.y, method="spearman"))
    assert abs(ic.mean()) < 0.03
