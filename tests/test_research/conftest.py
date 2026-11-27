import numpy as np
import pandas as pd
import pytest

ZERO_COSTS = {"commission": 0.0, "sales_tax_on_commission": 0.0, "exchange_clearing_cdc_secp": 0.0,
              "cvt_on_buys": 0.0, "slippage_by_adv": [[0, 0.0]], "cgt_rate": 0.15}
EXEC = {"account_pkr": 100.0, "max_adv_fraction": 1.0, "circuit_band_pre_2020": 0.05,
        "circuit_band_from_2020": 0.075, "circuit_change_date": "2020-03-01"}
CLEAN = {"impossible_move": 0.11, "penny_cap": 0.30, "holiday_zero_volume_share": 0.5}


def bt_inputs(prices, can_buy=None, can_sell=None, adv=None):
    p = np.asarray(prices, dtype=float).reshape(len(prices), -1)
    T, N = p.shape
    dates = pd.bdate_range("2021-01-04", periods=T).values
    ret = np.vstack([np.zeros((1, N)), p[1:] / p[:-1] - 1])
    cb = np.ones((T, N), bool) if can_buy is None else np.asarray(can_buy).reshape(T, N)
    cs = np.ones((T, N), bool) if can_sell is None else np.asarray(can_sell).reshape(T, N)
    a = np.full((T, N), 1e12) if adv is None else np.asarray(adv, dtype=float).reshape(T, N)
    return dates, [f"T{i}" for i in range(N)], ret, cb, cs, a


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2014-01-01", periods=700)
    rows = []
    for t in ["AAA", "BBB", "CCC"]:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(dates))))
        for d, c in zip(dates, px):
            rows.append((t, d, c, 1000.0))
    df = pd.DataFrame(rows, columns=["ticker", "date", "close", "volume"])
    meta = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "sector": ["X", "X", "Y"], "is_active": [True] * 3})
    return df, meta, {"data_cleaning": CLEAN, "execution": EXEC}
