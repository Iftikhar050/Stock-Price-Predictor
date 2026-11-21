import numpy as np
import pandas as pd

from src.psx_predictor.data import build_features as bf


def _periods(rows):
    df = pd.DataFrame(rows)
    for col in ("revenue", "net_income", "eps", "total_assets", "total_debt", "shares_outstanding"):
        if col not in df:
            df[col] = np.nan
    df["ticker"] = "TEST"
    df["source"] = df.get("source", "yfinance")
    return df


def _clean(mocker, rows):
    mocker.patch.object(bf.pd, "read_sql", return_value=_periods(rows))
    mocker.patch.object(bf, "engine")
    return bf.clean_period_fundamentals("TEST").set_index("date")


QUARTERS = [
    {"period_end": d, "period_type": "Q", "revenue": 100.0, "net_income": 10.0, "eps": 1.0, "shares_outstanding": 10.0}
    for d in ("2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31")
]


def test_four_quarters_sum_to_twelve_months(mocker):
    out = _clean(mocker, QUARTERS)
    last = out.iloc[-1]
    assert (last.revenue, last.net_income, last.eps_trailing) == (400.0, 40.0, 4.0)
    # The 4th quarter is public 60 days after it ends.
    assert out.index[-1] == pd.Timestamp("2025-12-31") + pd.Timedelta(days=bf.REPORT_PUBLICATION_LAG_DAYS)


def test_year_and_quarters_ending_together_prefer_the_year(mocker):
    year = {"period_end": "2025-12-31", "period_type": "A", "revenue": 410.0, "net_income": 41.0, "eps": 4.1,
            "shares_outstanding": 10.0}
    out = _clean(mocker, QUARTERS + [year])
    assert out.iloc[-1].net_income == 41.0


def test_missing_eps_is_filled_from_profit_and_shares(mocker):
    rows = [dict(q) for q in QUARTERS]
    rows[-1]["eps"] = np.nan
    out = _clean(mocker, rows)
    assert out.iloc[-1].eps_trailing == 4.0


def test_items_come_from_the_same_period(mocker):
    # Net income is reported for the newer quarters; revenue only for an older year.
    rows = [dict(q, revenue=np.nan) for q in QUARTERS] + [
        {"period_end": "2024-06-30", "period_type": "A", "revenue": 999.0, "net_income": 30.0, "eps": 3.0,
         "shares_outstanding": 10.0}]
    out = _clean(mocker, rows)
    last = out.iloc[-1]
    assert last.net_income == 40.0 and last.eps_trailing == 4.0
    # revenue isn't in the quarterly period, so it falls back to its own newest figure
    assert last.revenue == 999.0


def test_year_on_year_growth(mocker):
    prior = [{"period_end": "2024-12-31", "period_type": "A", "revenue": 200.0, "net_income": 20.0, "eps": 2.0,
              "shares_outstanding": 10.0}]
    out = _clean(mocker, prior + QUARTERS)
    last = out.iloc[-1]
    assert round(last.revenue_growth, 6) == 1.0 and round(last.profit_growth, 6) == 1.0


def test_yahoo_preferred_over_psx_when_both_exist(mocker):
    psx = [{"period_end": "2025-12-31", "period_type": "A", "source": "psx_dps", "revenue": 5.0,
            "net_income": 5.0, "eps": 5.0, "shares_outstanding": 10.0}]
    rows = [dict(q, source="yfinance") for q in QUARTERS] + psx
    out = _clean(mocker, rows)
    assert out.iloc[-1].net_income == 40.0
