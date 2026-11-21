import pandas as pd

from src.psx_predictor.data import build_features as bf
from src.psx_predictor.db.models import StockFundamentals

# Every stock_fundamentals column, so merge_fundamentals sees the shape it
# gets from the real table.
TABLE_COLS = [c for c in StockFundamentals.__table__.columns.keys() if c not in ("report_date", "created_at")]


def _prices(start, end):
    dates = pd.bdate_range(start, end)
    return pd.DataFrame({"date": dates, "close": 100.0})


def _fundamentals(rows):
    df = pd.DataFrame(rows).reindex(columns=["report_date"] + TABLE_COLS)
    df["date"] = pd.to_datetime(df["report_date"])
    df["ticker"] = "TEST"
    df["shares_outstanding"] = 1.0e6
    return df.drop(columns=["report_date"]).astype({c: float for c in TABLE_COLS if c != "ticker"})


def _merge(mocker, prices, fund):
    # 1st query: stock_fundamentals; 2nd: stock_financial_periods (none here,
    # so the stock_fundamentals values are what's merged).
    mocker.patch.object(bf.pd, "read_sql", side_effect=[fund, pd.DataFrame()])
    mocker.patch.object(bf, "engine")
    return bf.merge_fundamentals(prices, "TEST")


def test_weekend_report_still_reaches_features(mocker):
    # 31 Dec 2023 was a Sunday - the old exact-date join dropped it entirely.
    fund = _fundamentals([
        {"report_date": "2023-09-30", "revenue": 100.0, "net_income": 10.0, "eps": 1.0},
        {"report_date": "2023-12-31", "revenue": 200.0, "net_income": 20.0, "eps": 2.0},
    ])
    out = _merge(mocker, _prices("2023-09-01", "2024-04-30"), fund)
    by_date = out.set_index("date")["revenue"]
    lag = pd.Timedelta(days=bf.REPORT_PUBLICATION_LAG_DAYS)
    assert by_date[by_date.index >= pd.Timestamp("2023-12-31") + lag].eq(200.0).all()


def test_report_not_used_before_publication(mocker):
    fund = _fundamentals([{"report_date": "2023-12-31", "revenue": 200.0, "net_income": 20.0, "eps": 2.0}])
    out = _merge(mocker, _prices("2023-12-01", "2024-04-30"), fund)
    by_date = out.set_index("date")["revenue"]
    available = pd.Timestamp("2023-12-31") + pd.Timedelta(days=bf.REPORT_PUBLICATION_LAG_DAYS)
    # Before the figures are public there is no data (0.0), not a back-filled future value.
    assert by_date[by_date.index < available].eq(0.0).all()
    assert by_date[by_date.index >= available].eq(200.0).all()
