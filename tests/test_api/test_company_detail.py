import datetime
import math
from fastapi.testclient import TestClient
from src.psx_predictor.api.main import app
from src.psx_predictor.api.state import psx_cache

client = TestClient(app)


def _mock_conn(mocker, fetchall_return):
    """Builds a MagicMock standing in for `with engine.connect() as conn: ...`
    where conn.execute(...).mappings().fetchall() returns fetchall_return."""
    mock_conn = mocker.MagicMock()
    mock_conn.execute.return_value.mappings.return_value.fetchall.return_value = fetchall_return
    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False
    mocker.patch("src.psx_predictor.api.routers.company.engine", mock_engine)
    return mock_conn


def test_fundamentals_invalid_ticker():
    response = client.get("/api/company/NOTREAL/fundamentals")
    assert response.status_code == 400


def test_fundamentals_valid(mocker):
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    _mock_conn(mocker, [
        {"ticker": "PSO", "report_date": datetime.date(2025, 12, 31), "pe_ratio": 8.5, "revenue": 1000000.0},
    ])

    response = client.get("/api/company/PSO/fundamentals")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "PSO"
    assert len(data["fundamentals"]) == 1
    assert data["fundamentals"][0]["report_date"] == "2025-12-31"
    assert data["fundamentals"][0]["pe_ratio"] == 8.5


def test_fundamentals_nan_becomes_null(mocker):
    """Some stock_fundamentals rows store NaN instead of NULL for missing ratios;
    JSON has no NaN literal, so this used to 500 the whole endpoint."""
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    _mock_conn(mocker, [
        {"ticker": "PSO", "report_date": datetime.date(2025, 12, 31), "pe_ratio": math.nan, "revenue": 1000000.0},
    ])

    response = client.get("/api/company/PSO/fundamentals")
    assert response.status_code == 200
    data = response.json()
    assert data["fundamentals"][0]["pe_ratio"] is None


def test_history_valid(mocker):
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    _mock_conn(mocker, [
        {"date": datetime.date(2026, 9, 1), "open": 200.0, "high": 205.0, "low": 199.0, "close": 202.0, "volume": 500000},
    ])

    response = client.get("/api/company/PSO/history", params={"range": "30D"})
    assert response.status_code == 200
    data = response.json()
    assert data["range"] == "30D"
    assert len(data["history"]) == 1
    assert data["history"][0]["close"] == 202.0


def test_history_invalid_range(mocker):
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    response = client.get("/api/company/PSO/history", params={"range": "13X"})
    assert response.status_code == 400


def test_events_valid(mocker):
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    _mock_conn(mocker, [
        {"event_date": datetime.date(2026, 8, 1), "event_type": "dividend", "title": "Cash Dividend: Rs. 5", "sentiment_score": None},
    ])

    response = client.get("/api/company/PSO/events")
    assert response.status_code == 200
    data = response.json()
    assert len(data["events"]) == 1
    assert data["events"][0]["type"] == "dividend"


def test_company_profile_falls_back_to_db_on_scrape_failure(mocker):
    """The live scrape can fail (PSX down, network error) — the profile endpoint
    must degrade to DB-only data instead of 500ing or 404ing the whole page."""
    psx_cache.clear()
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    mocker.patch("src.psx_predictor.api.routers.company.get_with_retry", side_effect=Exception("connection refused"))

    mock_conn = mocker.MagicMock()
    mock_row = mocker.MagicMock(company_name="Pakistan State Oil", sector="Oil & Gas Marketing")
    mock_conn.execute.return_value.fetchone.return_value = mock_row
    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False
    mocker.patch("src.psx_predictor.api.routers.company.engine", mock_engine)

    response = client.get("/api/company/PSO")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Pakistan State Oil"
    assert data["sector"] == "Oil & Gas Marketing"
    assert data["people"] == []


def test_snapshot_invalid_ticker():
    response = client.get("/api/company/NOTREAL/snapshot")
    assert response.status_code == 400


def test_snapshot_valid(mocker):
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    mocker.patch(
        "src.psx_predictor.api.routers.company.get_price_snapshot",
        return_value={"PSO": {"price": 364.92, "week_52_high": 400.0, "week_52_low": 300.0}},
    )
    mocker.patch(
        "src.psx_predictor.api.routers.company.get_latest_ratios",
        return_value={"market_cap": 1.6e11, "pe_ratio": 6.9, "shares_outstanding": 455209732.0, "free_float_pct": 0.5},
    )

    response = client.get("/api/company/PSO/snapshot")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "PSO"
    assert data["week_52_high"] == 400.0
    assert data["shares_outstanding"] == 455209732.0
    assert data["free_float_pct"] == 0.5


def test_realtime_falls_back_to_db_on_scrape_failure(mocker):
    psx_cache.clear()
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    mocker.patch("src.psx_predictor.api.routers.company.get_with_retry", side_effect=Exception("connection refused"))

    mock_conn = mocker.MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        mocker.MagicMock(close=202.0),
        mocker.MagicMock(close=200.0),
    ]
    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False
    mocker.patch("src.psx_predictor.api.routers.company.engine", mock_engine)

    response = client.get("/api/realtime/PSO")
    assert response.status_code == 200
    data = response.json()
    assert data["price"] == 202.0
    assert data["change"] == 2.0
