from fastapi.testclient import TestClient
from src.psx_predictor.api.main import app
from src.psx_predictor.api.state import psx_cache

client = TestClient(app)

FAKE_ROWS = [
    {"ticker": "PSO", "name": "Pakistan State Oil", "sector": "Oil & Gas Marketing",
     "price": 200.0, "change": 2.0, "change_percent": 1.0, "volume": 500000,
     "week_52_high": 210.0, "week_52_low": 150.0, "pct_from_52w_high": -4.76, "pct_from_52w_low": 33.33,
     "market_cap": 5e10, "pe_ratio": 8.0, "pb_ratio": 1.2, "dividend_yield": 0.05, "roe": 0.15},
    {"ticker": "MEBL", "name": "Meezan Bank", "sector": "Commercial Banks",
     "price": 300.0, "change": -3.0, "change_percent": -1.0, "volume": 100000,
     "week_52_high": 305.0, "week_52_low": 280.0, "pct_from_52w_high": -1.64, "pct_from_52w_low": 7.14,
     "market_cap": 8e10, "pe_ratio": 12.0, "pb_ratio": 3.0, "dividend_yield": 0.03, "roe": 0.25},
]


def test_screener_default(mocker):
    psx_cache.clear()
    mocker.patch("src.psx_predictor.api.routers.screener._compute_screener_rows", return_value=FAKE_ROWS)

    response = client.get("/api/screener")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["results"]) == 2
    # default sort_by=market_cap desc -> MEBL (8e10) before PSO (5e10)
    assert data["results"][0]["ticker"] == "MEBL"


def test_screener_sector_filter(mocker):
    psx_cache.clear()
    mocker.patch("src.psx_predictor.api.routers.screener._compute_screener_rows", return_value=FAKE_ROWS)

    response = client.get("/api/screener", params={"sector": "Commercial Banks"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["results"][0]["ticker"] == "MEBL"


def test_screener_52w_high_preset(mocker):
    """The '52 Week High Stocks' preset is sort_by=pct_from_52w_high&order=desc -
    closest to 0 (i.e. closest to its own high) should sort first."""
    psx_cache.clear()
    mocker.patch("src.psx_predictor.api.routers.screener._compute_screener_rows", return_value=FAKE_ROWS)

    response = client.get("/api/screener", params={"sort_by": "pct_from_52w_high", "order": "desc"})
    assert response.status_code == 200
    data = response.json()
    # MEBL is -1.64% from its high, PSO is -4.76% - MEBL is closer, so first under order=desc
    assert data["results"][0]["ticker"] == "MEBL"


def test_screener_52w_low_preset(mocker):
    """The '52 Week Low Stocks' preset is sort_by=pct_from_52w_low&order=asc -
    closest to 0 (i.e. closest to its own low) should sort first."""
    psx_cache.clear()
    mocker.patch("src.psx_predictor.api.routers.screener._compute_screener_rows", return_value=FAKE_ROWS)

    response = client.get("/api/screener", params={"sort_by": "pct_from_52w_low", "order": "asc"})
    assert response.status_code == 200
    data = response.json()
    # MEBL is +7.14% above its low, PSO is +33.33% - MEBL is closer, so first under order=asc
    assert data["results"][0]["ticker"] == "MEBL"


def test_screener_invalid_sort_by():
    response = client.get("/api/screener", params={"sort_by": "not_a_field"})
    assert response.status_code == 400


def test_screener_invalid_order():
    response = client.get("/api/screener", params={"order": "sideways"})
    assert response.status_code == 400


def test_compare_tickers(mocker):
    psx_cache.clear()
    mocker.patch("src.psx_predictor.api.routers.screener._compute_screener_rows", return_value=FAKE_ROWS)
    mocker.patch("src.psx_predictor.api.routers.screener.VALID_TICKERS", {"PSO", "MEBL"})

    response = client.get("/api/compare", params={"tickers": "PSO,MEBL"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2


def test_compare_requires_at_least_two():
    response = client.get("/api/compare", params={"tickers": "PSO"})
    assert response.status_code == 400
