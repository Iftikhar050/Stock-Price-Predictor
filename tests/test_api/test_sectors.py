from fastapi.testclient import TestClient
from src.psx_predictor.api.main import app
from src.psx_predictor.api.state import psx_cache

client = TestClient(app)


def test_sectors_overview(mocker):
    psx_cache.clear()
    mocker.patch(
        "src.psx_predictor.api.routers.market._compute_sectors_overview",
        return_value={"sectors": [
            {"sector": "Commercial Banks", "ticker_count": 13, "tickers": ["MEBL", "UBL"],
             "avg_pe": 10.0, "avg_pb": 2.0, "index_level": 22000.0},
            {"sector": "Oil & Gas Marketing", "ticker_count": 5, "tickers": ["PSO"],
             "avg_pe": 8.0, "avg_pb": 1.2, "index_level": 15000.0},
        ]},
    )

    response = client.get("/api/sectors")
    assert response.status_code == 200
    data = response.json()
    assert len(data["sectors"]) == 2
    names = {s["sector"] for s in data["sectors"]}
    assert "Commercial Banks" in names
    assert "Oil & Gas Marketing" in names


def test_sector_detail_found(mocker):
    mocker.patch(
        "src.psx_predictor.api.routers.market._compute_sector_detail",
        return_value={"sector": "Commercial Banks", "tickers": [
            {"ticker": "MEBL", "name": "Meezan Bank", "sector": "Commercial Banks",
             "price": 300.0, "change": -3.0, "change_percent": -1.0, "volume": 100000,
             "market_cap": 8e10, "pe_ratio": 12.0, "pb_ratio": 3.0, "dividend_yield": 0.03, "roe": 0.25},
        ]},
    )

    response = client.get("/api/sectors/Commercial Banks")
    assert response.status_code == 200
    data = response.json()
    assert data["sector"] == "Commercial Banks"
    assert len(data["tickers"]) == 1


def test_sector_detail_not_found(mocker):
    mocker.patch("src.psx_predictor.api.routers.market._compute_sector_detail", return_value=None)

    response = client.get("/api/sectors/Not A Real Sector")
    assert response.status_code == 404
