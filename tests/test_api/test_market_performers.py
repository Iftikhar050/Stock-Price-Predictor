from fastapi.testclient import TestClient
from src.psx_predictor.api.main import app, psx_cache

client = TestClient(app)


def test_get_market_performers(mocker):
    psx_cache.clear()

    mocker.patch(
        "src.psx_predictor.api.routers.market.get_price_snapshot",
        return_value={
            "PSO": {"price": 100.5, "change": 1.5, "change_percent": 1.5, "volume": 500000},
            "MEBL": {"price": 200.0, "change": -2.0, "change_percent": -1.0, "volume": 100000},
        },
    )

    response = client.get("/api/market_performers")
    assert response.status_code == 200
    data = response.json()
    assert "top_active" in data
    assert "top_advancers" in data
    assert "top_decliners" in data
    assert len(data["top_active"]) == 2
    # top_active sorted by volume desc -> PSO (500000) before MEBL (100000)
    assert data["top_active"][0]["symbol"] == "PSO"
    assert data["top_active"][0]["price"] == 100.5
    # top_advancers sorted by change_percent desc -> PSO (+1.5%) before MEBL (-1.0%)
    assert data["top_advancers"][0]["symbol"] == "PSO"
    # top_decliners sorted by change_percent asc -> MEBL (-1.0%) first
    assert data["top_decliners"][0]["symbol"] == "MEBL"


def test_market_performers_no_live_scrape(mocker):
    """The old implementation live-scraped dps.psx.com.pk for every active ticker;
    the DB-backed rewrite must never call requests.get at all."""
    psx_cache.clear()
    get_mock = mocker.patch("requests.get")
    mocker.patch(
        "src.psx_predictor.api.routers.market.get_price_snapshot",
        return_value={"PSO": {"price": 100.5, "change": 1.5, "change_percent": 1.5, "volume": 500000}},
    )

    response = client.get("/api/market_performers")
    assert response.status_code == 200
    get_mock.assert_not_called()
