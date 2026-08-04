from fastapi.testclient import TestClient
from src.psx_predictor.api.main import app, psx_cache

client = TestClient(app)

def test_get_market_performers(mocker):
    psx_cache.clear()
    
    class MockResponse:
        status_code = 200
        text = '''
        <html>
            <body>
                <div class="quote__close">Rs. 100.50</div>
                <div class="change__value">1.50</div>
                <div class="change__percent">(1.5%)</div>
            </body>
        </html>
        '''
    mocker.patch("requests.get", return_value=MockResponse())
    mocker.patch("os.path.exists", return_value=False)

    response = client.get("/api/market_performers")
    assert response.status_code == 200
    data = response.json()
    assert "top_active" in data
    assert "top_advancers" in data
    assert "top_decliners" in data
    assert len(data["top_active"]) > 0
    assert data["top_active"][0]["price"] == 100.5
