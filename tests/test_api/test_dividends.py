import datetime
from fastapi.testclient import TestClient
from src.psx_predictor.api.main import app

client = TestClient(app)


def _mock_conn(mocker, fetchall_return):
    mock_conn = mocker.MagicMock()
    mock_conn.execute.return_value.mappings.return_value.fetchall.return_value = fetchall_return
    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False
    mocker.patch("src.psx_predictor.api.routers.company.engine", mock_engine)
    return mock_conn


def test_dividends_invalid_ticker():
    response = client.get("/api/company/NOTREAL/dividends")
    assert response.status_code == 400


def test_dividends_valid(mocker):
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    _mock_conn(mocker, [
        {
            "ex_dividend_date": datetime.date(2026, 8, 15),
            "announcement_date": datetime.date(2026, 8, 1),
            "dividend_amount": 10.0,
            "dividend_type": "Cash",
        },
    ])

    response = client.get("/api/company/PSO/dividends")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "PSO"
    assert len(data["dividends"]) == 1
    assert data["dividends"][0]["ex_date"] == "2026-08-15"
    assert data["dividends"][0]["amount"] == 10.0
    assert data["dividends"][0]["type"] == "Cash"


def test_dividends_empty(mocker):
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"PSO"})
    _mock_conn(mocker, [])

    response = client.get("/api/company/PSO/dividends")
    assert response.status_code == 200
    assert response.json()["dividends"] == []
