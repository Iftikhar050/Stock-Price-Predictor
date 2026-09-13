from fastapi.testclient import TestClient
from src.psx_predictor.api.main import app
from src.psx_predictor.api.state import psx_cache

client = TestClient(app)


class FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data


def test_reports_invalid_ticker():
    response = client.get("/api/company/NOTREAL/reports")
    assert response.status_code == 400


def test_reports_valid(mocker):
    psx_cache.clear()
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"ABL"})

    def fake_post(url, data=None, timeout=None):
        return FakeResponse(200, [
            {
                "Reports": '<a target="_blank" href="lib/DownloadPDF.php?id=276117">Quarterly</a>',
                "period_ended": "2026-03-31",
                "posting_date": "2026-04-30",
            }
        ])

    mocker.patch("src.psx_predictor.api.routers.company.post_with_retry", side_effect=fake_post)

    response = client.get("/api/company/ABL/reports", params={"years": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "ABL"
    assert len(data["reports"]) == 1
    r = data["reports"][0]
    assert r["type"] == "Quarterly"
    assert r["download_url"] == "https://financials.psx.com.pk/lib/DownloadPDF.php?id=276117"
    assert r["period_ended"] == "2026-03-31"


def test_reports_handles_non_200_gracefully(mocker):
    psx_cache.clear()
    mocker.patch("src.psx_predictor.api.routers.company.VALID_TICKERS", {"ABL"})
    mocker.patch(
        "src.psx_predictor.api.routers.company.post_with_retry",
        side_effect=lambda *a, **k: FakeResponse(500, None),
    )

    response = client.get("/api/company/ABL/reports", params={"years": 1})
    assert response.status_code == 200
    assert response.json()["reports"] == []
