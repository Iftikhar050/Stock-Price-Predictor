import datetime
from fastapi.testclient import TestClient
from src.psx_predictor.api.main import app
from src.psx_predictor.api.state import psx_cache

client = TestClient(app)


def _mock_conn(mocker, fetchall_return):
    mock_conn = mocker.MagicMock()
    mock_conn.execute.return_value.mappings.return_value.fetchall.return_value = fetchall_return
    mock_engine = mocker.MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    mock_engine.connect.return_value.__exit__.return_value = False
    mocker.patch("src.psx_predictor.api.routers.market.engine", mock_engine)
    return mock_conn


def test_indices_excludes_kse100(mocker):
    """Our KSE100 is a documented synthetic proxy - must never appear as real market data."""
    psx_cache.clear()
    _mock_conn(mocker, [
        {"date": datetime.date(2026, 9, 1), "kmi30_index_level": 250000.0, "kse30_index_level": 52000.0, "all_share_index_level": 106000.0},
        {"date": datetime.date(2026, 9, 4), "kmi30_index_level": 250239.14, "kse30_index_level": 52294.83, "all_share_index_level": 106442.83},
    ])

    response = client.get("/api/indices")
    assert response.status_code == 200
    data = response.json()
    keys = [i["key"] for i in data["indices"]]
    assert "kse100" not in keys
    assert set(keys) == {"kmi30", "kse30", "all_share"}


def test_indices_change_percent_and_history(mocker):
    psx_cache.clear()
    _mock_conn(mocker, [
        {"date": datetime.date(2026, 9, 1), "kmi30_index_level": 250000.0, "kse30_index_level": 52000.0, "all_share_index_level": 106000.0},
        {"date": datetime.date(2026, 9, 4), "kmi30_index_level": 250250.0, "kse30_index_level": 52260.0, "all_share_index_level": 106318.0},
    ])

    response = client.get("/api/indices")
    data = response.json()
    kmi30 = next(i for i in data["indices"] if i["key"] == "kmi30")
    assert kmi30["level"] == 250250.0
    assert kmi30["change_percent"] == 0.1
    assert len(kmi30["history"]) == 2
    assert kmi30["history"][0]["date"] == "2026-09-01"


def test_indices_handles_missing_column_gracefully(mocker):
    psx_cache.clear()
    _mock_conn(mocker, [
        {"date": datetime.date(2026, 9, 1), "kmi30_index_level": None, "kse30_index_level": 52000.0, "all_share_index_level": 106000.0},
    ])

    response = client.get("/api/indices")
    assert response.status_code == 200
    kmi30 = next(i for i in response.json()["indices"] if i["key"] == "kmi30")
    assert kmi30["level"] is None
    assert kmi30["history"] == []
