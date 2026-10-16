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


# All columns _compute_indices() selects - a real DB row always has every key
# present (NULL where no source populated it yet), so mocked rows must match
# that shape too, not just the columns a given test cares about.
_ALL_COLS = {
    "kse100_real_index_level": None, "kse100_real_return_pct": None,
    "kmi30_index_level": None, "kmi30_return_pct": None,
    "kse30_index_level": None, "kse30_return_pct": None,
    "all_share_index_level": None, "all_share_return_pct": None,
}


def _row(date, **overrides):
    return {"date": date, **_ALL_COLS, **overrides}


def test_indices_includes_real_kse100(mocker):
    """kse100_real_index_level is the real PSX-published index (added
    2026-10-13, see psx_dps_index_scraper.py) - distinct from MarketIndex's
    synthetic "KSE100" proxy, which is never surfaced here."""
    psx_cache.clear()
    _mock_conn(mocker, [
        _row(datetime.date(2026, 9, 1), kse100_real_index_level=160000.0, kmi30_index_level=250000.0, kse30_index_level=52000.0, all_share_index_level=106000.0),
        _row(datetime.date(2026, 9, 4), kse100_real_index_level=164776.44, kse100_real_return_pct=-0.0021, kmi30_index_level=250239.14, kse30_index_level=52294.83, all_share_index_level=106442.83),
    ])

    response = client.get("/api/indices")
    assert response.status_code == 200
    data = response.json()
    keys = [i["key"] for i in data["indices"]]
    assert set(keys) == {"kse100", "kmi30", "kse30", "all_share"}
    kse100 = next(i for i in data["indices"] if i["key"] == "kse100")
    assert kse100["level"] == 164776.44
    assert kse100["change_percent"] == -0.21


def test_indices_change_percent_and_history(mocker):
    psx_cache.clear()
    _mock_conn(mocker, [
        _row(datetime.date(2026, 9, 1), kmi30_index_level=250000.0, kse30_index_level=52000.0, all_share_index_level=106000.0),
        _row(datetime.date(2026, 9, 4), kmi30_index_level=250250.0, kse30_index_level=52260.0, all_share_index_level=106318.0),
    ])

    response = client.get("/api/indices")
    data = response.json()
    kmi30 = next(i for i in data["indices"] if i["key"] == "kmi30")
    assert kmi30["level"] == 250250.0
    assert kmi30["change_percent"] == 0.1
    assert len(kmi30["history"]) == 2
    assert kmi30["history"][0]["date"] == "2026-09-01"


def test_indices_prefers_stored_return_pct_over_history_diff(mocker):
    """When the sync wrote PSX's own reported daily % change alongside the
    level, that's authoritative over diffing the last two available history
    rows - which can span a large gap (e.g. the index-sync outage from
    2026-09-16 to 2026-10-13) and silently produce a wrong "daily" figure."""
    psx_cache.clear()
    _mock_conn(mocker, [
        _row(datetime.date(2026, 9, 15), kmi30_index_level=241108.75),
        _row(datetime.date(2026, 10, 13), kmi30_index_level=237568.77, kmi30_return_pct=-0.0010),
    ])

    response = client.get("/api/indices")
    kmi30 = next(i for i in response.json()["indices"] if i["key"] == "kmi30")
    assert kmi30["change_percent"] == -0.1


def test_indices_handles_missing_column_gracefully(mocker):
    psx_cache.clear()
    _mock_conn(mocker, [
        _row(datetime.date(2026, 9, 1), kse30_index_level=52000.0, all_share_index_level=106000.0),
    ])

    response = client.get("/api/indices")
    assert response.status_code == 200
    kmi30 = next(i for i in response.json()["indices"] if i["key"] == "kmi30")
    assert kmi30["level"] is None
    assert kmi30["history"] == []
