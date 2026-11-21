import json

from src.psx_predictor.api.routers import predict

HOLDOUT = {
    "rf": {"test_start": "2025-10-28", "test_end": "2026-11-05", "tickers": {"HBL": {"accuracy": 51.2, "days": 250}}},
    "xgb": {"test_start": "2025-10-28", "test_end": "2026-11-05", "tickers": {"HBL": {"accuracy": 49.6, "days": 250}}},
    "sector": {"test_start": "2025-10-28", "test_end": "2026-11-05", "tickers": {"HBL": {"accuracy": 50.4, "days": 248}}},
}


def _use_holdout(tmp_path, monkeypatch, data=HOLDOUT):
    path = tmp_path / "holdout_accuracy.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(predict, "HOLDOUT_PATH", str(path))
    monkeypatch.setattr(predict, "_holdout_cache", {"mtime": None, "data": {}})


def test_pooled_and_sector_models_use_holdout(tmp_path, monkeypatch):
    _use_holdout(tmp_path, monkeypatch)
    acc, days, basis = predict._unseen_accuracy("HBL", None)
    assert acc == {"rf": 51.2, "xgb": 49.6, "specialized": 50.4}
    assert days == {"rf": 250, "xgb": 250, "specialized": 248}
    assert basis["rf"] == "held-out test, 28 Oct 2025 – 5 Nov 2026"
    assert basis["specialized"].startswith("sector model, held-out test")


def test_specialized_model_uses_its_walk_forward_score(tmp_path, monkeypatch):
    _use_holdout(tmp_path, monkeypatch)
    spec = {"model_type": "xgboost", "walk_forward_summary": {"xgboost": {"mean_da": 49.188, "n_windows": 20}}}
    acc, days, basis = predict._unseen_accuracy("HBL", spec)
    assert acc["specialized"] == 49.2
    assert "specialized" not in days
    assert basis["specialized"] == "walk-forward test over 20 periods"


def test_unmeasured_models_are_left_out(tmp_path, monkeypatch):
    _use_holdout(tmp_path, monkeypatch)
    acc, _, _ = predict._unseen_accuracy("PSO", None)
    # No held-out record for PSO and no LR/LSTM figures at all: nothing is invented.
    assert acc == {}


def test_missing_file_gives_no_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "HOLDOUT_PATH", str(tmp_path / "absent.json"))
    assert predict._unseen_accuracy("HBL", None) == ({}, {}, {})
