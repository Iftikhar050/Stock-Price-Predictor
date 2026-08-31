from fastapi.testclient import TestClient
import os
from src.psx_predictor.api.main import app

client = TestClient(app)

def test_reload_models_no_key(mocker):
    mocker.patch.dict(os.environ, {"ADMIN_API_KEY": "test-key"})
    response = client.post("/api/reload_models")
    assert response.status_code == 401

def test_reload_models_invalid_key(mocker):
    mocker.patch.dict(os.environ, {"ADMIN_API_KEY": "test-key"})
    response = client.post("/api/reload_models", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401

def test_reload_models_valid_key(mocker):
    mocker.patch.dict(os.environ, {"ADMIN_API_KEY": "test-key"})
    
    mock_scaler = mocker.Mock()
    mock_scaler.n_features_in_ = 20
    
    def mock_joblib_load(path):
        if "categories" in path:
            return ["cat1", "cat2"]
        elif "scaler" in path:
            return mock_scaler
        elif "to_id" in path:
            return {"PSO": 0}
        return mocker.Mock()

    mocker.patch("joblib.load", side_effect=mock_joblib_load)
    
    import torch
    mock_state_dict = {
        'ticker_embed.weight': torch.zeros((10, 16)),
        'sector_embed.weight': torch.zeros((10, 16))
    }
    mocker.patch("torch.load", return_value=mock_state_dict)
    
    mock_lstm = mocker.Mock()
    mocker.patch("src.psx_predictor.api.main.LSTMModel", return_value=mock_lstm)
    
    response = client.post("/api/reload_models", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
