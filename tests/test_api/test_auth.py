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
    
    mocker.patch("joblib.load", return_value="mock_model")
    mocker.patch("torch.load", return_value="mock_state_dict")
    
    mock_lstm = mocker.Mock()
    mocker.patch("src.psx_predictor.api.main.LSTMModel", return_value=mock_lstm)
    
    response = client.post("/api/reload_models", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
