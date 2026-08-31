from fastapi.testclient import TestClient
import pytest
import pandas as pd
import numpy as np
import torch
from src.psx_predictor.api.main import app, ml_models

client = TestClient(app)

def test_predict_invalid_ticker():
    response = client.post("/api/predict", json={"ticker": "INVALID"})
    assert response.status_code == 400
    assert "Invalid ticker" in response.json()["detail"]

def test_predict_valid_ticker(mocker):
    class MockModel:
        def predict(self, x):
            return np.ones(len(x)) * 100.0

    class MockLSTM:
        def __call__(self, x):
            return torch.ones(x.shape[0], 1)
            
    class MockScaler:
        def transform(self, x):
            return x
        def inverse_transform(self, x):
            return x

    mocker.patch.dict(ml_models, {
        "rf_predictor": MockModel(),
        "lr_predictor": MockModel(),
        "xgb_predictor": MockModel(),
        "lstm_predictor": MockLSTM(),
        "feature_scaler": MockScaler(),
        "target_scaler": MockScaler(),
        "xgb_ticker_categories": ["PSO", "MEBL"],
        "xgb_sector_categories": ["Oil & Gas", "Banking"],
        "ticker_to_id": {"PSO": 0, "MEBL": 1},
        "sector_to_id": {"Oil & Gas": 0, "Banking": 1}
    })
    
    mocker.patch("os.path.exists", return_value=True)
    
    dummy_df = pd.DataFrame({
        'date': pd.date_range(start='1/1/2020', periods=40).astype(str),
        'close': np.random.rand(40),
        'volume': np.random.randint(100, 1000, 40),
    })
    for i in range(19):
        dummy_df[f'feat_{i}'] = np.random.rand(40)
        
    mocker.patch("pandas.read_csv", return_value=dummy_df)

    response = client.post("/api/predict", json={"ticker": "PSO"})
    assert response.status_code == 200
    data = response.json()
    assert "rf_predicted_price" in data
    assert "lstm_predicted_price" in data
    assert "model_agreement_score" in data
    assert data["ticker"] == "PSO"
