import copy
import pytest
from src.psx_predictor.api.state import ml_models


@pytest.fixture(autouse=True)
def _isolate_ml_models():
    """ml_models is a module-level singleton dict shared by the whole process
    (api/state.py), so any test that writes into it directly - not just via
    mocker.patch.dict, which cleans up after itself - leaks those entries into
    every test that runs afterward. Confirmed 2026-10-13:
    test_auth.py::test_reload_models_valid_key calls the real
    /api/reload_models endpoint with joblib.load mocked to return bare Mock()
    objects; that endpoint assigns them straight into ml_models
    (ml_models["pso_specialized"] = joblib.load(...)), which isn't undone by
    mocker's teardown, and silently broke test_predict.py's PSO prediction
    test whenever the two ran in the same session.
    """
    snapshot = copy.copy(ml_models)
    yield
    ml_models.clear()
    ml_models.update(snapshot)
