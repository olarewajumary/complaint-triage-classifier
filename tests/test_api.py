from fastapi.testclient import TestClient

from src import api


def must_not_be_called(text):
    raise AssertionError("predict() should not run for a request that fails validation")


def test_health_returns_ok():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_confident_prediction_has_no_top_guess(monkeypatch):
    monkeypatch.setattr(api, "predict", lambda text: {"prediction": "Mortgage", "confidence": 0.91})
    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "some complaint text"})
    assert response.status_code == 200
    assert response.json() == {"prediction": "Mortgage", "confidence": 0.91}


def test_human_review_includes_top_guess(monkeypatch):
    fake = {"prediction": "needs_human_review", "top_guess": "Mortgage", "confidence": 0.658}
    monkeypatch.setattr(api, "predict", lambda text: fake)
    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "some complaint text"})
    assert response.status_code == 200
    assert response.json() == fake


def test_invalid_input_becomes_422_with_the_message(monkeypatch):
    message = "Input too short to classify reliably (minimum 5 words)."
    monkeypatch.setattr(api, "predict", lambda text: {"prediction": "invalid_input", "error": message})
    client = TestClient(api.app)
    response = client.post("/predict", json={"text": "yes"})
    assert response.status_code == 422
    assert response.json() == {"detail": message}


def test_missing_text_field_is_422_before_predict_runs(monkeypatch):
    monkeypatch.setattr(api, "predict", must_not_be_called)
    client = TestClient(api.app)
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_number_instead_of_text_is_422_before_predict_runs(monkeypatch):
    monkeypatch.setattr(api, "predict", must_not_be_called)
    client = TestClient(api.app)
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422


def test_startup_loads_the_model_once(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "load_model", lambda: calls.append("loaded"))
    with TestClient(api.app) as client:
        client.get("/health")
    assert calls == ["loaded"]