"""Tests unitaires — GET /health/."""


def test_health_ok(client_solvable):
    """L'API retourne 200 et status=healthy quand le modèle est chargé."""
    response = client_solvable.get("/health/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"] == "1.0.0-test"
    assert body["environment"] == "production"


def test_health_degraded_sans_modele(client_no_model):
    """L'API retourne status=degraded quand le modèle n'est pas chargé."""
    response = client_no_model.get("/health/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["model_version"] == "unavailable"


def test_health_schema_complet(client_solvable):
    """La réponse contient tous les champs du schéma HealthResponse."""
    response = client_solvable.get("/health/")
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body
    assert "model_version" in body
    assert "environment" in body
