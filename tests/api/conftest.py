"""Fixtures pytest pour les tests de l'API Credit Scoring.

Architecture des fixtures :
- mock_model / mock_preprocessor : remplacent les artefacts Kedro (pas besoin de kedro run)
- app_state : injecte directement les mocks dans app.state
- client : TestClient FastAPI prêt à l'emploi
- auth_headers : headers JWT valides pour les endpoints protégés

Pattern : on bypasse le lifespan Kedro pour les tests unitaires en
injectant directement les mocks dans app.state après création de l'app.
Les endpoints protégés par require_scope() nécessitent un token JWT
réel — obtenu via POST /auth/token dans la fixture auth_headers.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from credit_scoring_kedro.api.app import create_app
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Données de test réutilisables
# ─────────────────────────────────────────────────────────────────────────────

PROFIL_SOLVABLE = {
    "age": 42.0,
    "income": 75000.0,
    "loan_amount": 10000.0,
    "loan_term": 24.0,
    "credit_score": 760.0,
    "employment_years": 12.0,
    "debt_to_income": 0.15,
    "num_credit_lines": 6.0,
    "num_late_payments": 0.0,
    "has_mortgage": 1.0,
    "has_dependents": 0.0,
    "loan_purpose": "personal",
    "education_level": "master",
    "employment_type": "employed",
}

PROFIL_RISQUE = {
    "age": 23.0,
    "income": 18000.0,
    "loan_amount": 25000.0,
    "loan_term": 60.0,
    "credit_score": 340.0,
    "employment_years": 0.5,
    "debt_to_income": 0.85,
    "num_credit_lines": 1.0,
    "num_late_payments": 5.0,
    "has_mortgage": 0.0,
    "has_dependents": 1.0,
    "loan_purpose": "personal",
    "education_level": "high_school",
    "employment_type": "unemployed",
}

PROFIL_INVALIDE = {
    "age": -5.0,  # invalide : < 18
    "income": -1000.0,  # invalide : < 0
    "loan_amount": 10000.0,
    "loan_term": 24.0,
    "credit_score": 999.0,  # invalide : > 850
    "employment_years": 5.0,
    "debt_to_income": 2.5,  # invalide : > 1
    "num_credit_lines": 3.0,
    "num_late_payments": 0.0,
    "has_mortgage": 0.0,
    "has_dependents": 0.0,
    "loan_purpose": "personal",
    "education_level": "bachelor",
    "employment_type": "employed",
}

MODEL_METADATA = {
    "model_name": "credit_scoring_model",
    "model_version": "1.0.0-test",
    "algorithm": "GradientBoostingClassifier",
    "feature_names": list(PROFIL_SOLVABLE.keys()),
    "metrics": {
        "roc_auc_test": 0.765,
        "f1_test": 0.481,
        "accuracy_test": 0.732,
    },
    "training_date": "2025-01-01",
    "threshold": 0.5,
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — mocks ML
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_model_solvable():
    """Modèle mock qui prédit toujours 'solvable' (proba défaut = 0.10)."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.90, 0.10]])
    return model


@pytest.fixture
def mock_model_risque():
    """Modèle mock qui prédit toujours 'défaut' (proba défaut = 0.85)."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.15, 0.85]])
    return model


@pytest.fixture
def mock_preprocessor():
    """Preprocesseur mock qui retourne les données telles quelles (tableau numpy)."""
    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.zeros((1, 14))
    return preprocessor


@pytest.fixture
def mock_preprocessor_batch():
    """Preprocesseur mock pour les prédictions batch."""
    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.zeros((1, 14))
    return preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — clients HTTP
# ─────────────────────────────────────────────────────────────────────────────


def _build_client(model_mock, preprocessor_mock) -> TestClient:
    """Construit un TestClient avec les mocks injectés dans app.state."""
    app = create_app()

    # Injection directe dans app.state (bypass du lifespan Kedro)
    app.state.model = model_mock
    app.state.preprocessor = preprocessor_mock
    app.state.model_version = MODEL_METADATA["model_version"]
    app.state.model_name = MODEL_METADATA["model_name"]
    app.state.algorithm = MODEL_METADATA["algorithm"]
    app.state.feature_names = MODEL_METADATA["feature_names"]
    app.state.metrics = MODEL_METADATA["metrics"]
    app.state.training_date = MODEL_METADATA["training_date"]
    app.state.threshold = MODEL_METADATA["threshold"]

    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def client(mock_model_solvable, mock_preprocessor):
    """TestClient générique (alias de client_solvable) pour les tests de sécurité."""
    return _build_client(mock_model_solvable, mock_preprocessor)


@pytest.fixture
def mock_model_state(mock_model_solvable, mock_preprocessor):
    """Mock combiné modèle + préprocesseur pour les tests de scopes."""
    return MagicMock(model=mock_model_solvable, preprocessor=mock_preprocessor)


@pytest.fixture
def client_solvable(mock_model_solvable, mock_preprocessor):
    """TestClient configuré avec un modèle qui prédit 'solvable'."""
    return _build_client(mock_model_solvable, mock_preprocessor)


@pytest.fixture
def client_risque(mock_model_risque, mock_preprocessor):
    """TestClient configuré avec un modèle qui prédit 'défaut'."""
    return _build_client(mock_model_risque, mock_preprocessor)


@pytest.fixture
def client_no_model():
    """TestClient sans modèle chargé (mode dégradé)."""
    app = create_app()
    app.state.model = None
    app.state.preprocessor = None
    app.state.model_version = "unavailable"
    app.state.model_name = "unavailable"
    app.state.algorithm = "unavailable"
    app.state.feature_names = []
    app.state.metrics = {}
    app.state.training_date = "unavailable"
    app.state.threshold = 0.5
    return TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — authentification JWT
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def auth_headers(client_solvable: TestClient) -> dict:
    """Headers JWT valides pour data_scientist (scopes : predict, batch, model:read).

    Obtient un token réel via POST /auth/token — les endpoints protégés
    par require_scope() nécessitent un token JWT valide, pas un mock.

    Args:
        client_solvable: TestClient avec modèle solvable injecté.

    Returns:
        Dict {"Authorization": "Bearer <token>"} prêt à passer en headers.
    """
    response = client_solvable.post(
        "/auth/token",
        data={"username": "data_scientist", "password": "mlops2024"},
    )
    assert response.status_code == 200, f"Auth échouée : {response.text}"
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_risque(client_risque: TestClient) -> dict:
    """Headers JWT valides pour le client avec modèle risqué.

    Args:
        client_risque: TestClient avec modèle risqué injecté.

    Returns:
        Dict {"Authorization": "Bearer <token>"} prêt à passer en headers.
    """
    response = client_risque.post(
        "/auth/token",
        data={"username": "data_scientist", "password": "mlops2024"},
    )
    assert response.status_code == 200, f"Auth échouée : {response.text}"
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def readonly_headers(client: TestClient) -> dict:
    """Headers JWT pour l'utilisateur readonly (scope model:read uniquement).

    Args:
        client: TestClient générique.

    Returns:
        Dict {"Authorization": "Bearer <token>"} pour readonly.
    """
    response = client.post(
        "/auth/token",
        data={"username": "readonly", "password": "readonly123"},
    )
    assert response.status_code == 200, f"Auth readonly échouée : {response.text}"
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
