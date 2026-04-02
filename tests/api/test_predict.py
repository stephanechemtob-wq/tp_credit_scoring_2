"""Tests unitaires — POST /predict/."""

# Constantes de test (définies localement pour éviter les problèmes d'import pytest)
PROFIL_SOLVABLE = {
    "age": 42.0,
    "income": 75000.0,
    "loan_amount": 10000.0,
    "loan_term": 24.0,
    "credit_score": 760.0,
    "employment_years": 12.0,
    "debt_to_income": 0.15,
    "num_credit_lines": 5.0,
    "num_late_payments": 0.0,
    "has_mortgage": 1.0,
    "has_dependents": 0.0,
    "loan_purpose": "home_improvement",
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


class TestPredictSolvable:
    """Tests avec un profil solvable (proba défaut = 0.10)."""

    def test_status_200(self, client_solvable, auth_headers):
        response = client_solvable.post("/predict/", json=PROFIL_SOLVABLE, headers=auth_headers)
        assert response.status_code == 200

    def test_prediction_zero(self, client_solvable, auth_headers):
        """Prédiction = 0 (solvable) quand proba < seuil."""
        response = client_solvable.post("/predict/", json=PROFIL_SOLVABLE, headers=auth_headers)
        assert response.json()["prediction"] == 0

    def test_probabilite_faible(self, client_solvable, auth_headers):
        """Probabilité de défaut faible pour un profil solvable."""
        response = client_solvable.post("/predict/", json=PROFIL_SOLVABLE, headers=auth_headers)
        proba = response.json()["probability_of_default"]
        assert 0.0 <= proba <= 1.0
        assert proba < 0.5

    def test_risk_level_low(self, client_solvable, auth_headers):
        """Niveau de risque LOW pour proba = 0.10."""
        response = client_solvable.post("/predict/", json=PROFIL_SOLVABLE, headers=auth_headers)
        assert response.json()["risk_level"] == "LOW"

    def test_recommendation_approve(self, client_solvable, auth_headers):
        """Recommandation APPROVE pour profil solvable."""
        response = client_solvable.post("/predict/", json=PROFIL_SOLVABLE, headers=auth_headers)
        assert response.json()["recommendation"] == "APPROVE"

    def test_risk_score_faible(self, client_solvable, auth_headers):
        """Score de risque normalisé entre 0 et 100."""
        response = client_solvable.post("/predict/", json=PROFIL_SOLVABLE, headers=auth_headers)
        score = response.json()["risk_score"]
        assert 0 <= score <= 100
        assert score < 50

    def test_latence_presente(self, client_solvable, auth_headers):
        """La latence est présente et positive."""
        response = client_solvable.post("/predict/", json=PROFIL_SOLVABLE, headers=auth_headers)
        assert response.json()["latency_ms"] > 0

    def test_version_modele_presente(self, client_solvable, auth_headers):
        """La version du modèle est retournée."""
        response = client_solvable.post("/predict/", json=PROFIL_SOLVABLE, headers=auth_headers)
        assert response.json()["model_version"] == "1.0.0-test"


class TestPredictRisque:
    """Tests avec un profil à risque élevé (proba défaut = 0.85)."""

    def test_prediction_un(self, client_risque, auth_headers_risque):
        """Prédiction = 1 (défaut) quand proba > seuil."""
        response = client_risque.post("/predict/", json=PROFIL_RISQUE, headers=auth_headers_risque)
        assert response.status_code == 200
        assert response.json()["prediction"] == 1

    def test_probabilite_elevee(self, client_risque, auth_headers_risque):
        """Probabilité de défaut élevée pour un profil risqué."""
        response = client_risque.post("/predict/", json=PROFIL_RISQUE, headers=auth_headers_risque)
        assert response.json()["probability_of_default"] > 0.5

    def test_risk_level_critical(self, client_risque, auth_headers_risque):
        """Niveau de risque CRITICAL pour proba = 0.85."""
        response = client_risque.post("/predict/", json=PROFIL_RISQUE, headers=auth_headers_risque)
        assert response.json()["risk_level"] == "CRITICAL"

    def test_recommendation_reject(self, client_risque, auth_headers_risque):
        """Recommandation REJECT pour profil à risque critique."""
        response = client_risque.post("/predict/", json=PROFIL_RISQUE, headers=auth_headers_risque)
        assert response.json()["recommendation"] == "REJECT"

    def test_risk_score_eleve(self, client_risque, auth_headers_risque):
        """Score de risque > 50 pour profil risqué."""
        response = client_risque.post("/predict/", json=PROFIL_RISQUE, headers=auth_headers_risque)
        assert response.json()["risk_score"] > 50


class TestPredictValidation:
    """Tests de validation des données d'entrée (Pydantic)."""

    def test_donnees_invalides_422(self, client_solvable, auth_headers):
        """Retourne 422 pour des données hors des contraintes Pydantic."""
        response = client_solvable.post("/predict/", json=PROFIL_INVALIDE, headers=auth_headers)
        assert response.status_code == 422

    def test_erreurs_validation_detaillees(self, client_solvable, auth_headers):
        """Le corps de la réponse 422 contient les détails des erreurs."""
        response = client_solvable.post("/predict/", json=PROFIL_INVALIDE, headers=auth_headers)
        body = response.json()
        assert "detail" in body
        assert isinstance(body["detail"], list)
        assert len(body["detail"]) > 0

    def test_champ_manquant_422(self, client_solvable, auth_headers):
        """Retourne 422 si un champ obligatoire est absent."""
        profil_incomplet = {k: v for k, v in PROFIL_SOLVABLE.items() if k != "credit_score"}
        response = client_solvable.post("/predict/", json=profil_incomplet, headers=auth_headers)
        assert response.status_code == 422

    def test_loan_purpose_invalide_422(self, client_solvable, auth_headers):
        """Retourne 422 si loan_purpose n'est pas dans les valeurs attendues."""
        profil = {**PROFIL_SOLVABLE, "loan_purpose": "casino"}
        response = client_solvable.post("/predict/", json=profil, headers=auth_headers)
        assert response.status_code == 422

    def test_modele_non_disponible_503(self, client_no_model, auth_headers):
        """Retourne 503 si le modèle n'est pas chargé.

        Note : client_no_model n'a pas de modèle mais l'auth doit quand même
        fonctionner. On utilise auth_headers (issu de client_solvable) — le
        token JWT est valide indépendamment du client utilisé pour la requête.
        """
        response = client_no_model.post("/predict/", json=PROFIL_SOLVABLE, headers=auth_headers)
        assert response.status_code == 503
