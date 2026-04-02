"""Tests unitaires — POST /predict/batch.

L'endpoint /predict/batch est protégé par JWT (scope batch).
Tous les appels utilisent la fixture `auth_headers` (data_scientist).
"""

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


class TestBatchPrediction:
    """Tests du endpoint de prédiction en batch."""

    def test_batch_status_200(self, client_solvable, auth_headers):
        """Retourne 200 pour un batch valide."""
        payload = {"records": [PROFIL_SOLVABLE, PROFIL_SOLVABLE]}
        response = client_solvable.post("/predict/batch", json=payload, headers=auth_headers)
        assert response.status_code == 200

    def test_batch_total_correct(self, client_solvable, auth_headers):
        """Le total retourné correspond au nombre de dossiers envoyés."""
        payload = {"records": [PROFIL_SOLVABLE] * 5}
        response = client_solvable.post("/predict/batch", json=payload, headers=auth_headers)
        assert response.json()["total"] == 5

    def test_batch_resultats_complets(self, client_solvable, auth_headers):
        """Chaque résultat contient tous les champs obligatoires."""
        payload = {"records": [PROFIL_SOLVABLE, PROFIL_SOLVABLE]}
        response = client_solvable.post("/predict/batch", json=payload, headers=auth_headers)
        for result in response.json()["results"]:
            assert "prediction" in result
            assert "probability_of_default" in result
            assert "risk_level" in result
            assert "recommendation" in result
            assert "latency_ms" in result
            assert "model_version" in result

    def test_batch_somme_decisions(self, client_solvable, auth_headers):
        """La somme approved + rejected + review = total."""
        payload = {"records": [PROFIL_SOLVABLE, PROFIL_SOLVABLE, PROFIL_SOLVABLE]}
        response = client_solvable.post("/predict/batch", json=payload, headers=auth_headers)
        body = response.json()
        assert body["approved"] + body["rejected"] + body["review"] == body["total"]

    def test_batch_latence_presente(self, client_solvable, auth_headers):
        """La latence batch est positive."""
        payload = {"records": [PROFIL_SOLVABLE]}
        response = client_solvable.post("/predict/batch", json=payload, headers=auth_headers)
        assert response.json()["batch_latency_ms"] > 0

    def test_batch_vide_422(self, client_solvable, auth_headers):
        """Un batch vide retourne 422 (contrainte min_length=1)."""
        payload = {"records": []}
        response = client_solvable.post("/predict/batch", json=payload, headers=auth_headers)
        assert response.status_code == 422

    def test_batch_un_seul_dossier(self, client_solvable, auth_headers):
        """Un batch avec un seul dossier fonctionne correctement."""
        payload = {"records": [PROFIL_SOLVABLE]}
        response = client_solvable.post("/predict/batch", json=payload, headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["total"] == 1

    def test_batch_mixte_solvable_risque(self, client_solvable, auth_headers):
        """Un batch avec des profils mixtes retourne les bons compteurs."""
        # Avec mock solvable : tous approuvés
        payload = {"records": [PROFIL_SOLVABLE, PROFIL_RISQUE]}
        response = client_solvable.post("/predict/batch", json=payload, headers=auth_headers)
        body = response.json()
        assert body["total"] == 2
        # Avec mock solvable, proba = 0.10 → tous APPROVE
        assert body["approved"] == 2
        assert body["rejected"] == 0

    def test_batch_modele_non_disponible_503(self, client_no_model, auth_headers):
        """Retourne 503 si le modèle n'est pas chargé.

        Note : auth_headers est issu de client_solvable — le token JWT
        est valide indépendamment du client utilisé pour la requête.
        """
        payload = {"records": [PROFIL_SOLVABLE]}
        response = client_no_model.post("/predict/batch", json=payload, headers=auth_headers)
        assert response.status_code == 503
