"""Tests unitaires — GET /model/info.

L'endpoint /model/info est protégé par JWT (scope model:read).
Tous les appels utilisent la fixture `auth_headers` (data_scientist).
"""


class TestModelInfo:
    """Tests du endpoint d'informations sur le modèle."""

    def test_model_info_status_200(self, client_solvable, auth_headers):
        """Retourne 200 pour le endpoint /model/info."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        assert response.status_code == 200

    def test_model_name_present(self, client_solvable, auth_headers):
        """Le nom du modèle est présent et correct."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        assert response.json()["model_name"] == "credit_scoring_model"

    def test_model_version_correcte(self, client_solvable, auth_headers):
        """La version du modèle correspond à celle injectée."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        assert response.json()["model_version"] == "1.0.0-test"

    def test_algorithme_present(self, client_solvable, auth_headers):
        """L'algorithme du modèle est défini."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        assert response.json()["algorithm"] == "GradientBoostingClassifier"

    def test_features_non_vides(self, client_solvable, auth_headers):
        """La liste des features est non vide."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        features = response.json()["features"]
        assert isinstance(features, list)
        assert len(features) > 0

    def test_features_contiennent_age_et_income(self, client_solvable, auth_headers):
        """Les features clés (age, income, credit_score) sont présentes."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        features = response.json()["features"]
        assert "age" in features
        assert "income" in features
        assert "credit_score" in features

    def test_metriques_roc_auc(self, client_solvable, auth_headers):
        """Les métriques contiennent roc_auc_test."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        metrics = response.json()["metrics"]
        assert "roc_auc_test" in metrics
        assert metrics["roc_auc_test"] == 0.765

    def test_seuil_entre_0_et_1(self, client_solvable, auth_headers):
        """Le seuil de décision est entre 0 et 1."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        threshold = response.json()["threshold"]
        assert 0.0 <= threshold <= 1.0

    def test_date_entrainement_presente(self, client_solvable, auth_headers):
        """La date d'entraînement est présente."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        assert response.json()["training_date"] == "2025-01-01"

    def test_schema_complet(self, client_solvable, auth_headers):
        """Tous les champs du schéma ModelMetadata sont présents."""
        response = client_solvable.get("/model/info", headers=auth_headers)
        body = response.json()
        required_fields = [
            "model_name",
            "model_version",
            "algorithm",
            "features",
            "metrics",
            "training_date",
            "threshold",
        ]
        for field in required_fields:
            assert field in body, f"Champ manquant : {field}"
