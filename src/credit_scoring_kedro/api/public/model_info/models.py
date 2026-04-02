"""Modèles Pydantic du domaine métadonnées du modèle ML.

Attributes:
    ModelMetadata: Modèle de réponse du endpoint GET /model/info.
"""

from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Métadonnées du modèle ML en production.

    Attributes:
        model_name: Nom du modèle.
        model_version: Version sémantique du modèle.
        algorithm: Algorithme utilisé (ex. GradientBoostingClassifier).
        features: Liste ordonnée des features utilisées à l'entraînement.
        metrics: Métriques de performance (ROC-AUC, F1, Precision, Recall…).
        training_date: Date d'entraînement (ISO 8601).
        threshold: Seuil de décision probabilité → classe binaire.
    """

    model_config = {"protected_namespaces": ()}

    model_name: str = Field(..., description="Nom du modèle")
    model_version: str = Field(..., description="Version du modèle")
    algorithm: str = Field(..., description="Algorithme ML utilisé")
    features: list[str] = Field(..., description="Features utilisées par le modèle")
    metrics: dict = Field(..., description="Métriques de performance")
    training_date: str = Field(..., description="Date d'entraînement (ISO 8601)")
    threshold: float = Field(..., description="Seuil de décision (0.0–1.0)")
