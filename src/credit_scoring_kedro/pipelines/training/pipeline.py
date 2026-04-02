"""Pipeline training — Scoring de Crédit."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    """Crée le pipeline d'entraînement et d'évaluation du modèle."""
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["X_train", "X_val", "params:training"],
                outputs=["credit_scoring_model", "train_val_metrics"],
                name="train_model_node",
                tags=["training", "mlflow"],
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "credit_scoring_model",
                    "X_test",
                    "train_val_metrics",
                    "params:training",
                ],
                outputs="model_metrics",
                name="evaluate_model_node",
                tags=["training", "evaluation", "mlflow"],
            ),
        ]
    )
