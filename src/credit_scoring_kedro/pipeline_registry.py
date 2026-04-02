"""Registre des pipelines Kedro — Scoring de Crédit.

Ce fichier déclare tous les pipelines disponibles dans le projet.
La commande `kedro run` exécute le pipeline __default__ (pipeline complet).
On peut aussi exécuter un pipeline individuel :
  kedro run --pipeline data_processing
  kedro run --pipeline feature_engineering
  kedro run --pipeline training
  kedro run --pipeline reporting
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from credit_scoring_kedro.pipelines.data_processing.pipeline import (
    create_pipeline as create_dp_pipeline,
)
from credit_scoring_kedro.pipelines.feature_engineering.pipeline import (
    create_pipeline as create_fe_pipeline,
)
from credit_scoring_kedro.pipelines.reporting.pipeline import (
    create_pipeline as create_rp_pipeline,
)
from credit_scoring_kedro.pipelines.training.pipeline import (
    create_pipeline as create_tr_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Enregistre et retourne tous les pipelines du projet."""

    dp_pipeline = create_dp_pipeline()
    fe_pipeline = create_fe_pipeline()
    tr_pipeline = create_tr_pipeline()
    rp_pipeline = create_rp_pipeline()

    return {
        "data_processing": dp_pipeline,
        "feature_engineering": fe_pipeline,
        "training": tr_pipeline,
        "reporting": rp_pipeline,
        "__default__": dp_pipeline + fe_pipeline + tr_pipeline + rp_pipeline,
    }
