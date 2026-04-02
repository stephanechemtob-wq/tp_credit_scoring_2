"""Pipeline data_processing — Scoring de Crédit.

Ce pipeline illustre la philosophie Kedro :
- Chaque node déclare ses entrées/sorties via le Data Catalog.
- Le même pipeline tourne localement (CSV) ou sur le cloud (S3/GCS/Azure Blob)
  sans modifier une seule ligne de code Python — seul le catalog.yml change.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_credit_dataset, split_data, validate_and_clean_data


def create_pipeline(**kwargs) -> Pipeline:
    """Crée le pipeline de traitement des données."""
    return pipeline(
        [
            node(
                func=generate_credit_dataset,
                inputs="params:data_processing",
                outputs="raw_credit_data",
                name="generate_credit_dataset_node",
                tags=["data", "generation"],
            ),
            node(
                func=validate_and_clean_data,
                inputs=["raw_credit_data", "params:data_processing"],
                outputs="cleaned_credit_data",
                name="validate_and_clean_data_node",
                tags=["data", "validation"],
            ),
            node(
                func=split_data,
                inputs=["cleaned_credit_data", "params:data_processing"],
                outputs=["train_set", "val_set", "test_set"],
                name="split_data_node",
                tags=["data", "split"],
            ),
        ]
    )
