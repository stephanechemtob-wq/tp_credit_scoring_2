"""Pipeline feature_engineering — Scoring de Crédit."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    add_derived_features,
    apply_preprocessing_test,
    apply_preprocessing_train,
    apply_preprocessing_val,
    build_preprocessor,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crée le pipeline de feature engineering."""
    return pipeline(
        [
            # Ajout des features dérivées sur les 3 splits
            node(
                func=add_derived_features,
                inputs="train_set",
                outputs="train_set_enriched",
                name="add_features_train_node",
                tags=["features"],
            ),
            node(
                func=add_derived_features,
                inputs="val_set",
                outputs="val_set_enriched",
                name="add_features_val_node",
                tags=["features"],
            ),
            node(
                func=add_derived_features,
                inputs="test_set",
                outputs="test_set_enriched",
                name="add_features_test_node",
                tags=["features"],
            ),
            # Construction du preprocessor (fit sur train uniquement — pas de leakage)
            node(
                func=build_preprocessor,
                inputs="train_set_enriched",
                outputs="preprocessor",
                name="build_preprocessor_node",
                tags=["features", "preprocessor"],
            ),
            # Application du preprocessor sur les 3 splits
            node(
                func=apply_preprocessing_train,
                inputs=["train_set_enriched", "preprocessor"],
                outputs="X_train",
                name="preprocess_train_node",
                tags=["features"],
            ),
            node(
                func=apply_preprocessing_val,
                inputs=["val_set_enriched", "preprocessor"],
                outputs="X_val",
                name="preprocess_val_node",
                tags=["features"],
            ),
            node(
                func=apply_preprocessing_test,
                inputs=["test_set_enriched", "preprocessor"],
                outputs="X_test",
                name="preprocess_test_node",
                tags=["features"],
            ),
        ]
    )
