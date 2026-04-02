"""Pipeline reporting — Scoring de Crédit."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    generate_feature_importance_plot,
    generate_model_metadata,
    generate_performance_report,
    generate_roc_curve_plot,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crée le pipeline de reporting et visualisation."""
    return pipeline(
        [
            node(
                func=generate_model_metadata,
                inputs=["model_metrics", "params:training"],
                outputs="model_metadata",
                name="generate_model_metadata_node",
                tags=["reporting", "metadata"],
            ),
            node(
                func=generate_performance_report,
                inputs="model_metrics",
                outputs="performance_report",
                name="generate_performance_report_node",
                tags=["reporting"],
            ),
            node(
                func=generate_feature_importance_plot,
                inputs=["credit_scoring_model", "X_train"],
                outputs="feature_importance_plot",
                name="generate_feature_importance_plot_node",
                tags=["reporting", "visualization"],
            ),
            node(
                func=generate_roc_curve_plot,
                inputs=["credit_scoring_model", "X_test"],
                outputs="roc_curve_plot",
                name="generate_roc_curve_plot_node",
                tags=["reporting", "visualization"],
            ),
        ]
    )
