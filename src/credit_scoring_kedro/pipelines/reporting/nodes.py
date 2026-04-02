"""Nodes du pipeline reporting — Scoring de Crédit.

Ce pipeline génère les rapports d'explicabilité (SHAP) et les visualisations
de performance du modèle, sauvegardés dans le Data Catalog.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
from datetime import UTC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "default"


def generate_model_metadata(model_metrics: dict, params: dict) -> dict:
    """Génère les métadonnées du modèle pour l'API FastAPI.

    Ce nœud produit data/08_reporting/model_metadata.json, exporté
    ensuite par le hook ArtifactsExportHook vers data/api/model_metadata.json.

    Args:
        model_metrics: Métriques complètes (train + val + test) du pipeline training.
        params: Paramètres d'entraînement (model_name, hyperparams, etc.).

    Returns:
        Dictionnaire de métadonnées sérialisable en JSON.
    """
    from datetime import datetime

    algorithm_map = {
        "gradient_boosting": "GradientBoostingClassifier",
        "random_forest": "RandomForestClassifier",
        "logistic_regression": "LogisticRegression",
    }
    model_name = params.get("model_name", "gradient_boosting")
    algorithm = algorithm_map.get(model_name, model_name)

    metadata = {
        "model_name": "credit_scoring_model",
        "model_version": "1.0.0",
        "algorithm": algorithm,
        "feature_names": [
            "age",
            "income",
            "loan_amount",
            "loan_term",
            "credit_score",
            "employment_years",
            "debt_to_income",
            "num_credit_lines",
            "num_late_payments",
            "has_mortgage",
            "has_dependents",
            "loan_purpose",
            "education_level",
            "employment_type",
        ],
        "metrics": {
            "roc_auc_train": round(model_metrics.get("train_roc_auc", 0.0), 4),
            "roc_auc_val": round(model_metrics.get("val_roc_auc", 0.0), 4),
            "roc_auc_test": round(model_metrics.get("test_roc_auc", 0.0), 4),
            "f1_train": round(model_metrics.get("train_f1", 0.0), 4),
            "f1_val": round(model_metrics.get("val_f1", 0.0), 4),
            "f1_test": round(model_metrics.get("test_f1", 0.0), 4),
            "precision_test": round(model_metrics.get("test_precision", 0.0), 4),
            "recall_test": round(model_metrics.get("test_recall", 0.0), 4),
            "accuracy_test": round(model_metrics.get("test_accuracy", 0.0), 4),
        },
        "training_date": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "threshold": 0.5,
    }

    logger.info(
        "Métadonnées du modèle générées — ROC-AUC test: %.4f | F1 test: %.4f",
        metadata["metrics"]["roc_auc_test"],
        metadata["metrics"]["f1_test"],
    )
    return metadata


def generate_performance_report(model_metrics: dict) -> pd.DataFrame:
    """Génère un rapport de performance synthétique sous forme de DataFrame.

    Args:
        model_metrics: Dictionnaire de métriques issu du pipeline training.

    Returns:
        DataFrame formaté pour export CSV/Excel via le Data Catalog.
    """
    splits = ["train", "val", "test"]
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    rows = []
    for split in splits:
        row = {"split": split}
        for metric in metric_names:
            key = f"{split}_{metric}"
            row[metric] = round(model_metrics.get(key, 0.0), 4)
        rows.append(row)

    report_df = pd.DataFrame(rows)
    logger.info("Rapport de performance généré :\n%s", report_df.to_string(index=False))
    return report_df


def generate_feature_importance_plot(
    model,
    X_train: pd.DataFrame,
) -> plt.Figure:
    """Génère le graphique d'importance des features.

    Compatible avec RandomForest et GradientBoosting (feature_importances_).
    Pour LogisticRegression, utilise les coefficients absolus.

    Args:
        model: Modèle entraîné.
        X_train: Features d'entraînement (pour les noms de colonnes).

    Returns:
        Figure matplotlib sauvegardée dans le Data Catalog.
    """
    feature_names = [c for c in X_train.columns if c != TARGET]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Importance des Features (Tree-based)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = "Importance des Features (|Coefficients| Logistic Regression)"
    else:
        logger.warning("Modèle sans attribut d'importance. Graphique vide.")
        return plt.figure()

    # Tri par importance décroissante
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#E84040" if v > np.median(sorted_values) else "#4A90D9" for v in sorted_values]
    _ = ax.barh(sorted_names[::-1], sorted_values[::-1], color=colors[::-1])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines["bottom"].set_color("#4A90D9")
    ax.spines["left"].set_color("#4A90D9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    logger.info("Graphique d'importance des features généré.")
    return fig


def generate_roc_curve_plot(
    model,
    X_test: pd.DataFrame,
) -> plt.Figure:
    """Génère la courbe ROC sur le jeu de test.

    Args:
        model: Modèle entraîné.
        X_test: Features de test avec la cible.

    Returns:
        Figure matplotlib de la courbe ROC.
    """
    from sklearn.metrics import auc, roc_curve

    X_te = X_test.drop(columns=[TARGET])
    y_te = X_test[TARGET]

    y_proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#E84040", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--", label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#E84040")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, color="white")
    ax.set_ylabel("True Positive Rate", fontsize=12, color="white")
    ax.set_title("Courbe ROC — Scoring de Crédit", fontsize=14, fontweight="bold", color="white")
    ax.legend(loc="lower right", facecolor="#1a1a2e", labelcolor="white")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#4A90D9")
    ax.spines["left"].set_color("#4A90D9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    logger.info("Courbe ROC générée. AUC = %.4f", roc_auc)
    return fig
