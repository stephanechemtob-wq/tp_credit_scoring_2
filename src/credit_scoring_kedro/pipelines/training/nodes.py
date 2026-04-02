"""Nodes du pipeline training — Scoring de Crédit avec MLflow.

Principe MLOps Kedro + MLflow :
- Kedro orchestre le pipeline et gère les données via le Data Catalog.
- MLflow enregistre chaque expérience (hyperparamètres, métriques, modèle).
- Le hook KedroMlflowHook connecte automatiquement les deux.
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

TARGET = "default"


def _compute_metrics(y_true, y_pred, y_proba, prefix: str) -> dict:
    """Calcule les métriques de classification standard."""
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    params: dict,
) -> tuple:
    """Entraîne le modèle de scoring de crédit et loggue dans MLflow.

    Principe Kedro : cette fonction est un node pur. Elle reçoit ses données
    via le Data Catalog et ses hyperparamètres via parameters.yml.
    Elle est cloud-agnostique : les données peuvent venir de S3, GCS ou Azure
    sans changer cette fonction.

    Args:
        X_train: Features d'entraînement avec la cible.
        X_val: Features de validation avec la cible.
        params: Hyperparamètres issus de conf/base/parameters/training.yml.

    Returns:
        Tuple (modèle entraîné, métriques dict).
    """
    model_name = params["model_name"]
    hyperparams = params["hyperparams"]

    X_tr = X_train.drop(columns=[TARGET])
    y_tr = X_train[TARGET]
    X_v = X_val.drop(columns=[TARGET])
    y_v = X_val[TARGET]

    # Sélection du modèle
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=hyperparams.get("C", 1.0),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", 10),
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),
            learning_rate=hyperparams.get("learning_rate", 0.1),
            max_depth=hyperparams.get("max_depth", 5),
            random_state=42,
        ),
    }

    if model_name not in models:
        raise ValueError(f"Modèle inconnu : {model_name}. Choix : {list(models.keys())}")

    model = models[model_name]
    logger.info("Entraînement du modèle : %s", model_name)

    # Entraînement
    model.fit(X_tr, y_tr)

    # Métriques train
    train_metrics = _compute_metrics(
        y_tr, model.predict(X_tr), model.predict_proba(X_tr)[:, 1], "train"
    )
    # Métriques validation
    val_metrics = _compute_metrics(y_v, model.predict(X_v), model.predict_proba(X_v)[:, 1], "val")

    all_metrics = {**train_metrics, **val_metrics}

    # Log MLflow (kedro-mlflow connecte automatiquement le run actif)
    mlflow.log_param("model_name", model_name)
    mlflow.log_params(hyperparams)
    mlflow.log_metrics(all_metrics)

    # Enregistrer le modèle dans le Model Registry
    mlflow.sklearn.log_model(
        model, artifact_path="model", registered_model_name="credit-scoring-model"
    )

    for split, metrics in [("TRAIN", train_metrics), ("VAL", val_metrics)]:
        logger.info(
            "%s — ROC-AUC: %.4f | F1: %.4f",
            split,
            metrics[f"{split.lower()}_roc_auc"],
            metrics[f"{split.lower()}_f1"],
        )

    return model, all_metrics


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    all_metrics: dict,
    params: dict,
) -> dict:
    """Évalue le modèle sur le jeu de test et valide les seuils de performance.

    Args:
        model: Modèle entraîné.
        X_test: Features de test avec la cible.
        all_metrics: Métriques train/val déjà calculées.
        params: Seuils de validation issus de parameters/training.yml.

    Returns:
        Dictionnaire complet des métriques (train + val + test).
    """
    X_te = X_test.drop(columns=[TARGET])
    y_te = X_test[TARGET]

    test_metrics = _compute_metrics(
        y_te, model.predict(X_te), model.predict_proba(X_te)[:, 1], "test"
    )

    mlflow.log_metrics(test_metrics)

    final_metrics = {**all_metrics, **test_metrics}

    # Validation des seuils
    thresholds = params.get("validation_thresholds", {})
    failures = []
    for metric_name, threshold in thresholds.items():
        value = final_metrics.get(f"test_{metric_name}", 0)
        if value < threshold:
            failures.append(f"test_{metric_name}={value:.4f} < seuil={threshold}")

    if failures:
        logger.warning("Seuils non atteints :\n  - %s", "\n  - ".join(failures))
    else:
        logger.info("Tous les seuils de performance sont atteints. Promotion en Production.")
        try:
            client = mlflow.tracking.MlflowClient()
            run_id = mlflow.active_run().info.run_id

            # Chercher la version du modèle associée à ce run
            for mv in client.search_model_versions("name='credit-scoring-model'"):
                if mv.run_id == run_id:
                    client.transition_model_version_stage(
                        name="credit-scoring-model",
                        version=mv.version,
                        stage="Production",
                        archive_existing_versions=True,
                    )
                    logger.info(f"Modèle promu en Production (version {mv.version})")
                    break
        except Exception as e:
            logger.error(f"Erreur lors de la promotion du modèle : {e}")

    logger.info(
        "TEST — ROC-AUC: %.4f | F1: %.4f", test_metrics["test_roc_auc"], test_metrics["test_f1"]
    )

    return final_metrics
