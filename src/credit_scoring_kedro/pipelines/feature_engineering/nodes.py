"""Nodes du pipeline feature_engineering — Scoring de Crédit.

Principe MLOps Kedro : le preprocessor scikit-learn est lui-même
versionné dans le Data Catalog comme un artefact (joblib), ce qui
garantit que le même transformateur est utilisé à l'entraînement
et en inférence — sans fuite de données.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "age",
    "income",
    "loan_amount",
    "loan_duration_months",
    "credit_score",
    "num_credit_lines",
    "employment_years",
    "debt_to_income_ratio",
    "num_late_payments",
]
BINARY_FEATURES = ["has_mortgage"]
TARGET = "default"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features dérivées métier au DataFrame.

    Ces features capturent des relations non-linéaires importantes
    pour le scoring de crédit (ratio mensualité/revenu, score de risque...).

    Args:
        df: DataFrame avec les features brutes.

    Returns:
        DataFrame enrichi avec les nouvelles features.
    """
    df = df.copy()

    # Ratio mensualité / revenu mensuel
    monthly_income = df["income"] / 12
    monthly_payment = df["loan_amount"] / df["loan_duration_months"]
    df["monthly_payment_ratio"] = (monthly_payment / monthly_income).clip(0, 5)

    # Score de risque composite (feature métier)
    df["risk_score"] = (
        (850 - df["credit_score"]) / 550 * 0.4
        + df["debt_to_income_ratio"] * 0.3
        + (df["num_late_payments"] / 10).clip(0, 1) * 0.3
    )

    # Tranche d'âge (encodage ordinal)
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 100], labels=[0, 1, 2, 3]).astype(int)

    # Transformations logarithmiques (réduction de l'asymétrie)
    df["log_income"] = np.log1p(df["income"])
    df["log_loan_amount"] = np.log1p(df["loan_amount"])

    logger.info("Features dérivées ajoutées : %d features au total", len(df.columns))
    return df


def build_preprocessor(train_set: pd.DataFrame) -> Pipeline:
    """Construit et ajuste le preprocessor scikit-learn sur les données d'entraînement.

    Principe MLOps Kedro : le preprocessor est un artefact du Data Catalog.
    Il est sauvegardé automatiquement par Kedro (joblib) et rechargé à l'inférence
    — garantissant l'absence de data leakage.

    Args:
        train_set: Données d'entraînement enrichies.

    Returns:
        Pipeline scikit-learn ajusté (fit) sur train_set.
    """
    all_numeric = NUMERIC_FEATURES + [
        "monthly_payment_ratio",
        "risk_score",
        "age_group",
        "log_income",
        "log_loan_amount",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_numeric),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )

    X_train = train_set.drop(columns=[TARGET])
    preprocessor.fit(X_train)

    logger.info(
        "Preprocessor ajusté sur %d échantillons, %d features", len(train_set), X_train.shape[1]
    )
    return preprocessor


def apply_preprocessing(
    dataset: pd.DataFrame, preprocessor: Pipeline, split_name: str = "unknown"
) -> pd.DataFrame:
    """Applique le preprocessor à un dataset (train, val ou test).

    Args:
        dataset: DataFrame à transformer.
        preprocessor: Preprocessor ajusté (issu du Data Catalog).
        split_name: Nom du split pour les logs.

    Returns:
        DataFrame transformé avec la cible conservée.
    """
    X = dataset.drop(columns=[TARGET])
    y = dataset[TARGET].reset_index(drop=True)

    X_transformed = preprocessor.transform(X)

    # Reconstruction d'un DataFrame nommé
    num_features = NUMERIC_FEATURES + [
        "monthly_payment_ratio",
        "risk_score",
        "age_group",
        "log_income",
        "log_loan_amount",
    ]
    feature_names = num_features + BINARY_FEATURES

    result = pd.DataFrame(X_transformed, columns=feature_names)
    result[TARGET] = y.values

    logger.info(
        "Preprocessing appliqué sur %s : %d lignes, %d features",
        split_name,
        len(result),
        result.shape[1] - 1,
    )
    return result


def apply_preprocessing_train(df: pd.DataFrame, preprocessor: Pipeline) -> pd.DataFrame:
    return apply_preprocessing(df, preprocessor, "train")


def apply_preprocessing_val(df: pd.DataFrame, preprocessor: Pipeline) -> pd.DataFrame:
    return apply_preprocessing(df, preprocessor, "val")


def apply_preprocessing_test(df: pd.DataFrame, preprocessor: Pipeline) -> pd.DataFrame:
    return apply_preprocessing(df, preprocessor, "test")
