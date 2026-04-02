"""Nodes du pipeline data_processing — Scoring de Crédit MLOps avec Kedro.

Ce module illustre le principe Kedro : chaque node est une fonction Python pure,
sans effet de bord, testable indépendamment et portable sur n'importe quel cloud.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_credit_dataset(params: dict) -> pd.DataFrame:
    """Génère un dataset synthétique de scoring de crédit.

    Principe MLOps : la génération de données est reproductible grâce
    au seed aléatoire versionné dans les paramètres Kedro.

    Args:
        params: Paramètres issus de conf/base/parameters/data_processing.yml

    Returns:
        DataFrame brut avec les features de crédit et la cible (default).
    """
    np.random.seed(params["random_seed"])
    n_samples = params["n_samples"]

    logger.info("Génération du dataset de crédit : %d échantillons", n_samples)

    age = np.random.randint(22, 65, n_samples)
    income = np.random.lognormal(10.8, 0.5, n_samples).astype(int)
    loan_amount = np.random.lognormal(9.5, 0.6, n_samples).astype(int)
    loan_duration_months = np.random.choice([12, 24, 36, 48, 60, 84], n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    num_credit_lines = np.random.randint(1, 15, n_samples)
    employment_years = np.random.randint(0, 30, n_samples)
    debt_to_income_ratio = np.random.uniform(0.05, 0.65, n_samples)
    num_late_payments = np.random.randint(0, 10, n_samples)
    has_mortgage = np.random.randint(0, 2, n_samples)

    # Probabilité de défaut basée sur des règles métier réalistes
    default_prob = (
        0.05
        + 0.3 * (credit_score < 500).astype(float)
        + 0.2 * (debt_to_income_ratio > 0.45).astype(float)
        + 0.15 * (num_late_payments > 3).astype(float)
        + 0.1 * (employment_years < 2).astype(float)
        - 0.1 * (income > 60000).astype(float)
    )
    default_prob = np.clip(default_prob, 0.02, 0.95)
    default = (np.random.uniform(0, 1, n_samples) < default_prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "loan_duration_months": loan_duration_months,
            "credit_score": credit_score,
            "num_credit_lines": num_credit_lines,
            "employment_years": employment_years,
            "debt_to_income_ratio": debt_to_income_ratio,
            "num_late_payments": num_late_payments,
            "has_mortgage": has_mortgage,
            "default": default,
        }
    )

    logger.info(
        "Dataset généré : %d lignes, taux de défaut = %.2f%%", len(df), df["default"].mean() * 100
    )
    return df


def validate_and_clean_data(raw_data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Valide et nettoie les données brutes.

    Principe MLOps : la validation est une étape explicite du pipeline,
    traçable dans MLflow via les métriques de qualité.

    Args:
        raw_data: DataFrame brut issu de generate_credit_dataset.
        params: Paramètres de validation (seuils, colonnes attendues).

    Returns:
        DataFrame nettoyé et validé.
    """
    logger.info("Validation des données : %d lignes initiales", len(raw_data))

    # Vérification du schéma
    expected_cols = params["expected_columns"]
    missing_cols = set(expected_cols) - set(raw_data.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes : {missing_cols}")

    # Suppression des doublons
    n_before = len(raw_data)
    cleaned = raw_data.drop_duplicates()
    n_duplicates = n_before - len(cleaned)
    if n_duplicates > 0:
        logger.warning("Suppression de %d doublons", n_duplicates)

    # Suppression des valeurs nulles
    cleaned = cleaned.dropna()

    # Validation des plages de valeurs
    assert cleaned["credit_score"].between(300, 850).all(), "credit_score hors plage"
    assert cleaned["debt_to_income_ratio"].between(0, 1).all(), "debt_to_income_ratio hors plage"
    assert cleaned["default"].isin([0, 1]).all(), "Cible binaire invalide"

    logger.info(
        "Validation OK : %d lignes retenues (%.1f%% conservées)",
        len(cleaned),
        len(cleaned) / n_before * 100,
    )
    return cleaned


def split_data(
    cleaned_data: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe les données en ensembles train / validation / test.

    Args:
        cleaned_data: DataFrame nettoyé.
        params: Ratios de découpage (train_ratio, val_ratio).

    Returns:
        Tuple (train_set, val_set, test_set).
    """
    from sklearn.model_selection import train_test_split

    train_ratio = params["train_ratio"]
    val_ratio = params["val_ratio"]
    random_seed = params["random_seed"]

    train_val, test = train_test_split(
        cleaned_data,
        test_size=1 - train_ratio - val_ratio,
        random_state=random_seed,
        stratify=cleaned_data["default"],
    )
    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=random_seed, stratify=train_val["default"]
    )

    logger.info("Split : Train=%d | Val=%d | Test=%d", len(train), len(val), len(test))
    return train, val, test
