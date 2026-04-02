"""Fonctions de feature engineering pour les field_validators Pydantic.

Ce module externalise la logique de transformation des champs bruts
en features dérivées, suivant le pattern du projet de référence
(utils/post_processing.py dans InvoicesPretPourProd).

Les fonctions ici sont les mêmes que dans
`pipelines/feature_engineering/nodes.py::add_derived_features()`.
Cette duplication intentionnelle garantit que le contrat API
reste indépendant du pipeline Kedro — seule la logique métier
est partagée, pas les dépendances.

Utilisation dans models.py :
    >>> from credit_scoring_kedro.api.utils.feature_engineering import (
    ...     compute_monthly_payment_ratio,
    ...     compute_risk_score,
    ...     compute_age_group,
    ... )
    >>> _monthly_payment_ratio_validator = field_validator(
    ...     "monthly_payment_ratio", mode="before"
    ... )(compute_monthly_payment_ratio)
"""

from __future__ import annotations

import math


def normalize_loan_term(value: float | None) -> float:
    """Renomme loan_term → loan_duration_months (alias métier → nom interne Kedro).

    Args:
        value: Durée du prêt en mois.

    Returns:
        La valeur inchangée (renommage sémantique uniquement).
    """
    if value is None:
        return 0.0
    return float(value)


def normalize_debt_to_income(value: float | None) -> float:
    """Renomme debt_to_income → debt_to_income_ratio (alias métier → nom interne Kedro).

    Args:
        value: Ratio dette/revenu (0–1).

    Returns:
        La valeur inchangée (renommage sémantique uniquement).
    """
    if value is None:
        return 0.0
    return float(value)


def compute_monthly_payment_ratio(
    income: float, loan_amount: float, loan_duration_months: float
) -> float:
    """Calcule le ratio mensualité / revenu mensuel.

    Reproduit exactement la logique de `add_derived_features()` dans
    `pipelines/feature_engineering/nodes.py`.

    Args:
        income: Revenu annuel (€).
        loan_amount: Montant du prêt (€).
        loan_duration_months: Durée du prêt (mois).

    Returns:
        Ratio mensualité/revenu mensuel, plafonné à 5.0.
    """
    monthly_income = income / 12.0
    monthly_payment = loan_amount / loan_duration_months
    return min(monthly_payment / monthly_income, 5.0)


def compute_risk_score(
    credit_score: float,
    debt_to_income_ratio: float,
    num_late_payments: float,
) -> float:
    """Calcule le score de risque composite (feature métier).

    Reproduit exactement la logique de `add_derived_features()` dans
    `pipelines/feature_engineering/nodes.py`.

    Args:
        credit_score: Score FICO (300–850).
        debt_to_income_ratio: Ratio dette/revenu (0–1).
        num_late_payments: Nombre de paiements en retard.

    Returns:
        Score de risque composite (0.0–1.0 environ).
    """
    return (
        (850.0 - credit_score) / 550.0 * 0.4
        + debt_to_income_ratio * 0.3
        + min(num_late_payments / 10.0, 1.0) * 0.3
    )


def compute_age_group(age: float) -> int:
    """Encode la tranche d'âge en valeur ordinale (0–3).

    Reproduit exactement le pd.cut() de `add_derived_features()` :
        bins=[0, 30, 40, 50, 100], labels=[0, 1, 2, 3]

    Args:
        age: Âge du demandeur (années).

    Returns:
        Tranche d'âge : 0 (≤30), 1 (31–40), 2 (41–50), 3 (>50).
    """
    if age <= 30:
        return 0
    if age <= 40:
        return 1
    if age <= 50:
        return 2
    return 3


def compute_log_income(income: float) -> float:
    """Applique log1p sur le revenu pour réduire l'asymétrie de distribution.

    Args:
        income: Revenu annuel (€).

    Returns:
        log(1 + income).
    """
    return math.log1p(income)


def compute_log_loan_amount(loan_amount: float) -> float:
    """Applique log1p sur le montant du prêt pour réduire l'asymétrie de distribution.

    Args:
        loan_amount: Montant du prêt (€).

    Returns:
        log(1 + loan_amount).
    """
    return math.log1p(loan_amount)
