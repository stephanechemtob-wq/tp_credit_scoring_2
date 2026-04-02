"""Modèles Pydantic du domaine prédiction.

Ce module définit le contrat de données entre l'API FastAPI et le
pipeline Kedro pour les endpoints de scoring de crédit.

Architecture (pattern InvoicesPretPourProd) :
    - `CreditRequest`  : entrée de l'API, avec field_validators qui
                         calculent les features dérivées à la validation.
    - `PredictionResult` : sortie unitaire.
    - `BatchCreditRequest` / `BatchPredictionResult` : entrée/sortie batch.

Les field_validators délèguent à `api/utils/feature_engineering.py`,
exactement comme `Facture` délègue à `utils/post_processing.py`
dans le projet de référence.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from credit_scoring_kedro.api.utils.feature_engineering import (
    compute_age_group,
    compute_log_income,
    compute_log_loan_amount,
    compute_monthly_payment_ratio,
    compute_risk_score,
)

# Ordre exact attendu par le ColumnTransformer du preprocesseur Kedro
# IMPORTANT : l'ordre doit correspondre exactement à preprocessor.feature_names_in_
# Vérifiable avec : preprocessor.feature_names_in_.tolist()
PREPROCESSOR_FEATURE_ORDER: list[str] = [
    "age",
    "income",
    "loan_amount",
    "loan_duration_months",
    "credit_score",
    "num_credit_lines",
    "employment_years",
    "debt_to_income_ratio",
    "num_late_payments",
    "has_mortgage",  # position 10 — avant monthly_payment_ratio
    "monthly_payment_ratio",
    "risk_score",
    "age_group",
    "log_income",
    "log_loan_amount",
]


class CreditRequest(BaseModel):
    """Dossier de crédit — contrat entre l'API et le pipeline ML.

    Les champs bruts correspondent au vocabulaire métier (ce que
    l'utilisateur connaît). Les features dérivées sont calculées
    automatiquement par les field_validators, en déléguant à
    `api/utils/feature_engineering.py`.

    Attributes:
        age: Âge du demandeur (années).
        income: Revenu annuel (€).
        loan_amount: Montant du prêt demandé (€).
        loan_term: Durée du prêt (mois) — alias de loan_duration_months.
        credit_score: Score de crédit FICO (300–850).
        employment_years: Ancienneté professionnelle (années).
        debt_to_income: Ratio dette/revenu (0–1) — alias de debt_to_income_ratio.
        num_credit_lines: Nombre de lignes de crédit ouvertes.
        num_late_payments: Paiements en retard (12 derniers mois).
        has_mortgage: Possède un crédit immobilier (0/1).
        has_dependents: A des personnes à charge (0/1).
        loan_purpose: Objet du prêt.
        education_level: Niveau d'études.
        employment_type: Type d'emploi.
    """

    # ── Champs bruts (exposés à l'API) ──────────────────────────────────
    age: float = Field(..., ge=18, le=100, description="Âge du demandeur (années)")
    income: float = Field(..., ge=0, description="Revenu annuel (€)")
    loan_amount: float = Field(..., ge=0, description="Montant du prêt demandé (€)")
    loan_term: float = Field(..., ge=1, le=360, description="Durée du prêt (mois)")
    credit_score: float = Field(..., ge=300, le=850, description="Score de crédit FICO (300–850)")
    employment_years: float = Field(..., ge=0, description="Ancienneté professionnelle (années)")
    debt_to_income: float = Field(..., ge=0, le=1, description="Ratio dette/revenu (0–1)")
    num_credit_lines: float = Field(..., ge=0, description="Nombre de lignes de crédit ouvertes")
    num_late_payments: float = Field(
        ..., ge=0, description="Paiements en retard (12 derniers mois)"
    )
    has_mortgage: float = Field(..., ge=0, le=1, description="Crédit immobilier en cours (0/1)")
    has_dependents: float = Field(..., ge=0, le=1, description="Personnes à charge (0/1)")
    loan_purpose: Literal["personal", "auto", "business", "education", "home_improvement"] = Field(
        ...,
        description="Objet du prêt",
    )
    education_level: Literal["high_school", "bachelor", "master", "phd"] = Field(
        ...,
        description="Niveau d'études",
    )
    employment_type: Literal["employed", "self_employed", "unemployed", "retired"] = Field(
        ...,
        description="Type d'emploi",
    )

    # ── Features dérivées (calculées par le model_validator) ────────────
    # Initialisées à 0.0 ; remplies après validation des champs bruts.
    loan_duration_months: float = Field(default=0.0, description="Alias interne de loan_term")
    debt_to_income_ratio: float = Field(default=0.0, description="Alias interne de debt_to_income")
    monthly_payment_ratio: float = Field(default=0.0, description="Mensualité / revenu mensuel")
    risk_score: float = Field(default=0.0, description="Score de risque composite")
    age_group: int = Field(default=0, description="Tranche d'âge ordinale (0–3)")
    log_income: float = Field(default=0.0, description="log1p(income)")
    log_loan_amount: float = Field(default=0.0, description="log1p(loan_amount)")

    model_config = {"protected_namespaces": ()}

    @model_validator(mode="after")
    def compute_derived_features(self) -> CreditRequest:
        """Calcule toutes les features dérivées après validation des champs bruts.

        Délègue à `api/utils/feature_engineering.py` — exactement comme
        `Facture` délègue à `utils/post_processing.py` dans le projet
        de référence InvoicesPretPourProd.

        Returns:
            L'instance enrichie avec toutes les features dérivées.
        """
        # Renommage alias métier → noms internes Kedro
        self.loan_duration_months = self.loan_term
        self.debt_to_income_ratio = self.debt_to_income

        # Délégation aux fonctions de feature_engineering.py
        self.monthly_payment_ratio = compute_monthly_payment_ratio(
            self.income,
            self.loan_amount,
            self.loan_duration_months,
        )
        self.risk_score = compute_risk_score(
            self.credit_score,
            self.debt_to_income_ratio,
            self.num_late_payments,
        )
        self.age_group = compute_age_group(self.age)
        self.log_income = compute_log_income(self.income)
        self.log_loan_amount = compute_log_loan_amount(self.loan_amount)

        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Produit un DataFrame d'une ligne dans l'ordre exact attendu par le preprocesseur Kedro.

        Returns:
            pd.DataFrame avec les colonnes dans PREPROCESSOR_FEATURE_ORDER.
        """
        row = {
            "age": self.age,
            "income": self.income,
            "loan_amount": self.loan_amount,
            "loan_duration_months": self.loan_duration_months,
            "credit_score": self.credit_score,
            "num_credit_lines": self.num_credit_lines,
            "employment_years": self.employment_years,
            "debt_to_income_ratio": self.debt_to_income_ratio,
            "num_late_payments": self.num_late_payments,
            "has_mortgage": self.has_mortgage,  # position 10 — avant monthly_payment_ratio
            "monthly_payment_ratio": self.monthly_payment_ratio,
            "risk_score": self.risk_score,
            "age_group": float(self.age_group),
            "log_income": self.log_income,
            "log_loan_amount": self.log_loan_amount,
        }
        return pd.DataFrame([row], columns=PREPROCESSOR_FEATURE_ORDER)


class PredictionResult(BaseModel):
    """Réponse de prédiction du risque de défaut de paiement.

    Attributes:
        prediction: 1 = défaut probable, 0 = solvable.
        probability_of_default: Probabilité de défaut (0.0–1.0).
        risk_level: Niveau de risque catégoriel.
        risk_score: Score normalisé (0–100).
        recommendation: Décision recommandée.
        model_version: Version du modèle utilisé.
        latency_ms: Latence de la prédiction (ms).
    """

    model_config = {"protected_namespaces": ()}

    prediction: int = Field(..., description="1 = défaut probable, 0 = solvable")
    probability_of_default: float = Field(..., description="Probabilité de défaut (0.0–1.0)")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        ..., description="Niveau de risque"
    )
    risk_score: int = Field(..., description="Score de risque normalisé (0–100)")
    recommendation: Literal["APPROVE", "REVIEW", "REJECT"] = Field(
        ..., description="Recommandation"
    )
    model_version: str = Field(..., description="Version du modèle utilisé")
    latency_ms: float = Field(..., description="Latence de la prédiction (ms)")


class BatchCreditRequest(BaseModel):
    """Requête de prédiction en batch (1 à 500 dossiers).

    Attributes:
        records: Liste des dossiers de crédit à scorer.
    """

    records: list[CreditRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Liste des dossiers de crédit à scorer",
    )


class BatchPredictionResult(BaseModel):
    """Réponse de prédiction en batch avec statistiques agrégées.

    Attributes:
        results: Liste des prédictions individuelles.
        total: Nombre total de dossiers traités.
        approved: Nombre de dossiers approuvés.
        rejected: Nombre de dossiers rejetés.
        review: Nombre de dossiers à examiner.
        batch_latency_ms: Latence totale du batch (ms).
    """

    results: list[PredictionResult]
    total: int = Field(..., description="Nombre total de dossiers traités")
    approved: int = Field(..., description="Dossiers approuvés")
    rejected: int = Field(..., description="Dossiers rejetés")
    review: int = Field(..., description="Dossiers à examiner")
    batch_latency_ms: float = Field(..., description="Latence totale du batch (ms)")
