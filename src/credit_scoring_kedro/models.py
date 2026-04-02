"""
Contrat de données entre le pipeline Kedro et l'API FastAPI.

Ce fichier est la **source de vérité unique** pour la représentation
d'un dossier de crédit dans tout le projet. Il définit :

  - Les champs bruts exposés à l'API (ce que l'utilisateur envoie)
  - Les features dérivées calculées automatiquement via `model_validator`
    (même logique que `feature_engineering/nodes.py`)
  - La méthode `to_dataframe()` qui produit le DataFrame attendu
    par le preprocesseur Kedro

Principe MLOps :
  Le `model_validator` garantit que la transformation
  brut → features dérivées est identique à l'entraînement et à
  l'inférence — sans duplication de code, sans risque de dérive.

Utilisation :
    >>> req = CreditRequest(age=35, income=55000, ...)
    >>> df  = req.to_dataframe()          # prêt pour preprocessor.transform()
    >>> X   = preprocessor.transform(df)
    >>> proba = model.predict_proba(X)[0][1]
"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

# ── Constantes partagées avec feature_engineering/nodes.py ────────────────
# Ordre exact attendu par le ColumnTransformer du preprocesseur Kedro
PREPROCESSOR_FEATURE_ORDER: list[str] = [
    # Numériques (StandardScaler)
    "age",
    "income",
    "loan_amount",
    "loan_duration_months",
    "credit_score",
    "num_credit_lines",
    "employment_years",
    "debt_to_income_ratio",
    "num_late_payments",
    "monthly_payment_ratio",
    "risk_score",
    "age_group",
    "log_income",
    "log_loan_amount",
    # Binaire (passthrough)
    "has_mortgage",
]


# ── Modèle principal ───────────────────────────────────────────────────────


class CreditRequest(BaseModel):
    """
    Dossier de crédit — contrat entre l'API et le pipeline ML.

    Champs bruts (saisis par l'utilisateur) :
      Les noms correspondent aux données métier, pas aux noms internes
      du pipeline (ex. `loan_term` plutôt que `loan_duration_months`).

    Champs dérivés (calculés automatiquement) :
      Produits par le `model_validator` après validation des champs bruts.
      Ils reproduisent exactement la logique de `add_derived_features()`
      dans `feature_engineering/nodes.py`.
    """

    # ── Champs bruts ────────────────────────────────────────────────────
    age: float = Field(..., ge=18, le=100, description="Âge du demandeur (années)")
    income: float = Field(..., ge=0, description="Revenu annuel (€)")
    loan_amount: float = Field(..., ge=0, description="Montant du prêt demandé (€)")
    loan_term: float = Field(
        ..., ge=1, le=360, description="Durée du prêt (mois) — alias de loan_duration_months"
    )
    credit_score: float = Field(..., ge=300, le=850, description="Score de crédit FICO (300–850)")
    employment_years: float = Field(..., ge=0, description="Ancienneté professionnelle (années)")
    debt_to_income: float = Field(
        ..., ge=0, le=1, description="Ratio dette/revenu (0–1) — alias de debt_to_income_ratio"
    )
    num_credit_lines: float = Field(..., ge=0, description="Nombre de lignes de crédit ouvertes")
    num_late_payments: float = Field(
        ..., ge=0, description="Paiements en retard (12 derniers mois)"
    )
    has_mortgage: float = Field(
        ..., ge=0, le=1, description="Possède un crédit immobilier (0 = non, 1 = oui)"
    )
    has_dependents: float = Field(
        ..., ge=0, le=1, description="A des personnes à charge (0 = non, 1 = oui)"
    )
    loan_purpose: Literal["personal", "auto", "business", "education", "home_improvement"] = Field(
        ..., description="Objet du prêt"
    )
    education_level: Literal["high_school", "bachelor", "master", "phd"] = Field(
        ..., description="Niveau d'études"
    )
    employment_type: Literal["employed", "self_employed", "unemployed", "retired"] = Field(
        ..., description="Type d'emploi"
    )

    # ── Champs dérivés (calculés par le validator) ───────────────────────
    # Déclarés avec default=None pour que Pydantic les accepte avant validation
    loan_duration_months: float = Field(
        default=0.0, exclude=True, description="Alias interne de loan_term"
    )
    debt_to_income_ratio: float = Field(
        default=0.0, exclude=True, description="Alias interne de debt_to_income"
    )
    monthly_payment_ratio: float = Field(
        default=0.0, exclude=True, description="Mensualité / revenu mensuel (feature dérivée)"
    )
    risk_score_derived: float = Field(
        default=0.0, exclude=True, description="Score de risque composite (feature dérivée)"
    )
    age_group: int = Field(
        default=0, exclude=True, description="Tranche d'âge ordinale 0–3 (feature dérivée)"
    )
    log_income: float = Field(
        default=0.0, exclude=True, description="log1p(income) — réduction asymétrie"
    )
    log_loan_amount: float = Field(
        default=0.0, exclude=True, description="log1p(loan_amount) — réduction asymétrie"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Profil solvable (LOW risk → APPROVE)",
                    "value": {
                        "age": 45.0,
                        "income": 85000.0,
                        "loan_amount": 15000.0,
                        "loan_term": 36.0,
                        "credit_score": 750.0,
                        "employment_years": 10.0,
                        "debt_to_income": 0.25,
                        "num_credit_lines": 3.0,
                        "num_late_payments": 0.0,
                        "has_mortgage": 1.0,
                        "has_dependents": 1.0,
                        "loan_purpose": "home_improvement",
                        "education_level": "master",
                        "employment_type": "employed",
                    },
                },
                {
                    "title": "Profil risqué (CRITICAL risk → REJECT)",
                    "value": {
                        "age": 22.0,
                        "income": 25000.0,
                        "loan_amount": 35000.0,
                        "loan_term": 60.0,
                        "credit_score": 520.0,
                        "employment_years": 1.0,
                        "debt_to_income": 0.65,
                        "num_credit_lines": 6.0,
                        "num_late_payments": 3.0,
                        "has_mortgage": 0.0,
                        "has_dependents": 0.0,
                        "loan_purpose": "personal",
                        "education_level": "high_school",
                        "employment_type": "unemployed",
                    },
                },
            ]
        }
    }

    # ── Validator : calcul des features dérivées ─────────────────────────

    @model_validator(mode="after")
    def compute_derived_features(self) -> CreditRequest:
        """
        Calcule les features dérivées après validation des champs bruts.

        Reproduit exactement `add_derived_features()` de
        `feature_engineering/nodes.py` — garantissant la cohérence
        entraînement / inférence (pas de training-serving skew).
        """
        # 1. Renommage des alias métier → noms internes Kedro
        self.loan_duration_months = self.loan_term
        self.debt_to_income_ratio = self.debt_to_income

        # 2. Ratio mensualité / revenu mensuel
        monthly_income = self.income / 12.0
        monthly_payment = self.loan_amount / self.loan_duration_months
        self.monthly_payment_ratio = min(monthly_payment / monthly_income, 5.0)

        # 3. Score de risque composite
        self.risk_score_derived = (
            (850.0 - self.credit_score) / 550.0 * 0.4
            + self.debt_to_income_ratio * 0.3
            + min(self.num_late_payments / 10.0, 1.0) * 0.3
        )

        # 4. Tranche d'âge (encodage ordinal identique à pd.cut)
        if self.age <= 30:
            self.age_group = 0
        elif self.age <= 40:
            self.age_group = 1
        elif self.age <= 50:
            self.age_group = 2
        else:
            self.age_group = 3

        # 5. Transformations logarithmiques
        self.log_income = math.log1p(self.income)
        self.log_loan_amount = math.log1p(self.loan_amount)

        return self

    # ── Méthode de conversion ────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """
        Produit un DataFrame d'une ligne dans l'ordre exact attendu
        par le ColumnTransformer du preprocesseur Kedro.

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
            "monthly_payment_ratio": self.monthly_payment_ratio,
            "risk_score": self.risk_score_derived,
            "age_group": float(self.age_group),
            "log_income": self.log_income,
            "log_loan_amount": self.log_loan_amount,
            "has_mortgage": self.has_mortgage,
        }
        return pd.DataFrame([row], columns=PREPROCESSOR_FEATURE_ORDER)


# ── Modèles de réponse ─────────────────────────────────────────────────────


class PredictionResult(BaseModel):
    """Réponse de prédiction du risque de défaut de paiement."""

    model_config = {"protected_namespaces": ()}

    prediction: int = Field(..., description="1 = défaut probable, 0 = solvable")
    probability_of_default: float = Field(..., description="Probabilité de défaut (0.0 – 1.0)")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        ..., description="Niveau de risque"
    )
    risk_score: int = Field(..., description="Score de risque normalisé (0–100)")
    recommendation: Literal["APPROVE", "REVIEW", "REJECT"] = Field(
        ..., description="Recommandation de décision"
    )
    model_version: str = Field(..., description="Version du modèle utilisé")
    latency_ms: float = Field(..., description="Latence de la prédiction (ms)")


class BatchCreditRequest(BaseModel):
    """Requête de prédiction en batch (1 à 500 dossiers)."""

    records: list[CreditRequest] = Field(
        ..., min_length=1, max_length=500, description="Liste des dossiers de crédit à scorer"
    )


class BatchPredictionResult(BaseModel):
    """Réponse de prédiction en batch avec statistiques agrégées."""

    results: list[PredictionResult]
    total: int = Field(..., description="Nombre total de dossiers traités")
    approved: int = Field(..., description="Nombre de dossiers approuvés")
    rejected: int = Field(..., description="Nombre de dossiers rejetés")
    review: int = Field(..., description="Nombre de dossiers à examiner")
    batch_latency_ms: float = Field(..., description="Latence totale du batch (ms)")


class HealthStatus(BaseModel):
    """Statut de santé de l'API."""

    model_config = {"protected_namespaces": ()}

    status: Literal["healthy", "degraded"] = Field(..., description="État de l'API")
    model_loaded: bool = Field(..., description="Modèle ML chargé en mémoire")
    model_version: str = Field(..., description="Version du modèle actif")
    environment: str = Field(..., description="Environnement de déploiement")


class ModelMetadata(BaseModel):
    """Métadonnées du modèle en production."""

    model_config = {"protected_namespaces": ()}

    model_name: str
    model_version: str
    algorithm: str
    features: list[str] = Field(..., description="Liste des features utilisées par le modèle")
    metrics: dict = Field(..., description="Métriques de performance (ROC-AUC, F1…)")
    training_date: str
    threshold: float = Field(..., description="Seuil de décision (probabilité → classe)")
