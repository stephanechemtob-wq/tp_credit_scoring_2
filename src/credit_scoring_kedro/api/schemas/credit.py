"""Schémas Pydantic pour l'API de scoring de crédit."""

from pydantic import BaseModel, Field


class CreditFeatures(BaseModel):
    """Caractéristiques d'un dossier de crédit pour la prédiction."""

    age: float = Field(..., ge=18, le=100, description="Âge du demandeur (années)")
    income: float = Field(..., ge=0, description="Revenu annuel (€)")
    loan_amount: float = Field(..., ge=0, description="Montant du prêt demandé (€)")
    loan_term: float = Field(..., ge=1, le=360, description="Durée du prêt (mois)")
    credit_score: float = Field(..., ge=300, le=850, description="Score de crédit (300-850)")
    employment_years: float = Field(..., ge=0, description="Ancienneté professionnelle (années)")
    debt_to_income: float = Field(..., ge=0, le=1, description="Ratio dette/revenu (0-1)")
    num_credit_lines: float = Field(..., ge=0, description="Nombre de lignes de crédit ouvertes")
    num_late_payments: float = Field(
        ..., ge=0, description="Nombre de paiements en retard (12 derniers mois)"
    )
    has_mortgage: float = Field(..., ge=0, le=1, description="Possède un crédit immobilier (0/1)")
    has_dependents: float = Field(..., ge=0, le=1, description="A des personnes à charge (0/1)")
    loan_purpose: str = Field(
        ...,
        description="Objet du prêt : 'personal', 'auto', 'business', 'education', 'home_improvement'",
    )
    education_level: str = Field(
        ..., description="Niveau d'études : 'high_school', 'bachelor', 'master', 'phd'"
    )
    employment_type: str = Field(
        ..., description="Type d'emploi : 'employed', 'self_employed', 'unemployed', 'retired'"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35.0,
                "income": 55000.0,
                "loan_amount": 15000.0,
                "loan_term": 36.0,
                "credit_score": 680.0,
                "employment_years": 7.0,
                "debt_to_income": 0.32,
                "num_credit_lines": 4.0,
                "num_late_payments": 0.0,
                "has_mortgage": 1.0,
                "has_dependents": 1.0,
                "loan_purpose": "personal",
                "education_level": "bachelor",
                "employment_type": "employed",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Réponse de prédiction du risque de défaut de paiement."""

    model_config = {"protected_namespaces": ()}

    prediction: int = Field(..., description="Prédiction : 1 = défaut probable, 0 = solvable")
    probability_of_default: float = Field(..., description="Probabilité de défaut (0.0 à 1.0)")
    risk_level: str = Field(..., description="Niveau de risque : LOW / MEDIUM / HIGH / CRITICAL")
    risk_score: int = Field(..., description="Score de risque normalisé (0-100)")
    recommendation: str = Field(..., description="Recommandation : APPROVE / REVIEW / REJECT")
    model_version: str = Field(..., description="Version du modèle utilisé")
    latency_ms: float = Field(..., description="Latence de la prédiction (ms)")


class BatchRequest(BaseModel):
    """Requête de prédiction en batch."""

    records: list[CreditFeatures] = Field(
        ..., min_length=1, max_length=500, description="Liste des dossiers à scorer"
    )


class BatchResponse(BaseModel):
    """Réponse de prédiction en batch."""

    results: list[PredictionResponse]
    total: int
    approved: int
    rejected: int
    review: int
    batch_latency_ms: float


class HealthResponse(BaseModel):
    """Réponse du endpoint de santé."""

    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_version: str
    environment: str


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle en production."""

    model_config = {"protected_namespaces": ()}

    model_name: str
    model_version: str
    algorithm: str
    features: list[str]
    metrics: dict
    training_date: str
    threshold: float
