"""Router de prédiction unitaire et batch.

Suit le pattern views.py du projet de référence InvoicesPretPourProd :
le router ne contient que le câblage HTTP — toute la logique métier
est dans crud.py, et le contrat de données dans
`api/public/predict/models.py`.

Note sur les exemples Swagger UI :
Les exemples sont définis via `openapi_extra` dans le décorateur @router.post,
en utilisant la structure OpenAPI 3.0 `requestBody.content.examples`.
C'est la seule méthode fiable avec FastAPI + Pydantic V2 pour que le bouton
"Try it out → Execute" de Swagger UI envoie le bon payload (sans enveloppe
{summary, value}).
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from credit_scoring_kedro.api.public.predict.models import (
    BatchCreditRequest,
    BatchPredictionResult,
    CreditRequest,
    PredictionResult,
)
from credit_scoring_kedro.api.security.auth import require_scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prédiction"])

# ── Constantes de décision ─────────────────────────────────────────────────
RISK_THRESHOLDS = {"LOW": 0.20, "MEDIUM": 0.45, "HIGH": 0.65, "CRITICAL": 1.0}
RECOMMENDATIONS = {"LOW": "APPROVE", "MEDIUM": "APPROVE", "HIGH": "REVIEW", "CRITICAL": "REJECT"}

# ── Exemples OpenAPI (utilisés dans openapi_extra) ─────────────────────────
# Structure correcte pour Swagger UI : requestBody.content.examples
# Chaque exemple a un "summary" et un "value" contenant le payload direct.
_PREDICT_EXAMPLES = {
    "profil_solvable": {
        "summary": "Profil solvable (LOW risk → APPROVE)",
        "value": {
            "age": 45,
            "income": 85000,
            "loan_amount": 15000,
            "loan_term": 36,
            "credit_score": 750,
            "employment_years": 10,
            "debt_to_income": 0.25,
            "num_credit_lines": 3,
            "num_late_payments": 0,
            "has_mortgage": 1,
            "has_dependents": 1,
            "loan_purpose": "home_improvement",
            "education_level": "master",
            "employment_type": "employed",
        },
    },
    "profil_risque": {
        "summary": "Profil risqué (CRITICAL risk → REJECT)",
        "value": {
            "age": 22,
            "income": 25000,
            "loan_amount": 35000,
            "loan_term": 60,
            "credit_score": 520,
            "employment_years": 1,
            "debt_to_income": 0.65,
            "num_credit_lines": 6,
            "num_late_payments": 3,
            "has_mortgage": 0,
            "has_dependents": 0,
            "loan_purpose": "personal",
            "education_level": "high_school",
            "employment_type": "unemployed",
        },
    },
    "profil_medium": {
        "summary": "Profil intermédiaire (MEDIUM risk → APPROVE)",
        "value": {
            "age": 35,
            "income": 55000,
            "loan_amount": 20000,
            "loan_term": 48,
            "credit_score": 650,
            "employment_years": 5,
            "debt_to_income": 0.40,
            "num_credit_lines": 4,
            "num_late_payments": 1,
            "has_mortgage": 0,
            "has_dependents": 1,
            "loan_purpose": "auto",
            "education_level": "bachelor",
            "employment_type": "employed",
        },
    },
}

_BATCH_EXAMPLES = {
    "batch_deux_dossiers": {
        "summary": "Batch de 2 dossiers (1 solvable + 1 risqué)",
        "value": {
            "records": [
                {
                    "age": 45,
                    "income": 85000,
                    "loan_amount": 15000,
                    "loan_term": 36,
                    "credit_score": 750,
                    "employment_years": 10,
                    "debt_to_income": 0.25,
                    "num_credit_lines": 3,
                    "num_late_payments": 0,
                    "has_mortgage": 1,
                    "has_dependents": 1,
                    "loan_purpose": "home_improvement",
                    "education_level": "master",
                    "employment_type": "employed",
                },
                {
                    "age": 22,
                    "income": 25000,
                    "loan_amount": 35000,
                    "loan_term": 60,
                    "credit_score": 520,
                    "employment_years": 1,
                    "debt_to_income": 0.65,
                    "num_credit_lines": 6,
                    "num_late_payments": 3,
                    "has_mortgage": 0,
                    "has_dependents": 0,
                    "loan_purpose": "personal",
                    "education_level": "high_school",
                    "employment_type": "unemployed",
                },
            ],
        },
    },
}


def _get_risk_level(proba: float) -> str:
    for level, threshold in RISK_THRESHOLDS.items():
        if proba <= threshold:
            return level
    return "CRITICAL"


def _score_one(request_data: CreditRequest, app_request: Request) -> PredictionResult:
    """Logique de prédiction unitaire — délègue à to_dataframe() du contrat.

    Args:
        request_data: Dossier de crédit validé (features dérivées déjà calculées).
        app_request: Requête FastAPI (accès à app.state).

    Returns:
        PredictionResult avec probabilité, niveau de risque et recommandation.
    """
    model = app_request.app.state.model
    preprocessor = app_request.app.state.preprocessor
    model_version = app_request.app.state.model_version
    threshold = app_request.app.state.threshold

    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Relancez le pipeline Kedro.",
        )

    start = time.perf_counter()

    # Le contrat fait tout le travail de transformation
    X = preprocessor.transform(request_data.to_dataframe())
    proba = float(model.predict_proba(X)[0][1])
    risk_level = _get_risk_level(proba)

    return PredictionResult(
        prediction=int(proba >= threshold),
        probability_of_default=round(proba, 4),
        risk_level=risk_level,
        risk_score=int(proba * 100),
        recommendation=RECOMMENDATIONS[risk_level],
        model_version=model_version,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@router.post(
    "/",
    response_model=PredictionResult,
    summary="Prédiction unitaire — risque de défaut de paiement",
    description=(
        "Soumet un dossier de crédit et retourne la probabilité de défaut, "
        "le niveau de risque (LOW/MEDIUM/HIGH/CRITICAL) et la recommandation "
        "(APPROVE/REVIEW/REJECT).\n\n"
        "**Authentification requise** : scope `predict`."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": _PREDICT_EXAMPLES,
                },
            },
        },
    },
)
def predict(
    request_data: CreditRequest,
    app_request: Request,
    _current_user: dict = Depends(require_scope("predict")),
) -> PredictionResult:
    """Prédiction unitaire du risque de défaut de paiement."""
    try:
        return _score_one(request_data, app_request)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur lors de la prédiction : %s", e)
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}") from e


@router.post(
    "/batch",
    response_model=BatchPredictionResult,
    summary="Prédiction en batch (jusqu'à 500 dossiers)",
    description=(
        "Soumet une liste de dossiers de crédit et retourne les prédictions "
        "pour chacun avec un résumé agrégé.\n\n"
        "**Authentification requise** : scope `batch`."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": _BATCH_EXAMPLES,
                },
            },
        },
    },
)
def predict_batch(
    batch: BatchCreditRequest,
    app_request: Request,
    _current_user: dict = Depends(require_scope("batch")),
) -> BatchPredictionResult:
    """Prédiction en batch du risque de défaut de paiement."""
    start = time.perf_counter()
    results = []

    for record in batch.records:
        try:
            results.append(_score_one(record, app_request))
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Erreur sur un enregistrement du batch : %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Erreur sur un enregistrement : {e}",
            ) from e

    return BatchPredictionResult(
        results=results,
        total=len(results),
        approved=sum(1 for r in results if r.recommendation == "APPROVE"),
        rejected=sum(1 for r in results if r.recommendation == "REJECT"),
        review=sum(1 for r in results if r.recommendation == "REVIEW"),
        batch_latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
