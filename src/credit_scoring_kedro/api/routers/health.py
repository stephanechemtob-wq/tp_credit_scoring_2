"""Router de santé de l'API."""

import logging

from fastapi import APIRouter, Request

from credit_scoring_kedro.api.public.health.models import Health, Status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "/",
    response_model=Health,
    summary="Vérification de l'état de l'API",
    description="Endpoint public — aucune authentification requise.",
)
def health_check(request: Request) -> Health:
    """Retourne l'état de santé de l'API et du modèle chargé."""
    model_loaded = hasattr(request.app.state, "model") and request.app.state.model is not None
    model_version = getattr(request.app.state, "model_version", "unknown")
    return Health(
        status=Status.HEALTHY if model_loaded else Status.DEGRADED,
        model_loaded=model_loaded,
        model_version=model_version,
        environment="production",
    )
