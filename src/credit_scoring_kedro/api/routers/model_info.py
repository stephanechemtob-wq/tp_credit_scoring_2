"""Router d'informations sur le modèle en production."""

import logging

from fastapi import APIRouter, Depends, Request

from credit_scoring_kedro.api.public.model_info.models import ModelMetadata
from credit_scoring_kedro.api.security.auth import require_scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Modèle"])


@router.get(
    "/info",
    response_model=ModelMetadata,
    summary="Informations sur le modèle en production",
    description=(
        "Retourne les métadonnées du modèle chargé : algorithme, features, "
        "métriques, version.\n\n"
        "**Authentification requise** : scope `model:read`."
    ),
)
def model_info(
    request: Request,
    _current_user: dict = Depends(require_scope("model:read")),
) -> ModelMetadata:
    """Retourne les informations du modèle actuellement en production."""
    state = request.app.state
    return ModelMetadata(
        model_name=getattr(state, "model_name", "credit_scoring_model"),
        model_version=getattr(state, "model_version", "unknown"),
        algorithm=getattr(state, "algorithm", "GradientBoostingClassifier"),
        features=getattr(state, "feature_names", []),
        metrics=getattr(state, "metrics", {}),
        training_date=getattr(state, "training_date", "unknown"),
        threshold=getattr(state, "threshold", 0.5),
    )
