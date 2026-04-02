"""Modèles Pydantic du domaine santé de l'API.

Attributes:
    Status: Énumération des états possibles de l'API.
    Health: Modèle de réponse du endpoint /health.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Status(str, Enum):
    """Énumération des états possibles de l'API."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"


class Health(BaseModel):
    """Statut de santé de l'API.

    Attributes:
        status: État de l'API (healthy / degraded).
        model_loaded: Modèle ML chargé en mémoire.
        model_version: Version du modèle actif.
        environment: Environnement de déploiement.
    """

    model_config = {"protected_namespaces": ()}

    status: Status = Field(..., description="État de l'API")
    model_loaded: bool = Field(..., description="Modèle ML chargé en mémoire")
    model_version: str = Field(..., description="Version du modèle actif")
    environment: Literal["local", "dev", "production"] = Field(
        ...,
        description="Environnement de déploiement",
    )
