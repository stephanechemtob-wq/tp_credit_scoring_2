"""Utilitaires pour charger le contexte Kedro dans l'API FastAPI.

Pattern inspiré du projet InvoicesPretPourProd :
- Le contexte Kedro est chargé UNE SEULE FOIS au démarrage (lru_cache).
- Les datasets sont lus depuis le Data Catalog Kedro (env='api').
- L'environnement 'api' (conf/api/) surcharge conf/base/ pour le serving.

Note sur les hooks :
    Le chargement des variables d'environnement (.env) et l'export des
    artefacts ML (data/api/) sont gérés automatiquement par les hooks
    définis dans hooks.py (LoadEnvVarsHook, ArtifactsExportHook).
    Ce module se concentre uniquement sur le chargement du catalog Kedro.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from kedro.framework.context import KedroContext
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

logger = logging.getLogger(__name__)

# Résolution du project_path :
# 1. Variable d'environnement KEDRO_PROJECT_ROOT (priorité — utilisée en Docker)
# 2. Remontée de 4 niveaux depuis ce fichier (mode editable / développement local)
#
# En mode non-editable (Docker), le fichier est dans :
#   /app/.venv/lib/python3.12/site-packages/credit_scoring_kedro/api/utils/session.py
# La remontée de 4 niveaux donnerait /app/.venv/lib/python3.12 — incorrect.
# La variable KEDRO_PROJECT_ROOT=/app est donc nécessaire en Docker.
_PROJECT_ROOT = Path(os.environ.get("KEDRO_PROJECT_ROOT", str(Path(__file__).resolve().parents[4])))


@lru_cache(maxsize=1)
def get_context() -> KedroContext:
    """Charge et met en cache le contexte Kedro (env='api').

    Utilise l'environnement 'api' (conf/api/catalog.yml) qui expose
    les artefacts ML produits par le pipeline et exportés dans data/api/
    via le hook ArtifactsExportHook (hooks.py).

    Returns:
        KedroContext: Contexte Kedro initialisé avec accès au catalog.

    Raises:
        RuntimeError: Si le projet Kedro ne peut pas être initialisé
                      (pyproject.toml introuvable, env 'api' manquant, etc.)

    """
    logger.info("Initialisation du contexte Kedro depuis : %s", _PROJECT_ROOT)
    bootstrap_project(_PROJECT_ROOT)
    with KedroSession.create(project_path=_PROJECT_ROOT, env="api") as session:
        return session.load_context()


@lru_cache(maxsize=32)
def load_dataset(dataset: str) -> Any:  # noqa: ANN401
    """Charge un dataset depuis le Data Catalog Kedro (mis en cache).

    Les datasets disponibles dans l'env 'api' sont définis dans
    conf/api/catalog.yml :
        - api.model          : GradientBoostingClassifier sérialisé (.pkl)
        - api.preprocessor   : Pipeline de preprocessing sérialisé (.pkl)
        - api.model_metadata : Métadonnées du modèle (version, métriques, features)

    Args:
        dataset: Nom du dataset tel que défini dans catalog.yml.

    Returns:
        Any: Contenu du dataset chargé.

    """
    context = get_context()
    logger.info("Chargement du dataset Kedro : %s", dataset)
    return context.catalog.load(dataset)
