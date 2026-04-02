"""Module de configuration centralisée — Credit Scoring MLOps.

Pattern Settings centralisé :
- Toutes les variables d'environnement sont déclarées ici avec leurs valeurs par défaut.
- Le fichier .env est chargé automatiquement au démarrage (local).
- En CI/CD et en production, les variables sont injectées via les secrets GitHub / Render.
- Aucun os.getenv() éparpillé dans le code : tout passe par `settings`.

Usage :
    from credit_scoring_kedro.config import settings

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(settings.MODEL_REGISTRY_URI)
"""

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

# ── Chargement automatique du .env (local uniquement) ────────────────────────
try:
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path)
        logger.info("Variables d'environnement chargées depuis : %s", dotenv_path)
    else:
        logger.debug(
            "Aucun fichier .env trouvé — les variables doivent être définies "
            "dans l'environnement (CI/CD, Docker, Render)."
        )
except ImportError:
    logger.warning("python-dotenv non installé. Installez-le avec : uv add python-dotenv")


# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
class Settings:
    """Gestion centralisée des paramètres de l'application.

    Les valeurs sont lues depuis les variables d'environnement.
    Les valeurs par défaut correspondent à un environnement local.
    """

    # ── Environnement ─────────────────────────────────────────────────────────
    ENV: Literal["local", "dev", "prod"] = "local"
    PROJECT_NAME: str = "Credit Scoring API"
    VERSION: str = "2.0.0"

    # ── MLflow / DagsHub ──────────────────────────────────────────────────────
    # URI du serveur MLflow (local par défaut, DagsHub en CI/CD et prod)
    MLFLOW_TRACKING_URI: str = "mlruns"
    MLFLOW_TRACKING_USERNAME: str | None = None
    MLFLOW_TRACKING_PASSWORD: str | None = None

    # Nom du modèle dans le Model Registry et stage à charger
    MLFLOW_MODEL_NAME: str = "credit-scoring-model"
    MLFLOW_MODEL_STAGE: str = "Production"

    # ── API Security ──────────────────────────────────────────────────────────
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Chemins des artefacts locaux (fallback si MLflow indisponible) ─────────
    API_DATA_DIR: str = "data/api"
    PREPROCESSOR_PATH: str = "data/api/preprocessor.pkl"
    MODEL_METADATA_PATH: str = "data/api/model_metadata.json"

    @property
    def MODEL_REGISTRY_URI(self) -> str:
        """URI complète du modèle dans le Model Registry MLflow."""
        return f"models:/{self.MLFLOW_MODEL_NAME}/{self.MLFLOW_MODEL_STAGE}"

    @property
    def IS_REMOTE_MLFLOW(self) -> bool:
        """True si MLflow pointe vers un serveur distant (pas mlruns local)."""
        return self.MLFLOW_TRACKING_URI not in (
            "mlruns",
            "",
        ) and not self.MLFLOW_TRACKING_URI.startswith("file://")

    @classmethod
    def update_from_env(cls) -> None:
        """Lit les variables d'environnement et met à jour les attributs de la classe.

        Seuls les attributs en MAJUSCULES (hors dunder) sont mis à jour.
        Les propriétés (@property) sont ignorées.
        """
        for var in dir(cls):
            if not var.isupper() or var.startswith("__"):
                continue
            # Ignorer les propriétés (elles sont calculées, pas stockées)
            if isinstance(getattr(cls, var, None), property):
                continue
            env_val = os.getenv(var)
            if env_val is not None:
                # Cast vers le type annoté si possible
                annotation = cls.__annotations__.get(var)
                try:
                    if annotation is int:
                        setattr(cls, var, int(env_val))
                    elif annotation is float:
                        setattr(cls, var, float(env_val))
                    elif annotation is bool:
                        setattr(cls, var, env_val.lower() in ("true", "1", "yes"))
                    else:
                        setattr(cls, var, env_val)
                except (ValueError, TypeError):
                    setattr(cls, var, env_val)


# ── Instance globale ──────────────────────────────────────────────────────────
settings = Settings()
settings.update_from_env()

logger.debug(
    "Settings chargés — ENV=%s | MLFLOW_TRACKING_URI=%s | MODEL=%s/%s",
    settings.ENV,
    settings.MLFLOW_TRACKING_URI,
    settings.MLFLOW_MODEL_NAME,
    settings.MLFLOW_MODEL_STAGE,
)
