"""Factory de l'application FastAPI — Credit Scoring MLOps.

Pattern inspiré de InvoicesPretPourProd :
- Le lifespan charge les artefacts Kedro UNE SEULE FOIS au démarrage.
- Le modèle, le preprocesseur et les métadonnées sont stockés dans app.state.
- L'environnement Kedro 'api' (conf/api/) surcharge conf/base/ pour le serving.
- Toutes les variables d'environnement passent par `settings` (config.py).

Sécurité (selon PDF "Considérations de sécurité pour API") :
- JWT (JSON Web Tokens) via python-jose + passlib
- Rate limiting par IP via slowapi (protection DDoS / brute-force)
- Validation des données via Pydantic (protection injection)
- CORS configuré
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from credit_scoring_kedro.api.middleware.rate_limit import limiter
from credit_scoring_kedro.api.routers import auth, health, model_info, predict
from credit_scoring_kedro.api.utils.session import load_dataset
from credit_scoring_kedro.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Chargement des artefacts au démarrage de l'API.

    Stratégie de chargement du modèle :
    - Si MLflow pointe vers un serveur distant (DagsHub / serveur dédié) :
        → Téléchargement depuis le Model Registry (models:/credit-scoring-model/Production)
    - Sinon (développement local sans serveur MLflow) :
        → Fallback sur le fichier local data/api/model.pkl (si présent)

    Le preprocessor et les métadonnées sont toujours chargés depuis data/api/
    via le Data Catalog Kedro (conf/api/catalog.yml).
    """
    logger.info("Démarrage de l'API — chargement des artefacts...")
    logger.info(
        "MLflow tracking URI : %s | Modèle : %s",
        settings.MLFLOW_TRACKING_URI,
        settings.MODEL_REGISTRY_URI,
    )

    try:
        # ── 1. Configuration MLflow depuis settings ───────────────────────────
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        # ── 2. Chargement du modèle ───────────────────────────────────────────
        if settings.IS_REMOTE_MLFLOW:
            logger.info(
                "Téléchargement du modèle depuis MLflow Registry : %s",
                settings.MODEL_REGISTRY_URI,
            )
            app.state.model = mlflow.sklearn.load_model(settings.MODEL_REGISTRY_URI)
        else:
            # Fallback local : utile en développement sans serveur MLflow
            logger.warning(
                "MLflow local détecté (%s). "
                "Tentative de chargement depuis data/api/model.pkl...",
                settings.MLFLOW_TRACKING_URI,
            )
            app.state.model = load_dataset("api.model")

        # ── 3. Chargement du preprocessor et des métadonnées (via DVC/catalog) ─
        app.state.preprocessor = load_dataset("api.preprocessor")
        metadata = load_dataset("api.model_metadata")

        # ── 4. Métadonnées exposées dans /model/info ──────────────────────────
        app.state.model_version = metadata.get("model_version", "1.0.0")
        app.state.model_name = metadata.get("model_name", "credit_scoring_model")
        app.state.algorithm = metadata.get("algorithm", "GradientBoostingClassifier")
        app.state.feature_names = metadata.get("feature_names", [])
        app.state.metrics = metadata.get("metrics", {})
        app.state.training_date = metadata.get("training_date", "unknown")
        app.state.threshold = float(metadata.get("threshold", 0.5))

        logger.info(
            "Modèle '%s' v%s chargé avec succès (ROC-AUC: %.3f)",
            app.state.model_name,
            app.state.model_version,
            app.state.metrics.get("roc_auc_test", 0.0),
        )

    except Exception as e:
        logger.error("Erreur au chargement des artefacts : %s", e)
        logger.warning("L'API démarre en mode dégradé — lancez d'abord : kedro run")
        app.state.model = None
        app.state.preprocessor = None
        app.state.model_version = "unavailable"
        app.state.model_name = "unavailable"
        app.state.algorithm = "unavailable"
        app.state.feature_names = []
        app.state.metrics = {}
        app.state.training_date = "unavailable"
        app.state.threshold = 0.5

    yield

    # ── Nettoyage au shutdown ─────────────────────────────────────────────────
    logger.info("Arrêt de l'API — libération des ressources.")
    app.state.model = None
    app.state.preprocessor = None


def create_app() -> FastAPI:
    """Initialise et configure l'application FastAPI avec sécurité complète."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "API de scoring de crédit construite avec **FastAPI** et **Kedro**.\n\n"
            "Les artefacts ML (modèle, preprocesseur, métadonnées) sont chargés "
            "depuis le **MLflow Model Registry** (DagsHub) au démarrage, garantissant une "
            "cohérence totale entre le pipeline d'entraînement et le serving.\n\n"
            "## Sécurité\n"
            "- **JWT Bearer** : authentification requise sur `/predict` et `/predict/batch`\n"
            "- **Rate Limiting** : 60 req/min sur `/predict`, 10 req/min sur `/predict/batch`, "
            "5 req/min sur `/auth/token`\n"
            "- **Validation Pydantic** : protection contre les injections\n\n"
            "## Endpoints\n"
            "- `POST /auth/token` — Obtenir un token JWT\n"
            "- `POST /auth/refresh` — Rafraîchir le token\n"
            "- `POST /auth/logout` — Révoquer les tokens\n"
            "- `GET  /auth/me` — Profil utilisateur\n"
            "- `GET  /health` — Santé de l'API (public)\n"
            "- `POST /predict` — Prédiction unitaire (auth requise)\n"
            "- `POST /predict/batch` — Prédiction en batch (auth requise)\n"
            "- `GET  /model/info` — Métadonnées du modèle (auth requise)\n\n"
            "## Utilisateurs de démo\n"
            "| Username | Password | Scopes |\n"
            "|---|---|---|\n"
            "| `data_scientist` | `mlops2024` | predict, batch, model:read |\n"
            "| `admin` | `admin_secret` | tous les scopes |\n"
            "| `readonly` | `readonly123` | model:read uniquement |"
        ),
        docs_url="/",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Rate Limiter (slowapi) ────────────────────────────────────────────────
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        """Gestionnaire personnalisé pour les erreurs 429 Too Many Requests."""
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "detail": f"Trop de requêtes. Limite : {exc.detail}",
                "retry_after": "Réessayez dans quelques secondes.",
            },
        )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restreindre en production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(auth.router)  # /auth/token, /auth/refresh, /auth/logout, /auth/me
    app.include_router(health.router)  # /health (public)
    app.include_router(predict.router)  # /predict, /predict/batch (JWT requis)
    app.include_router(model_info.router)  # /model/info (JWT requis)

    return app


# Instance principale (utilisée par uvicorn)
api = create_app()
