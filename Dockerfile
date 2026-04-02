# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile multi-stage — Credit Scoring API (FastAPI + Kedro)
#
# Usage :
#   # Build
#   docker build -t credit-scoring-api:latest .
#
#   # Lancer l'API (après uv run credit-scoring-kedro)
#   docker run -p 8000:8000 \
#     -v "$(pwd)/data/api:/app/data/api:ro" \
#     credit-scoring-api:latest
#
#   # Accéder à la documentation Swagger
#   http://localhost:8000/docs
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Builder : résolution et installation des dépendances via uv
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS builder

LABEL maintainer="MLOps Team"
LABEL description="Credit Scoring API — FastAPI + Kedro"
LABEL version="5.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT="/app/.venv" \
    UV_FROZEN=1

# Installer uv depuis l'image officielle
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copier les fichiers de résolution en premier (optimise le cache Docker)
COPY pyproject.toml uv.lock ./

# Installer les dépendances sans le package lui-même
RUN uv sync --frozen --no-install-project --no-dev

# Copier le code source et installer le package (non-editable pour la prod)
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Runtime : image de production légère
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    API_HOST="0.0.0.0" \
    API_PORT="8000" \
    LOG_LEVEL="info" \
    KEDRO_ENV="api" \
    KEDRO_PROJECT_ROOT="/app"

# Créer un utilisateur non-root pour la sécurité
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copier le venv complet (avec le package installé) depuis le builder
# --chown garantit que appuser peut lire le venv sans problème de permissions
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Copier pyproject.toml — requis par Kedro pour localiser le projet
COPY --chown=appuser:appgroup pyproject.toml ./

# Copier les configurations Kedro (base + api)
COPY --chown=appuser:appgroup conf/base/ ./conf/base/
COPY --chown=appuser:appgroup conf/api/ ./conf/api/

# Créer les dossiers requis :
# - src/  : Kedro valide que source_dir existe (pyproject.toml: source_dir="src")
#           Le package est dans le venv, src/ peut être vide
# - data/ : data/api/ sera monté en volume au docker run
RUN mkdir -p src mlruns data/api data/06_models data/08_reporting && \
    chown -R appuser:appgroup src/ mlruns/ data/

# Copier le modèle entraîné (généré par kedro run dans le job CT)
COPY --chown=appuser:appgroup data/api/ ./data/api/

# Exposer le port de l'API
EXPOSE 8000

# Passer à l'utilisateur non-root
USER appuser

# Healthcheck Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/')" \
    || exit 1

# Point d'entrée : le script CLI installé par uv dans le venv
# credit-scoring-api = credit_scoring_kedro.api.main:main
# qui appelle uvicorn.run("credit_scoring_kedro.api.app:api", ...)
CMD ["credit-scoring-api"]
