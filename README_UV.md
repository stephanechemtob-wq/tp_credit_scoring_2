# Workflow uv — Credit Scoring Kedro

Ce projet utilise [**uv**](https://docs.astral.sh/uv/) comme gestionnaire d'environnement et de dépendances Python.
Le `uv.lock` garantit une reproductibilité parfaite entre le poste local et le conteneur Docker.

---

## Prérequis

```bash
# Installer uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

## Workflow local

### 1. Créer l'environnement virtuel (Python 3.12)

```bash
uv venv
# → Crée .venv/ avec CPython 3.12
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows PowerShell
```

### 2. Installer les dépendances

```bash
# Installation standard (depuis pyproject.toml → génère uv.lock)
uv sync

# Avec les dépendances de développement
uv sync --extra dev

# Avec les tests de charge
uv sync --extra dev --extra load-testing

# Avec un extra cloud (UN SEUL à la fois)
uv sync --extra cloud-aws    # AWS S3
uv sync --extra cloud-gcp    # GCP GCS
uv sync --extra cloud-azure  # Azure Blob
```

### 3. Lancer le pipeline Kedro

```bash
# Entraîner le modèle (génère data/06_models/)
uv run kedro run

# Exporter les artefacts vers data/api/ (pour le serving)
uv run python scripts/export_artifacts.py

# Visualiser le pipeline
uv run kedro viz
```

### 4. Lancer l'API FastAPI

```bash
# Mode développement (avec rechargement automatique)
uv run uvicorn credit_scoring_kedro.api.main:app --reload --port 8000

# Mode production
uv run uvicorn credit_scoring_kedro.api.main:app --workers 2 --port 8000
```

L'API est accessible sur :
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **Health** : http://localhost:8000/health/

### 5. Lancer les tests

```bash
# Tests unitaires et d'intégration
uv run pytest

# Avec couverture
uv run pytest --cov=src --cov-report=html

# Tests de sécurité uniquement
uv run pytest tests/api/test_security.py -v

# Tests de charge (Locust — interface web)
uv run locust -f tests/load/locustfile.py --host=http://localhost:8000
```

---

## Workflow Docker

Le Dockerfile utilise `uv sync --frozen` pour reproduire **exactement** le `uv.lock` généré en local.

### Prérequis

```bash
# 1. Générer le uv.lock en local (à faire une seule fois, ou après chaque update)
uv lock

# 2. Lancer le pipeline Kedro pour générer les artefacts ML
uv run kedro run
uv run python scripts/export_artifacts.py
```

### Build de l'image

```bash
docker build -t credit-scoring-api:latest .
```

### Lancer le conteneur

```bash
# Monter data/api/ en volume (artefacts ML)
docker run -p 8000:8000 \
  -v $(pwd)/data/api:/app/data/api:ro \
  -e JWT_SECRET_KEY="votre-secret-de-production" \
  credit-scoring-api:latest
```

### Variables d'environnement disponibles

| Variable | Défaut | Description |
|---|---|---|
| `JWT_SECRET_KEY` | `mlops-credit-scoring-secret-key-change-in-prod` | Clé secrète JWT — **à changer en production** |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Durée de vie du token d'accès (minutes) |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Durée de vie du token de refresh (jours) |
| `API_WORKERS` | `2` | Nombre de workers uvicorn |
| `LOG_LEVEL` | `info` | Niveau de log (`debug`, `info`, `warning`, `error`) |
| `KEDRO_ENV` | `api` | Environnement Kedro utilisé pour le catalog |

---

## Mettre à jour les dépendances

```bash
# Mettre à jour toutes les dépendances (recalcule uv.lock)
uv lock --upgrade

# Mettre à jour une dépendance spécifique
uv lock --upgrade-package fastapi

# Appliquer les mises à jour
uv sync
```

---

## Alignement local ↔ Docker

| Étape | Local | Docker |
|---|---|---|
| **Résolution** | `uv lock` | `uv sync --frozen` (depuis le `uv.lock` local) |
| **Installation** | `uv sync` | `uv sync --frozen --no-dev` |
| **Lancement** | `uv run uvicorn ...` | `uv run uvicorn ...` |
| **Python** | CPython 3.12 (`.venv`) | `python:3.12-slim` |
| **Reproductibilité** | `uv.lock` versionné dans Git | `COPY uv.lock ./` dans le Dockerfile |

> **Règle d'or** : le `uv.lock` doit toujours être commité dans Git.
> C'est lui qui garantit que le Docker build produit exactement le même environnement que le poste local.
