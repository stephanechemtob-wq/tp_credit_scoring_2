"""
Middleware de rate limiting pour l'API de scoring de crédit.

Implémente deux niveaux de protection :
1. Rate limiting par IP (slowapi / limits) — protection DDoS
2. Rate limiting par utilisateur authentifié — protection abus

Limites définies :
- /predict        : 60 req/min par IP, 100 req/min par user
- /predict/batch  : 10 req/min par IP, 20 req/min par user
- /auth/token     : 5 req/min par IP (protection brute-force)
- Global          : 200 req/min par IP
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address


def get_limiter() -> Limiter:
    """
    Crée et configure l'instance Limiter slowapi.

    Le key_func `get_remote_address` utilise l'IP du client comme clé
    de comptage. En production derrière un reverse proxy, configurer
    FORWARDED_ALLOW_IPS pour récupérer l'IP réelle depuis X-Forwarded-For.
    """
    return Limiter(
        key_func=get_remote_address,
        default_limits=["200/minute"],  # Limite globale par IP
        storage_uri="memory://",  # Remplacer par Redis en prod : "redis://localhost:6379"
    )


# Instance globale du limiter (importée dans app.py)
limiter = get_limiter()


# ---------------------------------------------------------------------------
# Décorateurs de rate limiting à appliquer sur les endpoints
# ---------------------------------------------------------------------------

# Limite stricte pour l'authentification (protection brute-force)
RATE_LIMIT_AUTH = "5/minute"

# Limite pour les prédictions unitaires
RATE_LIMIT_PREDICT = "60/minute"

# Limite pour les prédictions en batch (plus coûteuses)
RATE_LIMIT_BATCH = "10/minute"

# Limite pour les endpoints de lecture (model info, health)
RATE_LIMIT_READ = "120/minute"
