"""Point d'entrée uvicorn de l'API Credit Scoring.

Usage :
    # Développement local
    python -m credit_scoring_kedro.api.main

    # Via uvicorn directement
    uvicorn credit_scoring_kedro.api.app:api --host 0.0.0.0 --port 8000 --reload

    # En production (Docker)
    uvicorn credit_scoring_kedro.api.app:api --host 0.0.0.0 --port 8000 --workers 2
"""

import os

import uvicorn


def main() -> None:
    """Lance le serveur uvicorn."""
    uvicorn.run(
        "credit_scoring_kedro.api.app:api",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=int(os.getenv("API_WORKERS", "1")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
