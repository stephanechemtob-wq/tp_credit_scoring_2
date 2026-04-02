"""Script d'export des artefacts Kedro vers data/api/.

Ce script est à exécuter APRÈS le pipeline Kedro (kedro run) pour préparer
les artefacts nécessaires au démarrage de l'API FastAPI.

Usage :
    python scripts/export_artifacts.py

Il copie :
    data/06_models/credit_scoring_model.pkl  → data/api/model.pkl
    data/06_models/preprocessor.pkl          → data/api/preprocessor.pkl
    data/08_reporting/model_metadata.json    → data/api/model_metadata.json
"""

import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[1]
DATA_MODELS = PROJECT_ROOT / "data" / "06_models"
DATA_REPORTING = PROJECT_ROOT / "data" / "08_reporting"
DATA_API = PROJECT_ROOT / "data" / "api"


def export_artifacts() -> None:
    """Exporte les artefacts du pipeline Kedro vers data/api/."""
    DATA_API.mkdir(parents=True, exist_ok=True)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model_src = DATA_MODELS / "credit_scoring_model.pkl"
    model_dst = DATA_API / "model.pkl"
    if model_src.exists():
        shutil.copy2(model_src, model_dst)
        logger.info("Modèle exporté : %s → %s", model_src, model_dst)
    else:
        logger.error("Modèle introuvable : %s — Lancez d'abord : kedro run", model_src)
        raise FileNotFoundError(f"Modèle introuvable : {model_src}")

    # ── Preprocesseur ─────────────────────────────────────────────────────────
    prep_src = DATA_MODELS / "preprocessor.pkl"
    prep_dst = DATA_API / "preprocessor.pkl"
    if prep_src.exists():
        shutil.copy2(prep_src, prep_dst)
        logger.info("Preprocesseur exporté : %s → %s", prep_src, prep_dst)
    else:
        logger.error("Preprocesseur introuvable : %s", prep_src)
        raise FileNotFoundError(f"Preprocesseur introuvable : {prep_src}")

    # ── Métadonnées ───────────────────────────────────────────────────────────
    meta_src = DATA_REPORTING / "model_metadata.json"
    meta_dst = DATA_API / "model_metadata.json"
    if meta_src.exists():
        shutil.copy2(meta_src, meta_dst)
        logger.info("Métadonnées exportées : %s → %s", meta_src, meta_dst)
    else:
        # Générer des métadonnées minimales si le fichier n'existe pas encore
        logger.warning("Métadonnées introuvables, génération de métadonnées par défaut.")
        default_meta = {
            "model_name": "credit_scoring_model",
            "model_version": "1.0.0",
            "algorithm": "GradientBoostingClassifier",
            "feature_names": [
                "age",
                "income",
                "loan_amount",
                "loan_term",
                "credit_score",
                "employment_years",
                "debt_to_income",
                "num_credit_lines",
                "num_late_payments",
                "has_mortgage",
                "has_dependents",
                "loan_purpose",
                "education_level",
                "employment_type",
            ],
            "metrics": {"roc_auc_test": 0.0, "f1_test": 0.0},
            "training_date": "unknown",
            "threshold": 0.5,
        }
        with open(meta_dst, "w", encoding="utf-8") as f:
            json.dump(default_meta, f, indent=2, ensure_ascii=False)

    logger.info("Export terminé. Artefacts disponibles dans : %s", DATA_API)
    logger.info("Vous pouvez maintenant démarrer l'API : python -m credit_scoring_kedro.api.main")


if __name__ == "__main__":
    export_artifacts()
