"""Module de hooks Kedro — comportements dynamiques du projet.

Hooks disponibles :
- LoadEnvVarsHook      : charge automatiquement les variables d'environnement depuis .env
- ArtifactsExportHook  : exporte automatiquement les artefacts ML vers data/api/ après kedro run
- MLflowSetupHook      : configure MLflow (experiment, tags) avant chaque run de pipeline

Pattern inspiré du projet InvoicesPretPourProd.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import CatalogProtocol
from kedro.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hook 1 : Chargement automatique des variables d'environnement
# ─────────────────────────────────────────────────────────────────────────────
class LoadEnvVarsHook:
    """Kedro hook pour charger automatiquement les variables d'environnement depuis .env.

    Déclenché juste après la création du contexte Kedro, avant tout pipeline.
    Permet de centraliser la configuration sensible (clés API, credentials cloud)
    dans un fichier .env sans le committer dans Git.
    """

    @hook_impl
    def after_context_created(self, context: KedroContext) -> None:  # noqa: ARG002
        """Charge les variables d'environnement depuis le fichier .env.

        Args:
            context: Le contexte Kedro qui vient d'être créé.

        """
        try:
            from dotenv import find_dotenv, load_dotenv

            dotenv_path = find_dotenv()
            if dotenv_path:
                load_dotenv(dotenv_path)
                logger.info(
                    "Variables d'environnement chargées depuis : %s",
                    dotenv_path,
                )
            else:
                logger.warning(
                    "Aucun fichier .env trouvé. "
                    "Les variables d'environnement doivent être définies manuellement."
                )
        except ImportError:
            logger.warning(
                "python-dotenv non installé. " "Installez-le avec : uv add python-dotenv"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Hook 2 : Export automatique des artefacts ML vers data/api/ après kedro run
# ─────────────────────────────────────────────────────────────────────────────
class ArtifactsExportHook:
    """Kedro hook pour exporter automatiquement les artefacts ML vers data/api/.

    Déclenché APRÈS la fin du pipeline complet (after_pipeline_run).
    Remplace le script manuel scripts/export_artifacts.py :
    l'export se fait automatiquement à chaque `kedro run` réussi.

    Artefacts exportés :
        data/06_models/credit_scoring_model.pkl  → data/api/model.pkl
        data/06_models/preprocessor.pkl          → data/api/preprocessor.pkl
        data/08_reporting/model_metadata.json    → data/api/model_metadata.json
    """

    # Mapping source → destination des artefacts à exporter
    # Le modèle n'est plus exporté ici, il est géré par MLflow Model Registry
    ARTIFACTS = {
        "preprocessor": {
            "src": Path("data/06_models/preprocessor.pkl"),
            "dst": Path("data/api/preprocessor.pkl"),
        },
        "metadata": {
            "src": Path("data/08_reporting/model_metadata.json"),
            "dst": Path("data/api/model_metadata.json"),
        },
    }

    # Métadonnées par défaut si model_metadata.json n'existe pas encore
    DEFAULT_METADATA = {
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

    @hook_impl
    def after_pipeline_run(
        self,
        run_params: dict[str, Any],
        pipeline: Pipeline,  # noqa: ARG002
        catalog: CatalogProtocol,  # noqa: ARG002
    ) -> None:
        """Exporte les artefacts ML vers data/api/ après la fin du pipeline.

        Uniquement déclenché si le pipeline contient le nœud d'entraînement
        (pour éviter l'export lors de pipelines partiels).

        Args:
            run_params: Paramètres du run (pipeline_name, env, runner, etc.)
            pipeline: Le pipeline qui vient de s'exécuter.
            catalog: Le catalogue de données Kedro.

        """
        # Exporter uniquement si le pipeline training a tourné
        pipeline_name = run_params.get("pipeline_name") or "__default__"
        if pipeline_name not in ("__default__", "training", "all"):
            logger.debug(
                "Pipeline '%s' : export des artefacts ignoré (non applicable).",
                pipeline_name,
            )
            return

        project_root = Path(run_params.get("project_path", Path.cwd()))
        api_dir = project_root / "data" / "api"
        api_dir.mkdir(parents=True, exist_ok=True)

        exported = []
        errors = []

        for name, paths in self.ARTIFACTS.items():
            src = project_root / paths["src"]
            dst = project_root / paths["dst"]

            if src.exists():
                shutil.copy2(src, dst)
                exported.append(name)
                logger.info("Artefact exporté : %s → %s", src.name, dst)
            elif name == "metadata":
                # Générer des métadonnées minimales si absentes
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(self.DEFAULT_METADATA, f, indent=2, ensure_ascii=False)
                logger.warning(
                    "model_metadata.json absent — métadonnées par défaut générées dans %s",
                    dst,
                )
            else:
                errors.append(name)
                logger.error(
                    "Artefact '%s' introuvable : %s — Vérifiez que kedro run s'est terminé correctement.",
                    name,
                    src,
                )

        if exported:
            logger.info(
                "Export terminé (%d artefacts) → %s. "
                "Vous pouvez démarrer l'API : uv run credit-scoring-api",
                len(exported),
                api_dir,
            )
        if errors:
            logger.error(
                "Export incomplet — artefacts manquants : %s",
                ", ".join(errors),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Hook 3 : Configuration MLflow avant chaque run
# ─────────────────────────────────────────────────────────────────────────────
class MLflowSetupHook:
    """Kedro hook pour configurer MLflow avant chaque run de pipeline.

    Déclenché avant le démarrage du pipeline (before_pipeline_run).
    Configure l'experiment MLflow et ajoute des tags de traçabilité
    (pipeline_name, env, runner) sur chaque run.
    """

    @hook_impl
    def before_pipeline_run(
        self,
        run_params: dict[str, Any],
        pipeline: Pipeline,  # noqa: ARG002
        catalog: CatalogProtocol,  # noqa: ARG002
    ) -> None:
        """Configure les tags MLflow avant le démarrage du pipeline.

        Args:
            run_params: Paramètres du run (pipeline_name, env, runner, etc.)
            pipeline: Le pipeline qui va s'exécuter.
            catalog: Le catalogue de données Kedro.

        """
        try:
            import mlflow

            # Tags de traçabilité automatiques sur chaque run MLflow
            tags = {
                "kedro.pipeline": run_params.get("pipeline_name") or "__default__",
                "kedro.env": run_params.get("env", "base"),
                "kedro.runner": run_params.get("runner", "SequentialRunner"),
            }
            # Ajouter les tags au run actif s'il existe
            active_run = mlflow.active_run()
            if active_run:
                mlflow.set_tags(tags)
                logger.debug("Tags MLflow définis : %s", tags)

        except ImportError:
            logger.debug("MLflow non disponible — tags de traçabilité ignorés.")
        except Exception as e:  # noqa: BLE001
            logger.debug("Impossible de définir les tags MLflow : %s", e)
