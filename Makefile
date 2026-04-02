.PHONY: install \
        lint lint-ci \
        test test-ci test-quick test-all test-wheel-locally \
        run-pipeline run-api \
        build release-test release-prod publish-test publish-prod \
        serve-coverage-report \
        clean help

# =============================================================================
# SETUP
# =============================================================================

install:
	/bin/bash ./run.sh install

# =============================================================================
# QUALITÉ DU CODE
# En local (branche feature)  → make lint
# En CI (GitHub Actions)      → make lint-ci  (skip no-commit-to-branch)
# =============================================================================

lint:
	/bin/bash ./run.sh lint

lint-ci:
	/bin/bash ./run.sh lint:ci

# =============================================================================
# TESTS
# En local                    → make test  ou  make test-quick
# En CI (GitHub Actions)      → make test-ci  (exclut les tests 'slow')
# Test du wheel buildé        → make test-wheel-locally
# =============================================================================

test:
	/bin/bash ./run.sh test

test-ci:
	/bin/bash ./run.sh test:ci

test-quick:
	/bin/bash ./run.sh test:quick

test-all:
	/bin/bash ./run.sh test:all

test-wheel-locally:
	/bin/bash ./run.sh test:wheel-locally

serve-coverage-report:
	/bin/bash ./run.sh serve-coverage-report

# =============================================================================
# PIPELINE KEDRO (Continuous Training)
# =============================================================================

run-pipeline:
	/bin/bash ./run.sh run-pipeline

# =============================================================================
# API FASTAPI
# =============================================================================

run-api:
	/bin/bash ./run.sh run-api

# =============================================================================
# BUILD & PUBLICATION
# =============================================================================

build:
	/bin/bash ./run.sh build

release-test:
	/bin/bash ./run.sh release:test

release-prod:
	/bin/bash ./run.sh release:prod

publish-test:
	/bin/bash ./run.sh publish:test

publish-prod:
	/bin/bash ./run.sh publish:prod

# =============================================================================
# NETTOYAGE
# =============================================================================

clean:
	/bin/bash ./run.sh clean

# =============================================================================
# AIDE
# =============================================================================

help:
	@echo ""
	@echo "Cibles disponibles :"
	@echo ""
	@echo "  SETUP"
	@echo "    make install              Installe les dépendances (uv sync + pre-commit)"
	@echo ""
	@echo "  QUALITÉ DU CODE"
	@echo "    make lint                 Linting complet (usage local, branche feature)"
	@echo "    make lint-ci              Linting CI (skip no-commit-to-branch)"
	@echo ""
	@echo "  TESTS"
	@echo "    make test                 Tous les tests avec couverture"
	@echo "    make test-ci              Tests CI (exclut les tests marqués 'slow')"
	@echo "    make test-quick           Tests rapides (exclut 'slow')"
	@echo "    make test-all             Tous les tests sans seuil de couverture"
	@echo "    make test-wheel-locally   Test du package buildé (wheel)"
	@echo "    make serve-coverage-report  Rapport HTML de couverture sur localhost:8000"
	@echo ""
	@echo "  PIPELINE KEDRO (CT)"
	@echo "    make run-pipeline         Exécute kedro run"
	@echo ""
	@echo "  API FASTAPI"
	@echo "    make run-api              Lance uvicorn en mode reload"
	@echo ""
	@echo "  BUILD & PUBLICATION"
	@echo "    make build                Build sdist + wheel"
	@echo "    make release-test         lint + clean + build + publish TestPyPI"
	@echo "    make release-prod         release-test + publish PyPI"
	@echo "    make publish-test         Publie sur TestPyPI"
	@echo "    make publish-prod         Publie sur PyPI"
	@echo ""
	@echo "  NETTOYAGE"
	@echo "    make clean                Supprime dist, build, test-reports, caches"
	@echo ""
