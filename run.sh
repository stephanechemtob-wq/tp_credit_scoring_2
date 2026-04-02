#!/bin/bash

set -e

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PKG_NAME="credit_scoring_kedro"

function load-dotenv {
    if [ -f "$THIS_DIR/.env" ]; then
        while read -r line; do
            export "$line"
        done < <(grep -v '^#' "$THIS_DIR/.env" | grep -v '^$')
    fi
}

function install {
    echo "Installation des dépendances avec uv..."
    uv sync --extra dev
    uv run pre-commit install
}

function lint {
    echo "Exécution des linters..."
    uv run pre-commit run --all-files
}

function lint:ci {
    echo "Exécution des linters pour la CI..."
    SKIP=no-commit-to-branch uv run pre-commit run --all-files
}

function test:quick {
    echo "Exécution des tests rapides..."
    PYTEST_EXIT_STATUS=0
    uv run pytest -m 'not slow' "$THIS_DIR/tests/" \
           --cov "$THIS_DIR/src/$PKG_NAME" \
           --cov-report html \
           --cov-report term \
           --cov-report xml \
           --junit-xml "$THIS_DIR/test-reports/report.xml" \
           --cov-fail-under 50 || ((PYTEST_EXIT_STATUS+=$?))

    mkdir -p "$THIS_DIR/test-reports"
    mv coverage.xml "$THIS_DIR/test-reports/" 2>/dev/null || true
    mv htmlcov "$THIS_DIR/test-reports/" 2>/dev/null || true
    return $PYTEST_EXIT_STATUS
}

function test {
    echo "Exécution des tests..."
    PYTEST_EXIT_STATUS=0
    uv run pytest "${@:-$THIS_DIR/tests/}" \
           --cov "$THIS_DIR/src/$PKG_NAME" \
           --cov-report html \
           --cov-report term \
           --cov-report xml \
           --junit-xml "$THIS_DIR/test-reports/report.xml" \
           --cov-fail-under 50 || ((PYTEST_EXIT_STATUS+=$?))

    mkdir -p "$THIS_DIR/test-reports"
    mv coverage.xml "$THIS_DIR/test-reports/" 2>/dev/null || true
    mv htmlcov "$THIS_DIR/test-reports/" 2>/dev/null || true
    return $PYTEST_EXIT_STATUS
}

function test:ci {
    echo "Exécution des tests pour la CI..."
    PYTEST_EXIT_STATUS=0
    uv run pytest -m 'not slow' "$THIS_DIR/tests/" \
           --cov "$THIS_DIR/src/$PKG_NAME" \
           --cov-report html \
           --cov-report term \
           --cov-report xml \
           --junit-xml "$THIS_DIR/test-reports/report.xml" \
           --cov-fail-under 50 || ((PYTEST_EXIT_STATUS+=$?))

    mkdir -p "$THIS_DIR/test-reports"
    mv coverage.xml "$THIS_DIR/test-reports/" 2>/dev/null || true
    mv htmlcov "$THIS_DIR/test-reports/" 2>/dev/null || true
    return $PYTEST_EXIT_STATUS
}

function test:wheel-locally {
    echo "Test du package buildé (wheel) localement..."
    rm -rf test-env || true

    # Création d'un venv temporaire isolé
    uv venv test-env

    clean || true
    build

    PYTEST_EXIT_STATUS=0
    # Installation du wheel et des dépendances de test dans le venv isolé
    VIRTUAL_ENV=test-env uv pip install ./dist/*.whl pytest pytest-cov

    # Exécution des tests sur le package installé
    VIRTUAL_ENV=test-env uv run pytest -m 'not slow' "$THIS_DIR/tests/" \
           --cov "$PKG_NAME" \
           --cov-report html \
           --cov-report term \
           --cov-report xml \
           --junit-xml "$THIS_DIR/test-reports/report.xml" \
           --cov-fail-under 50 || ((PYTEST_EXIT_STATUS+=$?))

    mkdir -p "$THIS_DIR/test-reports"
    mv coverage.xml "$THIS_DIR/test-reports/" 2>/dev/null || true
    mv htmlcov "$THIS_DIR/test-reports/" 2>/dev/null || true

    rm -rf test-env
    return $PYTEST_EXIT_STATUS
}

function serve-coverage-report {
    echo "Serveur de rapport de couverture sur http://localhost:8000"
    uv run python -m http.server --directory "$THIS_DIR/test-reports/htmlcov/"
}

function test:all {
    if [ $# -eq 0 ]; then
        uv run pytest "$THIS_DIR/tests/" \
            --cov="$THIS_DIR/src/$PKG_NAME" \
            --cov-report html
    else
        uv run pytest "$@"
    fi
}

function build {
    echo "Build du package (sdist et wheel)..."
    uv build
}

function release:test {
    lint
    clean
    build
    publish:test
}

function release:prod {
    release:test
    publish:prod
}

function publish:test {
    echo "Publication sur TestPyPI..."
    load-dotenv
    uv publish dist/* \
        --publish-url https://test.pypi.org/legacy/ \
        --token "$TEST_PYPI_TOKEN"
}

function publish:prod {
    echo "Publication sur PyPI..."
    load-dotenv
    uv publish dist/* \
        --token "$PROD_PYPI_TOKEN"
}

function run-pipeline {
    echo "Exécution du pipeline Kedro..."
    load-dotenv
    uv run kedro run
}

function run-api {
    echo "Lancement de l'API FastAPI..."
    load-dotenv
    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
}

function clean {
    echo "Nettoyage des fichiers temporaires..."
    rm -rf dist build coverage.xml test-reports .pytest_cache .ruff_cache test-env
    find . \
      -type d \
      \( \
        -name "__pycache__" \
        -o -name "*.egg-info" \
        -o -name "*htmlcov" \
        -o -name ".kedro" \
      \) \
      -not -path "*/\.venv/*" \
      -exec rm -r {} + 2>/dev/null || true
}

function help {
    echo "Utilisation: $0 <task> <args>"
    echo "Tâches disponibles :"
    compgen -A function | grep -v "^_" | cat -n
}

TIMEFORMAT="Tâche terminée en %3lR"
time ${@:-help}
