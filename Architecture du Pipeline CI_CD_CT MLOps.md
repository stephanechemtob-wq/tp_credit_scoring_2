# Architecture du Pipeline CI/CD/CT MLOps

Ce document présente l'architecture visuelle du pipeline d'intégration, de déploiement et d'entraînement continus (CI/CD/CT) pour le projet de Credit Scoring (Kedro + FastAPI + Docker).

---

## 1. Vue d'ensemble des Environnements

Contrairement à un package Python classique publié sur PyPI, un projet MLOps déploie une API conteneurisée et gère des modèles de Machine Learning.

```mermaid
flowchart LR
    Dev[Laptop Developpeur] --> CI[GitHub Actions CI/CT]
    CI --> Registry[Docker Hub]
    Registry --> Staging[Environnement Staging]
    Staging --> Prod[Environnement Production]
```

---

## 2. Workflow CI/CD/CT Haut Niveau

```mermaid
sequenceDiagram
    actor Dev as Developpeur
    participant GitHub
    participant GA as GitHub Actions
    participant Docker as Docker Hub
    participant Cloud as Serveur Render

    Dev->>GitHub: Ouvre une Pull Request
    GitHub->>GA: Evenement pull_request
    GA->>GA: Job CI - Lint et Tests
    Dev->>GitHub: Merge la PR sur main
    GitHub->>GA: Evenement push main
    GA->>GA: Job CI - Lint et Tests
    GA->>GA: Job CT - kedro run
    GA->>GA: Job CD - Build image Docker
    GA->>Docker: Push image Docker
    GA->>Cloud: Webhook de deploiement
    Cloud->>Docker: Pull nouvelle image
    Cloud->>Cloud: Redemarrage API FastAPI
```

---

## 3. Workflow CI/CD/CT Détaillé

```mermaid
sequenceDiagram
    actor Dev as Developpeur
    participant GitHub
    participant GA as GitHub Actions
    participant Docker as Docker Hub
    participant Cloud as Serveur Render

    Dev->>Dev: Cree une branche feature
    Dev->>Dev: Developpe et commit
    Dev->>GitHub: Push la branche feature
    Dev->>GitHub: Ouvre une Pull Request

    GitHub->>GA: Evenement pull_request opened
    GA->>GA: Lint Ruff et Mypy
    GA->>GA: Tests Pytest avec couverture

    Dev->>Dev: Itere selon les retours CI
    Dev->>GitHub: Push nouveaux commits
    GitHub->>GA: Evenement pull_request synchronize
    GA->>GA: Re-execute CI

    Dev->>GitHub: Merge PR sur main
    GitHub->>GA: Evenement push main

    GA->>GA: CI - Validation finale lint et tests
    GA->>GA: CT - uv run kedro run
    GA->>GA: CT - Sauvegarde artefacts model.pkl
    GA->>GA: CD - Build image Docker avec nouveau modele
    GA->>Docker: Push image avec tag version
    GA->>Cloud: Declenchement webhook deploiement
```

---

## 4. GitHub Actions comme Système Pub-Sub MLOps

Dans un contexte MLOps, les déclencheurs sont plus variés pour gérer le réentraînement automatique.

```mermaid
graph LR
    subgraph Triggers["Declencheurs"]
        Dev(Developpeur)
        Schedule(Cron mensuel)
        Monitoring(Evidently Data Drift)
    end

    subgraph GitHub_API["GitHub Event Bus"]
        GitHub("GitHub API")
    end

    subgraph Workflows["Jobs GitHub Actions"]
        CI(Job CI - Lint et Tests)
        CT(Job CT - Kedro Run)
        CD(Job CD - Docker et Deploy)
    end

    Dev -->|Push PR ou Merge main| GitHub
    Dev -->|workflow_dispatch manuel| GitHub
    Schedule -->|Cron 1er du mois| GitHub
    Monitoring -->|repository_dispatch drift| GitHub

    GitHub -->|pull_request| CI
    GitHub -->|push main| CI
    GitHub -->|push main| CT
    GitHub -->|schedule| CT
    GitHub -->|repository_dispatch| CT
    GitHub -->|workflow_dispatch| CT

    CT -->|succes| CD
```

---

## 5. Stratégie de Réentraînement (Continuous Training)

Le réentraînement du modèle est déclenché par quatre types d'événements distincts :

| Déclencheur | Mécanisme | Cas d'usage |
|---|---|---|
| **Nouveau code** | `push` sur `main` | Modification des hyperparamètres ou du feature engineering |
| **Temporel** | `schedule` cron mensuel | Intégration des nouvelles données collectées |
| **Data Drift** | `repository_dispatch` webhook | Dégradation détectée par Evidently en production |
| **Manuel** | `workflow_dispatch` | Forçage par un administrateur depuis l'interface GitHub |
