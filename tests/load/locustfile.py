"""
Tests de charge Locust pour l'API de scoring de crédit.

Scénarios disponibles (selon PDF "Tests pratiques de déploiement et d'inférence") :

1. CreditScoringUser       — Test de charge nominal (50 users, spawn 5/s)
2. BatchScoringUser        — Test batch (10 users, spawn 1/s)
3. AuthStressUser          — Test brute-force auth (5 users, spawn 1/s)
4. SpikeTestUser           — Test de pic soudain (montée rapide)
5. EnduranceTestUser       — Test d'endurance (1h, charge constante)

Lancement :
    # Interface web (recommandé pour la formation)
    locust -f tests/load/locustfile.py --host=http://localhost:8000

    # Mode headless (CI/CD)
    locust -f tests/load/locustfile.py \
           --host=http://localhost:8000 \
           --users=50 --spawn-rate=5 --run-time=5m \
           --headless --csv=reports/locust_results

Critères de succès (selon PDF) :
    - 0% d'échecs
    - Temps de réponse moyen < 200ms
    - P95 < 500ms
    - RPS stable (variation < 20%)
"""

from __future__ import annotations

import random

from locust import HttpUser, between, events, task

# ---------------------------------------------------------------------------
# Données de test — profils de demandeurs de crédit
# ---------------------------------------------------------------------------

# Profil solvable (faible risque)
PROFIL_SOLVABLE = {
    "age": 35,
    "income": 65000.0,
    "loan_amount": 10000.0,
    "loan_term_months": 36,
    "credit_score": 720,
    "employment_years": 8,
    "num_credit_lines": 4,
    "debt_to_income_ratio": 0.25,
    "has_mortgage": True,
    "has_car_loan": False,
    "num_late_payments": 0,
    "savings_balance": 15000.0,
    "monthly_expenses": 2500.0,
    "num_dependents": 1,
}

# Profil à risque élevé
PROFIL_RISQUE = {
    "age": 28,
    "income": 22000.0,
    "loan_amount": 18000.0,
    "loan_term_months": 60,
    "credit_score": 520,
    "employment_years": 1,
    "num_credit_lines": 8,
    "debt_to_income_ratio": 0.75,
    "has_mortgage": False,
    "has_car_loan": True,
    "num_late_payments": 5,
    "savings_balance": 500.0,
    "monthly_expenses": 1800.0,
    "num_dependents": 3,
}

# Profil intermédiaire
PROFIL_MOYEN = {
    "age": 42,
    "income": 45000.0,
    "loan_amount": 15000.0,
    "loan_term_months": 48,
    "credit_score": 640,
    "employment_years": 5,
    "num_credit_lines": 6,
    "debt_to_income_ratio": 0.45,
    "has_mortgage": True,
    "has_car_loan": True,
    "num_late_payments": 2,
    "savings_balance": 5000.0,
    "monthly_expenses": 3200.0,
    "num_dependents": 2,
}

PROFILS = [PROFIL_SOLVABLE, PROFIL_RISQUE, PROFIL_MOYEN]


def random_profil() -> dict:
    """Génère un profil aléatoire avec légères variations pour simuler des données réelles."""
    base = random.choice(PROFILS).copy()
    # Ajouter du bruit pour éviter le cache
    base["income"] = base["income"] * random.uniform(0.9, 1.1)
    base["loan_amount"] = base["loan_amount"] * random.uniform(0.95, 1.05)
    base["credit_score"] = min(850, max(300, base["credit_score"] + random.randint(-20, 20)))
    return base


def generate_batch(size: int = 10) -> list[dict]:
    """Génère un batch de profils aléatoires."""
    return [random_profil() for _ in range(size)]


# ---------------------------------------------------------------------------
# Token JWT partagé (obtenu une seule fois au démarrage)
# ---------------------------------------------------------------------------

_shared_token: str | None = None


def get_token(client) -> str:
    """Obtient un token JWT en s'authentifiant. Utilise le token partagé si disponible."""
    global _shared_token
    if _shared_token:
        return _shared_token

    response = client.post(
        "/auth/token",
        data={"username": "data_scientist", "password": "mlops2024"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        name="/auth/token [setup]",
    )
    if response.status_code == 200:
        _shared_token = response.json()["access_token"]
        return _shared_token
    return ""


# ---------------------------------------------------------------------------
# Scénario 1 : Test de charge nominal — CreditScoringUser
# ---------------------------------------------------------------------------


class CreditScoringUser(HttpUser):
    """
    Utilisateur standard simulant un usage normal de l'API.

    Paramètres recommandés (selon PDF Locust) :
        --users=50 --spawn-rate=5 --run-time=5m

    Critères de succès :
        - 0% d'échecs
        - Temps moyen < 200ms
        - P95 < 500ms
        - RPS ≈ 16-22 (comme dans le PDF)
    """

    wait_time = between(1, 3)
    weight = 3  # 3x plus fréquent que les autres scénarios

    def on_start(self):
        """Authentification au démarrage de chaque utilisateur simulé."""
        self.token = get_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(5)
    def predict_solvable(self):
        """Prédiction sur un profil solvable (tâche la plus fréquente)."""
        with self.client.post(
            "/predict",
            json=PROFIL_SOLVABLE,
            headers=self.headers,
            name="/predict [solvable]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "prediction" not in data:
                    response.failure("Réponse invalide : champ 'prediction' manquant")
                elif response.elapsed.total_seconds() > 1.0:
                    response.failure(
                        f"Temps de réponse trop élevé : {response.elapsed.total_seconds():.2f}s"
                    )
                else:
                    response.success()
            elif response.status_code == 429:
                response.failure("Rate limit atteint (429)")
            elif response.status_code == 401:
                response.failure("Token expiré ou invalide (401)")
            else:
                response.failure(f"Erreur inattendue : {response.status_code}")

    @task(3)
    def predict_random(self):
        """Prédiction sur un profil aléatoire."""
        with self.client.post(
            "/predict",
            json=random_profil(),
            headers=self.headers,
            name="/predict [random]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(2)
    def check_health(self):
        """Vérification de santé (endpoint public, sans token)."""
        with self.client.get(
            "/health",
            name="/health",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") != "healthy":
                    response.failure(f"API non saine : {data.get('status')}")
                else:
                    response.success()
            else:
                response.failure(f"Health check échoué : {response.status_code}")

    @task(1)
    def get_model_info(self):
        """Récupération des informations du modèle."""
        with self.client.get(
            "/model/info",
            headers=self.headers,
            name="/model/info",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# ---------------------------------------------------------------------------
# Scénario 2 : Test batch — BatchScoringUser
# ---------------------------------------------------------------------------


class BatchScoringUser(HttpUser):
    """
    Utilisateur simulant des appels batch (traitement par lots).

    Paramètres recommandés :
        --users=10 --spawn-rate=1 --run-time=5m

    Note : Les appels batch sont plus coûteux, d'où le wait_time plus long.
    """

    wait_time = between(3, 8)
    weight = 1

    def on_start(self):
        self.token = get_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task(3)
    def predict_batch_small(self):
        """Batch de 5 dossiers (petit)."""
        with self.client.post(
            "/predict/batch",
            json={"applications": generate_batch(5)},
            headers=self.headers,
            name="/predict/batch [5 dossiers]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data.get("predictions", [])) != 5:
                    response.failure("Nombre de prédictions incorrect")
                else:
                    response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)
    def predict_batch_large(self):
        """Batch de 50 dossiers (grand)."""
        with self.client.post(
            "/predict/batch",
            json={"applications": generate_batch(50)},
            headers=self.headers,
            name="/predict/batch [50 dossiers]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if len(data.get("predictions", [])) != 50:
                    response.failure("Nombre de prédictions incorrect")
                else:
                    response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# ---------------------------------------------------------------------------
# Scénario 3 : Test de sécurité — AuthStressUser
# ---------------------------------------------------------------------------


class AuthStressUser(HttpUser):
    """
    Simule des tentatives d'authentification répétées.
    Vérifie que le rate limiting (5 req/min) bloque les tentatives excessives.

    Paramètres recommandés :
        --users=5 --spawn-rate=1 --run-time=2m
    """

    wait_time = between(0.5, 1)
    weight = 1

    @task(3)
    def login_valid(self):
        """Authentification valide."""
        with self.client.post(
            "/auth/token",
            data={"username": "data_scientist", "password": "mlops2024"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            name="/auth/token [valid]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Rate limit atteint — comportement attendu sous forte charge
                response.success()  # On marque comme succès car c'est le comportement voulu
            else:
                response.failure(f"Status inattendu: {response.status_code}")

    @task(1)
    def login_invalid(self):
        """Tentative avec mauvais mot de passe — doit retourner 401."""
        with self.client.post(
            "/auth/token",
            data={"username": "data_scientist", "password": "wrong_password"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            name="/auth/token [invalid]",
            catch_response=True,
        ) as response:
            if response.status_code == 401:
                response.success()  # Comportement attendu
            elif response.status_code == 429:
                response.success()  # Rate limit — aussi attendu
            else:
                response.failure(f"Status inattendu: {response.status_code}")

    @task(1)
    def access_without_token(self):
        """Accès à un endpoint protégé sans token — doit retourner 401."""
        with self.client.post(
            "/predict",
            json=PROFIL_SOLVABLE,
            name="/predict [no token]",
            catch_response=True,
        ) as response:
            if response.status_code == 401:
                response.success()  # Comportement attendu
            else:
                response.failure(f"Endpoint non protégé ! Status: {response.status_code}")


# ---------------------------------------------------------------------------
# Scénario 4 : Test de pic — SpikeTestUser
# ---------------------------------------------------------------------------


class SpikeTestUser(HttpUser):
    """
    Simule un pic soudain de trafic (ex: campagne marketing, fin de mois).

    Paramètres recommandés :
        --users=200 --spawn-rate=50 --run-time=3m
    (Montée rapide à 200 users puis maintien)
    """

    wait_time = between(0.1, 0.5)  # Très peu d'attente = charge maximale
    weight = 2

    def on_start(self):
        self.token = get_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @task
    def predict_spike(self):
        """Prédiction sous pic de charge."""
        with self.client.post(
            "/predict",
            json=random_profil(),
            headers=self.headers,
            name="/predict [spike]",
            catch_response=True,
        ) as response:
            if response.status_code in (200, 429):
                response.success()
            elif response.status_code == 503:
                response.failure("Service indisponible sous charge (503)")
            else:
                response.failure(f"Status: {response.status_code}")


# ---------------------------------------------------------------------------
# Scénario 5 : Test d'endurance — EnduranceTestUser
# ---------------------------------------------------------------------------


class EnduranceTestUser(HttpUser):
    """
    Test d'endurance sur longue durée (détection de fuites mémoire, dégradation).

    Paramètres recommandés :
        --users=20 --spawn-rate=2 --run-time=1h

    Métriques à surveiller :
        - Temps de réponse stable (pas de dégradation progressive)
        - 0% d'échecs sur la durée
        - Pas de memory leak (surveiller via /health)
    """

    wait_time = between(2, 5)
    weight = 1

    def on_start(self):
        self.token = get_token(self.client)
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.request_count = 0

    @task(4)
    def predict_endurance(self):
        """Prédiction continue pour test d'endurance."""
        self.request_count += 1
        with self.client.post(
            "/predict",
            json=random_profil(),
            headers=self.headers,
            name="/predict [endurance]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                # Vérifier la dégradation des performances
                elapsed_ms = response.elapsed.total_seconds() * 1000
                if elapsed_ms > 2000:
                    response.failure(f"Dégradation détectée : {elapsed_ms:.0f}ms (seuil: 2000ms)")
                else:
                    response.success()
            elif response.status_code == 401:
                # Token expiré — renouveler
                self.token = get_token(self.client)
                self.headers = {"Authorization": f"Bearer {self.token}"}
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)
    def health_check_endurance(self):
        """Vérification périodique de la santé pendant l'endurance."""
        with self.client.get(
            "/health",
            name="/health [endurance]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") != "healthy":
                    response.failure(f"Dégradation de santé détectée : {data}")
                else:
                    response.success()
            else:
                response.failure(f"Health check échoué : {response.status_code}")


# ---------------------------------------------------------------------------
# Hooks Locust — Rapport personnalisé
# ---------------------------------------------------------------------------


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Affiche un résumé des résultats à la fin du test (comme dans le PDF Locust)."""
    stats = environment.stats

    print("\n" + "=" * 70)
    print("RAPPORT DE TEST DE CHARGE — Credit Scoring API")
    print("=" * 70)

    total_requests = sum(s.num_requests for s in stats.entries.values())
    total_failures = sum(s.num_failures for s in stats.entries.values())
    failure_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0

    print(f"\nTotal requêtes    : {total_requests:,}")
    print(f"Total échecs      : {total_failures:,}")
    print(f"Taux d'échec      : {failure_rate:.2f}%")

    print("\nDétail par endpoint :")
    print(f"{'Endpoint':<35} {'Req':>8} {'Échecs':>8} {'Moy (ms)':>10} {'P95 (ms)':>10} {'RPS':>8}")
    print("-" * 80)

    for name, stat in sorted(stats.entries.items(), key=lambda x: x[0][0]):
        if stat.num_requests > 0:
            print(
                f"{name[0]:<35} "
                f"{stat.num_requests:>8,} "
                f"{stat.num_failures:>8,} "
                f"{stat.avg_response_time:>10.1f} "
                f"{stat.get_response_time_percentile(0.95):>10.1f} "
                f"{stat.current_rps:>8.1f}"
            )

    print("\n" + "=" * 70)
    if failure_rate == 0:
        print("RÉSULTAT : SUCCÈS — 0% d'échecs")
    elif failure_rate < 1:
        print(f"RÉSULTAT : ACCEPTABLE — {failure_rate:.2f}% d'échecs (< 1%)")
    else:
        print(f"RÉSULTAT : ÉCHEC — {failure_rate:.2f}% d'échecs (> 1%)")
    print("=" * 70 + "\n")
