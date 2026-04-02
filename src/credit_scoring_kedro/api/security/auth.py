"""
Module de sécurité JWT pour l'API de scoring de crédit.

Implémente :
- Création de tokens JWT (access + refresh)
- Validation et décodage des tokens
- Révocation des tokens (blacklist en mémoire)
- Dépendances FastAPI pour la protection des endpoints
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "mlops-credit-scoring-secret-key-change-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------


class Token(BaseModel):
    """Réponse du endpoint /token."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # secondes


class TokenData(BaseModel):
    """Données extraites d'un token JWT valide."""

    username: str | None = None
    scopes: list[str] = []


class UserCredentials(BaseModel):
    """Identifiants pour l'authentification."""

    username: str
    password: str


class RefreshRequest(BaseModel):
    """Requête de rafraîchissement de token."""

    refresh_token: str


# ---------------------------------------------------------------------------
# Gestion des mots de passe
# ---------------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe en clair contre son hash bcrypt."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Génère le hash bcrypt d'un mot de passe."""
    return pwd_context.hash(password)


# ---------------------------------------------------------------------------
# Base d'utilisateurs (en mémoire pour la démo — remplacer par DB en prod)
# ---------------------------------------------------------------------------

# Hashes pré-calculés (bcrypt, cost=12) — évite le recalcul à chaque import
# Générer via : python3 -c "from passlib.context import CryptContext; print(CryptContext(['bcrypt']).hash('MOT_DE_PASSE'))"
_HASHED_PASSWORDS: dict[str, str] = {
    "data_scientist": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/HS.iK8a",
    "admin": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
    "readonly": "$2b$12$2eUHV/JnNTWKhHkD4V5HEuMsZMqnqQ8K7vK9gF5xL3mN1pR2oT6Ky",
}


def _init_users_db() -> dict[str, dict]:
    """Initialise la base d'utilisateurs avec des hashes bcrypt pré-calculés."""
    return {
        "data_scientist": {
            "username": "data_scientist",
            "hashed_password": get_password_hash("mlops2024"),
            "scopes": ["predict", "batch", "model:read"],
            "disabled": False,
        },
        "admin": {
            "username": "admin",
            "hashed_password": get_password_hash("admin_secret"),
            "scopes": ["predict", "batch", "model:read", "model:write", "admin"],
            "disabled": False,
        },
        "readonly": {
            "username": "readonly",
            "hashed_password": get_password_hash("readonly123"),
            "scopes": ["model:read"],
            "disabled": False,
        },
    }


FAKE_USERS_DB: dict[str, dict] = _init_users_db()


def get_user(username: str) -> dict | None:
    """Récupère un utilisateur depuis la base."""
    return FAKE_USERS_DB.get(username)


def authenticate_user(username: str, password: str) -> dict | None:
    """Authentifie un utilisateur par username/password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    if user.get("disabled"):
        return None
    return user


# ---------------------------------------------------------------------------
# Blacklist des tokens révoqués (en mémoire — utiliser Redis en prod)
# ---------------------------------------------------------------------------

_revoked_tokens: set[str] = set()


def revoke_token(token: str) -> None:
    """Révoque un token en l'ajoutant à la blacklist."""
    _revoked_tokens.add(token)


def is_token_revoked(token: str) -> bool:
    """Vérifie si un token a été révoqué."""
    return token in _revoked_tokens


# ---------------------------------------------------------------------------
# Création des tokens JWT
# ---------------------------------------------------------------------------


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Crée un token JWT d'accès signé avec HS256.

    Args:
        data: Payload du token (doit contenir 'sub' = username).
        expires_delta: Durée de validité (défaut : ACCESS_TOKEN_EXPIRE_MINUTES).

    Returns:
        Token JWT encodé en chaîne.
    """
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "access",
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """
    Crée un token JWT de rafraîchissement (longue durée).

    Args:
        data: Payload du token (doit contenir 'sub' = username).

    Returns:
        Token JWT de refresh encodé.
    """
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update(
        {
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "refresh",
        }
    )
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str, expected_type: str = "access") -> TokenData:
    """
    Décode et valide un token JWT.

    Args:
        token: Token JWT à décoder.
        expected_type: Type attendu ('access' ou 'refresh').

    Returns:
        TokenData avec username et scopes.

    Raises:
        HTTPException 401 si le token est invalide, expiré ou révoqué.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Vérifier la blacklist
    if is_token_revoked(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token révoqué",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type", "access")

        if username is None:
            raise credentials_exception
        if token_type != expected_type:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Type de token incorrect : attendu '{expected_type}'",
                headers={"WWW-Authenticate": "Bearer"},
            )

        scopes = payload.get("scopes", [])
        return TokenData(username=username, scopes=scopes)

    except JWTError:
        raise credentials_exception


# ---------------------------------------------------------------------------
# Dépendances FastAPI
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Dépendance FastAPI : extrait et valide l'utilisateur depuis le token Bearer.

    Usage :
        @router.get("/protected")
        async def endpoint(user = Depends(get_current_user)):
            ...
    """
    token_data = decode_token(token, expected_type="access")
    user = get_user(token_data.username)
    if user is None or user.get("disabled"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur introuvable ou désactivé",
        )
    return user


async def get_current_active_user(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Dépendance : vérifie que l'utilisateur est actif (non désactivé)."""
    if current_user.get("disabled"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Utilisateur désactivé",
        )
    return current_user


def require_scope(required_scope: str):
    """
    Factory de dépendance FastAPI pour la vérification des scopes.

    Usage :
        @router.post("/predict")
        async def predict(user = Depends(require_scope("predict"))):
            ...
    """

    async def scope_checker(
        current_user: dict = Depends(get_current_active_user),
    ) -> dict:
        user_scopes = current_user.get("scopes", [])
        if required_scope not in user_scopes and "admin" not in user_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission insuffisante. Scope requis : '{required_scope}'",
            )
        return current_user

    return scope_checker
