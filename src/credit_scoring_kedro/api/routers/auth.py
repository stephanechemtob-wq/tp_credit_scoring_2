"""
Router d'authentification JWT pour l'API de scoring de crédit.

Endpoints :
- POST /auth/token    : Obtenir access_token + refresh_token
- POST /auth/refresh  : Rafraîchir l'access_token via refresh_token
- POST /auth/logout   : Révoquer les tokens (blacklist)
- GET  /auth/me       : Informations sur l'utilisateur courant
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from credit_scoring_kedro.api.security.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    RefreshRequest,
    Token,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_active_user,
    revoke_token,
)

router = APIRouter(prefix="/auth", tags=["Authentification"])


@router.post(
    "/token",
    response_model=Token,
    summary="Obtenir un token JWT",
    description="""
Authentifie un utilisateur et retourne un **access_token** (30 min) et un **refresh_token** (7 jours).

**Utilisateurs de démo :**
| Username | Password | Scopes |
|---|---|---|
| `data_scientist` | `mlops2024` | predict, batch, model:read |
| `admin` | `admin_secret` | tous les scopes |
| `readonly` | `readonly123` | model:read uniquement |
""",
)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Token:
    """Authentification par username/password → JWT."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants incorrects",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_payload = {
        "sub": user["username"],
        "scopes": user["scopes"],
    }

    access_token = create_access_token(data=token_payload)
    refresh_token = create_refresh_token(data=token_payload)

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/refresh",
    response_model=Token,
    summary="Rafraîchir l'access token",
    description="Utilise le **refresh_token** pour obtenir un nouvel **access_token** sans se reconnecter.",
)
async def refresh_access_token(request: RefreshRequest) -> Token:
    """Rafraîchissement du token d'accès via le refresh token."""
    # Valider le refresh token
    token_data = decode_token(request.refresh_token, expected_type="refresh")

    # Révoquer l'ancien refresh token (rotation des tokens)
    revoke_token(request.refresh_token)

    # Créer de nouveaux tokens
    token_payload = {
        "sub": token_data.username,
        "scopes": token_data.scopes,
    }

    new_access_token = create_access_token(data=token_payload)
    new_refresh_token = create_refresh_token(data=token_payload)

    return Token(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/logout",
    summary="Révoquer les tokens (logout)",
    description="Révoque l'**access_token** et optionnellement le **refresh_token** (blacklist).",
)
async def logout(
    request: RefreshRequest,
    current_user: dict = Depends(get_current_active_user),
) -> dict:
    """Révocation des tokens — déconnexion sécurisée."""
    # Révoquer le refresh token fourni
    revoke_token(request.refresh_token)

    return {
        "message": f"Utilisateur '{current_user['username']}' déconnecté avec succès.",
        "tokens_revoked": True,
    }


@router.get(
    "/me",
    summary="Informations sur l'utilisateur courant",
    description="Retourne les informations de l'utilisateur authentifié (username, scopes).",
)
async def read_users_me(
    current_user: dict = Depends(get_current_active_user),
) -> dict:
    """Profil de l'utilisateur authentifié."""
    return {
        "username": current_user["username"],
        "scopes": current_user["scopes"],
        "disabled": current_user.get("disabled", False),
    }
