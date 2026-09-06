import hmac

from fastapi import Header, HTTPException, status

from app.config import settings


async def require_service_token(x_pipemind_token: str = Header(default="")) -> None:
    """Constant-time shared-secret check.

    This service is never exposed publicly: it sits on the internal network and
    trusts exactly one caller, Laravel. There are no user identities here, which
    is why a single shared secret is sufficient — and why it must never be
    compared with ==.
    """
    expected = settings().service_token

    if not expected or not hmac.compare_digest(x_pipemind_token, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid service token")
