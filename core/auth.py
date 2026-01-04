"""
SMARTCARE+ Firebase Authentication Middleware

Verifies Firebase ID tokens for protected endpoints.
"""

import logging
import sys
import time
from typing import Optional
from functools import wraps

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from firebase_admin import auth
from core.database import init_firebase, is_mock_mode

# Configure detailed logging
logger = logging.getLogger("smartcare.auth")
logger.setLevel(logging.DEBUG)

# Ensure handler if not present
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# HTTP Bearer scheme for token extraction
security = HTTPBearer(auto_error=False)


class AuthenticatedUser:
    """Represents an authenticated Firebase user."""
    
    def __init__(
        self,
        uid: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        role: Optional[str] = None,
        email_verified: bool = False,
        custom_claims: Optional[dict] = None
    ):
        self.uid = uid
        self.email = email
        self.name = name
        self.role = role or "user"
        self.email_verified = email_verified
        self.custom_claims = custom_claims or {}
    
    def __repr__(self):
        return f"AuthenticatedUser(uid={self.uid}, email={self.email}, role={self.role})"
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return self.role == role or role in self.custom_claims.get("roles", [])
    
    def is_guardian(self) -> bool:
        return self.has_role("guardian")
    
    def is_elderly(self) -> bool:
        return self.has_role("elderly")
    
    def is_caregiver(self) -> bool:
        return self.has_role("caregiver")


# Mock user for development without Firebase
MOCK_USER = AuthenticatedUser(
    uid="mock_user_001",
    email="demo@smartcare.plus",
    name="Demo User",
    role="guardian",
    email_verified=True
)


async def verify_firebase_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthenticatedUser:
    """
    Verify Firebase ID token from Authorization header.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: AuthenticatedUser = Depends(verify_firebase_token)):
            return {"message": f"Hello {user.email}"}
    """
    start_time = time.time()
    logger.debug(f"🔐 Token verification started")
    # Initialize Firebase if not already done
    init_firebase()
    
    # Mock mode - return demo user
    if is_mock_mode():
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"🧪 MOCK MODE: Returning demo user (uid={MOCK_USER.uid}) in {elapsed:.1f}ms")
        return MOCK_USER
    
    # Check for credentials
    if credentials is None:
        logger.warning(f"❌ No authorization header provided")
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    token_preview = token[:20] + "..." if len(token) > 20 else token
    logger.debug(f"🔑 Token received: {token_preview}")
    
    try:
        # Verify the token with Firebase
        logger.debug(f"🔥 Verifying token with Firebase...")
        verify_start = time.time()
        decoded_token = auth.verify_id_token(token)
        verify_time = (time.time() - verify_start) * 1000
        logger.debug(f"✅ Token verified in {verify_time:.1f}ms")
        
        # Extract user info
        user = AuthenticatedUser(
            uid=decoded_token.get("uid"),
            email=decoded_token.get("email"),
            name=decoded_token.get("name"),
            role=decoded_token.get("role", "user"),
            email_verified=decoded_token.get("email_verified", False),
            custom_claims=decoded_token
        )
        
        logger.debug(f"✅ Authenticated: {user}")
        return user
        
    except auth.ExpiredIdTokenError:
        logger.warning("Token expired")
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    except auth.RevokedIdTokenError:
        logger.warning("Token revoked")
        raise HTTPException(
            status_code=401,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    except auth.InvalidIdTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authentication service error"
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[AuthenticatedUser]:
    """
    Get user if authenticated, None otherwise.
    Useful for endpoints that work differently for logged-in users.
    """
    if credentials is None:
        return None
    
    try:
        return await verify_firebase_token(credentials)
    except HTTPException:
        return None


def require_role(*allowed_roles: str):
    """
    Dependency factory for role-based access control.
    
    Usage:
        @app.get("/admin-only")
        async def admin_route(user: AuthenticatedUser = Depends(require_role("admin", "guardian"))):
            return {"message": "Welcome admin!"}
    """
    async def role_checker(
        user: AuthenticatedUser = Depends(verify_firebase_token)
    ) -> AuthenticatedUser:
        if user.role not in allowed_roles and not any(
            user.has_role(role) for role in allowed_roles
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required roles: {allowed_roles}"
            )
        return user
    
    return role_checker


# Convenience dependencies for common roles
require_guardian = require_role("guardian", "admin")
require_elderly = require_role("elderly", "caregiver", "guardian", "admin")
require_caregiver = require_role("caregiver", "guardian", "admin")
