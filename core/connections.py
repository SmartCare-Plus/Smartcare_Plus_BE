"""
SMARTCARE+ Connection Service

Manages connections between elderly users and their guardians/caregivers.
Supports invite codes for easy linking.
"""

import secrets
import string
import logging
import traceback
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

# Configure logger with detailed formatting
logger = logging.getLogger("smartcare.connections")
logger.setLevel(logging.DEBUG)

# Ensure handler if not present
if not logger.handlers:
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from core.database import get_db, is_mock_mode
from core.auth import verify_firebase_token, AuthenticatedUser

router = APIRouter()


# ============================================
# Enums and Models
# ============================================

class ConnectionType(str, Enum):
    GUARDIAN = "guardian"
    CAREGIVER = "caregiver"


class ConnectionStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    REJECTED = "rejected"


class InviteCode(BaseModel):
    code: str
    created_by: str
    created_by_name: str
    created_by_role: str
    created_at: str
    expires_at: str
    used: bool = False
    used_by: Optional[str] = None


class ConnectionRequest(BaseModel):
    connection_type: ConnectionType


class JoinRequest(BaseModel):
    invite_code: str


class Connection(BaseModel):
    id: str
    elderly_id: str
    elderly_name: str
    linked_user_id: str
    linked_user_name: str
    connection_type: ConnectionType
    status: ConnectionStatus
    created_at: str


class ConnectionResponse(BaseModel):
    success: bool
    message: str
    connection: Optional[Connection] = None


# ============================================
# Mock Data Store (for development)
# ============================================

_mock_invite_codes: dict = {}
_mock_connections: List[dict] = []


# ============================================
# Helper Functions
# ============================================

def get_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_invite_code() -> str:
    """Generate a 6-character alphanumeric invite code."""
    chars = string.ascii_uppercase + string.digits
    # Exclude confusing characters
    chars = chars.replace('O', '').replace('0', '').replace('I', '').replace('1', '')
    return ''.join(secrets.choice(chars) for _ in range(6))


async def get_user_info(user_id: str) -> dict:
    """Get user name and role from database."""
    db = get_db()
    
    if is_mock_mode():
        # Return mock data
        return {
            "name": "Demo User",
            "role": "elderly"
        }
    
    user_doc = db.collection("users").document(user_id).get()
    if user_doc.exists:
        data = user_doc.to_dict()
        return {
            "name": data.get("name", "Unknown"),
            "role": data.get("role", "unknown")
        }
    return {"name": "Unknown", "role": "unknown"}


# ============================================
# Endpoints
# ============================================

@router.post("/generate-code")
async def generate_invite_code_endpoint(
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """
    Generate an invite code for the current user.
    - Elderly users generate codes to invite guardians/caregivers
    - Guardians/Caregivers generate codes to be added by elderly
    """
    start_time = time.time()
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"🔑 GENERATE INVITE CODE REQUEST")
    logger.info(f"{'='*60}")
    logger.info(f"  👤 User ID: {current_user.uid}")
    logger.info(f"  📧 Email: {current_user.email}")
    logger.info(f"  🎭 Role: {current_user.role}")
    
    db = get_db()
    logger.debug(f"  📦 Database mode: {'Firebase' if not is_mock_mode() else 'MOCK'}")
    now = datetime.now(timezone.utc)
    expires = now + timedelta(hours=24)
    
    code = generate_invite_code()
    logger.info(f"  🎟️ Generated code: {code}")
    
    # Get user info
    user_info = await get_user_info(current_user.uid)
    logger.debug(f"  👤 User info: {user_info}")
    
    invite_data = {
        "code": code,
        "created_by": current_user.uid,
        "created_by_name": user_info["name"],
        "created_by_role": user_info["role"],
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "used": False,
        "used_by": None
    }
    
    if is_mock_mode():
        _mock_invite_codes[code] = invite_data
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"  ✅ Code {code} created successfully (MOCK)")
        logger.info(f"  📋 Current mock codes: {list(_mock_invite_codes.keys())}")
        logger.info(f"  ⏱️ Response time: {elapsed:.1f}ms")
        logger.info(f"{'='*60}")
        return {
            "success": True,
            "code": code,
            "expires_at": expires.isoformat(),
            "message": f"Share this code with {'your guardian or caregiver' if user_info['role'] == 'elderly' else 'the elderly person you want to care for'}"
        }
    
    # Store in Firestore
    db.collection("invite_codes").document(code).set(invite_data)
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"  ✅ Code {code} created successfully (Firestore)")
    logger.info(f"  ⏱️ Response time: {elapsed:.1f}ms")
    logger.info(f"{'='*60}")
    
    return {
        "success": True,
        "code": code,
        "expires_at": expires.isoformat(),
        "message": f"Share this code with {'your guardian or caregiver' if user_info['role'] == 'elderly' else 'the elderly person you want to care for'}"
    }


@router.post("/join")
async def join_with_code(
    request: JoinRequest,
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """
    Join/link with another user using their invite code.
    Creates a connection between the code creator and the current user.
    """
    start_time = time.time()
    code = request.invite_code.upper().strip()
    
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"🔗 JOIN WITH CODE REQUEST")
    logger.info(f"{'='*60}")
    logger.info(f"  👤 User ID: {current_user.uid}")
    logger.info(f"  📧 Email: {current_user.email}")
    logger.info(f"  🎭 Role: {current_user.role}")
    logger.info(f"  🔑 Invite Code: {code}")
    logger.info(f"  🕐 Timestamp: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"{'='*60}")
    
    try:
        db = get_db()
        logger.debug(f"  📦 Database mode: {'Firebase' if not is_mock_mode() else 'MOCK'}")
        now = get_now_iso()
    
        # Get invite code data
        logger.info(f"  🔍 Looking up invite code: {code}")
        lookup_start = time.time()
        
        if is_mock_mode():
            logger.debug(f"  🧪 Mock mode - checking _mock_invite_codes")
            logger.debug(f"  🧪 Available codes: {list(_mock_invite_codes.keys())}")
            if code not in _mock_invite_codes:
                logger.warning(f"  ❌ Code {code} not found in mock store")
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"  ⏱️ Request failed in {elapsed:.1f}ms")
                raise HTTPException(status_code=404, detail="Invalid or expired invite code")
            invite_data = _mock_invite_codes[code]
            logger.info(f"  ✅ Code found in mock store")
        else:
            logger.debug(f"  🔥 Firebase mode - querying Firestore")
            code_doc = db.collection("invite_codes").document(code).get()
            lookup_time = (time.time() - lookup_start) * 1000
            logger.debug(f"  ⏱️ Firestore lookup took {lookup_time:.1f}ms")
            
            if not code_doc.exists:
                logger.warning(f"  ❌ Code {code} not found in Firestore")
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"  ⏱️ Request failed in {elapsed:.1f}ms")
                raise HTTPException(status_code=404, detail="Invalid or expired invite code")
            invite_data = code_doc.to_dict()
            logger.info(f"  ✅ Code found in Firestore")
        
        logger.debug(f"  📄 Invite data: created_by={invite_data.get('created_by')}, role={invite_data.get('created_by_role')}")
    
        # Validate code
        logger.info(f"  🔍 Validating invite code...")
        if invite_data.get("used"):
            logger.warning(f"  ❌ Code already used")
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"  ⏱️ Request failed in {elapsed:.1f}ms")
            raise HTTPException(status_code=400, detail="This invite code has already been used")
    
        expires_at = datetime.fromisoformat(invite_data["expires_at"].replace('Z', '+00:00'))
        if datetime.now(timezone.utc) > expires_at:
            logger.warning(f"  ❌ Code expired at {expires_at}")
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"  ⏱️ Request failed in {elapsed:.1f}ms")
            raise HTTPException(status_code=400, detail="This invite code has expired")
    
        if invite_data["created_by"] == current_user.uid:
            logger.warning(f"  ❌ User trying to use own code")
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"  ⏱️ Request failed in {elapsed:.1f}ms")
            raise HTTPException(status_code=400, detail="You cannot use your own invite code")
        
        logger.info(f"  ✅ Code validation passed")
    
        # Get both users' info
        logger.info(f"  👥 Fetching user information...")
        user_info_start = time.time()
        creator_info = await get_user_info(invite_data["created_by"])
        logger.debug(f"  👤 Creator info: {creator_info}")
        joiner_info = await get_user_info(current_user.uid)
        logger.debug(f"  👤 Joiner info: {joiner_info}")
        user_info_time = (time.time() - user_info_start) * 1000
        logger.debug(f"  ⏱️ User info lookup took {user_info_time:.1f}ms")
    
        # Determine connection structure
        creator_role = invite_data["created_by_role"]
        joiner_role = joiner_info["role"]
    
        # Determine who is elderly and who is guardian/caregiver
        if creator_role == "elderly":
            elderly_id = invite_data["created_by"]
            elderly_name = invite_data["created_by_name"]
            linked_user_id = current_user.uid
            linked_user_name = joiner_info["name"]
            connection_type = ConnectionType.GUARDIAN if joiner_role == "guardian" else ConnectionType.CAREGIVER
        else:
            # Creator is guardian/caregiver, joiner should be elderly
            if joiner_role != "elderly":
                raise HTTPException(
                    status_code=400, 
                    detail="Only elderly users can join using a guardian/caregiver's code"
                )
            elderly_id = current_user.uid
            elderly_name = joiner_info["name"]
            linked_user_id = invite_data["created_by"]
            linked_user_name = invite_data["created_by_name"]
            connection_type = ConnectionType.GUARDIAN if creator_role == "guardian" else ConnectionType.CAREGIVER
    
        # Create connection
        connection_id = f"{elderly_id}_{linked_user_id}"
        connection_data = {
            "id": connection_id,
            "elderly_id": elderly_id,
            "elderly_name": elderly_name,
            "linked_user_id": linked_user_id,
            "linked_user_name": linked_user_name,
            "connection_type": connection_type.value,
            "status": ConnectionStatus.ACTIVE.value,
            "created_at": now
        }
    
        if is_mock_mode():
            # Check for duplicate
            for conn in _mock_connections:
                if conn["id"] == connection_id:
                    raise HTTPException(status_code=400, detail="Connection already exists")
        
            _mock_connections.append(connection_data)
            _mock_invite_codes[code]["used"] = True
            _mock_invite_codes[code]["used_by"] = current_user.uid
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"")
            logger.info(f"  ✅ CONNECTION CREATED SUCCESSFULLY (MOCK)")
            logger.info(f"  🔗 Connection ID: {connection_id}")
            logger.info(f"  👴 Elderly: {elderly_name} ({elderly_id})")
            logger.info(f"  🛡️ Guardian: {linked_user_name} ({linked_user_id})")
            logger.info(f"  ⏱️ Total time: {elapsed:.1f}ms")
            logger.info(f"{'='*60}")
            logger.info(f"")
        
            return {
                "success": True,
                "message": f"Successfully connected with {invite_data['created_by_name']}!",
                "connection": connection_data
            }
    
        # Check for existing connection
        existing = db.collection("connections").document(connection_id).get()
        if existing.exists:
            raise HTTPException(status_code=400, detail="Connection already exists")
    
        # Save connection
        db.collection("connections").document(connection_id).set(connection_data)
    
        # Mark code as used
        db.collection("invite_codes").document(code).update({
            "used": True,
            "used_by": current_user.uid
        })
    
        # Update guardian's linked_elderly_ids if applicable
        if connection_type == ConnectionType.GUARDIAN:
            guardian_ref = db.collection("users").document(linked_user_id)
            guardian_doc = guardian_ref.get()
            if guardian_doc.exists:
                guardian_data = guardian_doc.to_dict()
                guardian_profile = guardian_data.get("guardian_profile", {})
                linked_ids = guardian_profile.get("linked_elderly_ids", [])
                if elderly_id not in linked_ids:
                    linked_ids.append(elderly_id)
                    guardian_ref.update({
                        "guardian_profile.linked_elderly_ids": linked_ids,
                        "updated_at": now
                    })
                    logger.debug(f"  📝 Updated guardian's linked_elderly_ids")
    
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"")
        logger.info(f"  ✅ CONNECTION CREATED SUCCESSFULLY (FIREBASE)")
        logger.info(f"  🔗 Connection ID: {connection_id}")
        logger.info(f"  👴 Elderly: {elderly_name} ({elderly_id})")
        logger.info(f"  🛡️ Guardian: {linked_user_name} ({linked_user_id})")
        logger.info(f"  ⏱️ Total time: {elapsed:.1f}ms")
        logger.info(f"{'='*60}")
        logger.info(f"")
        
        return {
            "success": True,
            "message": f"Successfully connected with {invite_data['created_by_name']}!",
            "connection": connection_data
        }
    
    except HTTPException as e:
        # Re-raise HTTP exceptions as-is
        elapsed = (time.time() - start_time) * 1000
        logger.warning(f"  ⚠️ HTTP Exception: {e.status_code} - {e.detail} ({elapsed:.1f}ms)")
        raise
    except KeyError as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"❌ KeyError in join_with_code ({elapsed:.1f}ms): {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Data processing error", "message": f"Missing required field: {e}"}
        )
    except ValueError as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"❌ ValueError in join_with_code ({elapsed:.1f}ms): {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid data", "message": str(e)}
        )
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"❌ Unexpected error in join_with_code ({elapsed:.1f}ms): {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Internal server error", "message": str(e)}
        )


@router.get("/my-connections")
async def get_my_connections(
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """
    Get all connections for the current user.
    Works for both elderly and guardians/caregivers.
    """
    start_time = time.time()
    logger.info(f"🔗 GET /my-connections for user: {current_user.uid}")
    
    db = get_db()
    
    if is_mock_mode():
        # Filter connections for current user - return only actual connections
        connections = []
        for conn in _mock_connections:
            if conn["elderly_id"] == current_user.uid or conn["linked_user_id"] == current_user.uid:
                connections.append(conn)
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"  ✅ Found {len(connections)} connections (MOCK) in {elapsed:.1f}ms")
        # Return actual connections only - no mock data injection
        return {"connections": connections}
    
    # Query connections where user is elderly or linked_user
    connections = []
    
    # As elderly
    elderly_query = db.collection("connections").where("elderly_id", "==", current_user.uid).stream()
    for doc in elderly_query:
        connections.append(doc.to_dict())
    
    # As guardian/caregiver
    linked_query = db.collection("connections").where("linked_user_id", "==", current_user.uid).stream()
    for doc in linked_query:
        connections.append(doc.to_dict())
    
    return {"connections": connections}


@router.get("/my-elderly")
async def get_my_elderly(
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """Get list of elderly users connected to the current guardian/caregiver."""
    start_time = time.time()
    logger.info(f"👴 GET /my-elderly for user: {current_user.uid}")
    
    db = get_db()
    
    if is_mock_mode():
        elderly_list = []
        for conn in _mock_connections:
            if conn["linked_user_id"] == current_user.uid and conn["status"] == "active":
                elderly_list.append({
                    "id": conn["elderly_id"],
                    "name": conn["elderly_name"],
                    "connection_type": conn["connection_type"],
                    "connected_at": conn["created_at"]
                })
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"  ✅ Found {len(elderly_list)} elderly users (MOCK) in {elapsed:.1f}ms")
        # Return actual connections only - no mock data injection
        return {"elderly": elderly_list}
    
    # Query connections
    query = db.collection("connections").where("linked_user_id", "==", current_user.uid).where("status", "==", "active").stream()
    
    elderly_list = []
    for doc in query:
        conn = doc.to_dict()
        elderly_list.append({
            "id": conn["elderly_id"],
            "name": conn["elderly_name"],
            "connection_type": conn["connection_type"],
            "connected_at": conn["created_at"]
        })
    
    return {"elderly": elderly_list}


@router.get("/my-caregivers")
async def get_my_caregivers(
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """Get list of guardians/caregivers connected to the current elderly user."""
    db = get_db()
    
    if is_mock_mode():
        caregiver_list = []
        for conn in _mock_connections:
            if conn["elderly_id"] == current_user.uid and conn["status"] == "active":
                caregiver_list.append({
                    "id": conn["linked_user_id"],
                    "name": conn["linked_user_name"],
                    "type": conn["connection_type"],
                    "connected_at": conn["created_at"]
                })
        
        # Return actual connections only - no mock data injection
        return {"caregivers": caregiver_list}
    
    # Query connections
    query = db.collection("connections").where("elderly_id", "==", current_user.uid).where("status", "==", "active").stream()
    
    caregiver_list = []
    for doc in query:
        conn = doc.to_dict()
        caregiver_list.append({
            "id": conn["linked_user_id"],
            "name": conn["linked_user_name"],
            "type": conn["connection_type"],
            "connected_at": conn["created_at"]
        })
    
    return {"caregivers": caregiver_list}


@router.delete("/{connection_id}")
async def delete_connection(
    connection_id: str,
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """Remove a connection. Both parties can remove the connection."""
    db = get_db()
    
    if is_mock_mode():
        for i, conn in enumerate(_mock_connections):
            if conn["id"] == connection_id:
                if conn["elderly_id"] == current_user.uid or conn["linked_user_id"] == current_user.uid:
                    _mock_connections.pop(i)
                    return {"success": True, "message": "Connection removed successfully"}
                else:
                    raise HTTPException(status_code=403, detail="Not authorized to remove this connection")
        raise HTTPException(status_code=404, detail="Connection not found")
    
    # Get connection
    conn_doc = db.collection("connections").document(connection_id).get()
    if not conn_doc.exists:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    conn = conn_doc.to_dict()
    if conn["elderly_id"] != current_user.uid and conn["linked_user_id"] != current_user.uid:
        raise HTTPException(status_code=403, detail="Not authorized to remove this connection")
    
    # Delete connection
    db.collection("connections").document(connection_id).delete()
    
    return {"success": True, "message": "Connection removed successfully"}
