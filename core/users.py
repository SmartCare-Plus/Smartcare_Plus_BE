 """
SMARTCARE+ User Service

CRUD operations for users (elderly, caregiver, guardian).
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime, timezone
from enum import Enum

from core.database import get_db, is_mock_mode
from core.auth import verify_firebase_token, AuthenticatedUser, require_guardian

router = APIRouter()


# ============================================
# Enums and Models
# ============================================

class UserRole(str, Enum):
    ELDERLY = "elderly"
    CAREGIVER = "caregiver"
    GUARDIAN = "guardian"
    ADMIN = "admin"


class EmergencyContact(BaseModel):
    name: str
    phone: str
    relationship: str
    is_primary: bool = False


class NutritionPreferences(BaseModel):
    """Nutrition and dietary preferences for meal planning."""
    dietary_restrictions: List[str] = []  # diabetic, low_sodium, vegetarian, etc.
    food_allergies: List[str] = []
    food_dislikes: List[str] = []
    budget_level: str = "medium"  # low, medium, high
    daily_budget: Optional[float] = None  # Daily budget in currency
    calorie_target: Optional[int] = None  # Override default RDA
    preferred_cuisines: List[str] = []


class UserBase(BaseModel):
    email: EmailStr
    name: str
    phone: Optional[str] = None
    role: UserRole
    profile_image_url: Optional[str] = None


class ElderlyProfile(BaseModel):
    age: int = Field(..., ge=0, le=150)
    conditions: List[str] = []
    mobility_level: str = "independent"  # independent, needs_assistance, wheelchair
    emergency_contacts: List[EmergencyContact] = []
    address: Optional[str] = None
    notes: Optional[str] = None
    nutrition_preferences: Optional[NutritionPreferences] = None


class CaregiverProfile(BaseModel):
    specialization: Optional[str] = None
    certifications: List[str] = []
    assigned_elderly_ids: List[str] = []


class GuardianProfile(BaseModel):
    relationship: str  # son, daughter, spouse, etc.
    linked_elderly_ids: List[str] = []
    notification_preferences: dict = {
        "push_enabled": True,
        "sms_enabled": True,
        "email_enabled": True
    }


class UserCreate(UserBase):
    elderly_profile: Optional[ElderlyProfile] = None
    caregiver_profile: Optional[CaregiverProfile] = None
    guardian_profile: Optional[GuardianProfile] = None


class UserResponse(UserBase):
    uid: str
    created_at: str
    updated_at: str
    elderly_profile: Optional[ElderlyProfile] = None
    caregiver_profile: Optional[CaregiverProfile] = None
    guardian_profile: Optional[GuardianProfile] = None


class UserUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    profile_image_url: Optional[str] = None
    elderly_profile: Optional[ElderlyProfile] = None
    caregiver_profile: Optional[CaregiverProfile] = None
    guardian_profile: Optional[GuardianProfile] = None


# ============================================
# Mock Data Store (for development)
# ============================================

_mock_users = {
    "mock_user_001": {
        "uid": "mock_user_001",
        "email": "demo@smartcare.plus",
        "name": "Demo Guardian",
        "phone": "+94771234567",
        "role": "guardian",
        "profile_image_url": None,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "guardian_profile": {
            "relationship": "daughter",
            "linked_elderly_ids": ["elder_001"],
            "notification_preferences": {
                "push_enabled": True,
                "sms_enabled": True,
                "email_enabled": True
            }
        }
    },
    "elder_001": {
        "uid": "elder_001",
        "email": "elder@smartcare.plus",
        "name": "Mrs. Silva",
        "phone": "+94777654321",
        "role": "elderly",
        "profile_image_url": None,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "elderly_profile": {
            "age": 78,
            "conditions": ["hypertension", "arthritis"],
            "mobility_level": "needs_assistance",
            "emergency_contacts": [
                {"name": "Nimali", "phone": "+94771234567", "relationship": "daughter", "is_primary": True}
            ],
            "address": "123 Temple Road, Colombo",
            "notes": "Prefers morning medication"
        }
    }
}


# ============================================
# Helper Functions
# ============================================

def get_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================
# Endpoints
# ============================================

@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """
    Create a new user profile.
    The UID comes from Firebase Auth.
    """
    db = get_db()
    now = get_now_iso()
    
    user_dict = user_data.model_dump()
    user_dict["uid"] = current_user.uid
    user_dict["created_at"] = now
    user_dict["updated_at"] = now
    
    if is_mock_mode():
        _mock_users[current_user.uid] = user_dict
        return UserResponse(**user_dict)
    
    # Store in Firestore
    users_ref = db.collection("users")
    users_ref.document(current_user.uid).set(user_dict)
    
    return UserResponse(**user_dict)


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """Get the current authenticated user's profile."""
    db = get_db()
    
    if is_mock_mode():
        if current_user.uid in _mock_users:
            return UserResponse(**_mock_users[current_user.uid])
        raise HTTPException(status_code=404, detail="User profile not found")
    
    # Fetch from Firestore
    user_doc = db.collection("users").document(current_user.uid).get()
    
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    return UserResponse(**user_doc.to_dict())


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """Get a user by ID. Guardians can view their linked elderly."""
    db = get_db()
    
    if is_mock_mode():
        if user_id in _mock_users:
            return UserResponse(**_mock_users[user_id])
        raise HTTPException(status_code=404, detail="User not found")
    
    user_doc = db.collection("users").document(user_id).get()
    
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(**user_doc.to_dict())


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    update_data: UserUpdate,
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """Update the current user's profile."""
    db = get_db()
    now = get_now_iso()
    
    # Get existing user
    if is_mock_mode():
        if current_user.uid not in _mock_users:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Update only provided fields
        update_dict = update_data.model_dump(exclude_unset=True)
        update_dict["updated_at"] = now
        _mock_users[current_user.uid].update(update_dict)
        return UserResponse(**_mock_users[current_user.uid])
    
    user_ref = db.collection("users").document(current_user.uid)
    user_doc = user_ref.get()
    
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    # Update
    update_dict = update_data.model_dump(exclude_unset=True)
    update_dict["updated_at"] = now
    user_ref.update(update_dict)
    
    # Fetch updated document
    updated_doc = user_ref.get()
    return UserResponse(**updated_doc.to_dict())


@router.delete("/me")
async def delete_current_user(
    current_user: AuthenticatedUser = Depends(verify_firebase_token)
):
    """Delete the current user's profile."""
    db = get_db()
    
    if is_mock_mode():
        if current_user.uid in _mock_users:
            del _mock_users[current_user.uid]
        return {"message": "User deleted successfully"}
    
    db.collection("users").document(current_user.uid).delete()
    return {"message": "User deleted successfully"}


@router.get("/elderly/list", response_model=List[UserResponse])
async def list_linked_elderly(
    current_user: AuthenticatedUser = Depends(require_guardian)
):
    """Get list of elderly users linked to the current guardian."""
    db = get_db()
    
    if is_mock_mode():
        # Return elderly users from mock data
        return [
            UserResponse(**user) 
            for user in _mock_users.values() 
            if user.get("role") == "elderly"
        ]
    
    # Get guardian's profile
    guardian_doc = db.collection("users").document(current_user.uid).get()
    
    if not guardian_doc.exists:
        return []
    
    guardian_data = guardian_doc.to_dict()
    guardian_profile = guardian_data.get("guardian_profile", {})
    linked_ids = guardian_profile.get("linked_elderly_ids", [])
    
    # Fetch linked elderly
    elderly_list = []
    for elderly_id in linked_ids:
        elderly_doc = db.collection("users").document(elderly_id).get()
        if elderly_doc.exists:
            elderly_list.append(UserResponse(**elderly_doc.to_dict()))
    
    return elderly_list


@router.post("/link-elderly/{elderly_id}")
async def link_elderly_to_guardian(
    elderly_id: str,
    current_user: AuthenticatedUser = Depends(require_guardian)
):
    """Link an elderly user to the current guardian."""
    db = get_db()
    now = get_now_iso()
    
    if is_mock_mode():
        return {"message": f"Elderly {elderly_id} linked successfully"}
    
    # Verify elderly exists
    elderly_doc = db.collection("users").document(elderly_id).get()
    if not elderly_doc.exists:
        raise HTTPException(status_code=404, detail="Elderly user not found")
    
    elderly_data = elderly_doc.to_dict()
    if elderly_data.get("role") != "elderly":
        raise HTTPException(status_code=400, detail="User is not an elderly profile")
    
    # Update guardian's linked elderly
    guardian_ref = db.collection("users").document(current_user.uid)
    guardian_doc = guardian_ref.get()
    
    if not guardian_doc.exists:
        raise HTTPException(status_code=404, detail="Guardian profile not found")
    
    guardian_data = guardian_doc.to_dict()
    guardian_profile = guardian_data.get("guardian_profile", {})
    linked_ids = guardian_profile.get("linked_elderly_ids", [])
    
    if elderly_id not in linked_ids:
        linked_ids.append(elderly_id)
        guardian_ref.update({
            "guardian_profile.linked_elderly_ids": linked_ids,
            "updated_at": now
        })
    
    return {"message": f"Elderly {elderly_id} linked successfully"}
