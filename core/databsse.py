"""
SMARTCARE+ Firebase Database Initialization

Initializes Firebase Admin SDK for Firestore access.
Supports mock mode when credentials are unavailable.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore

from core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Global Firestore client
_db: Optional[firestore.Client] = None
_mock_mode: bool = False


def init_firebase() -> bool:
    """
    Initialize Firebase Admin SDK.
    
    Returns:
        bool: True if connected successfully, False if running in mock mode.
    """
    global _db, _mock_mode
    
    # Already initialized?
    if _db is not None:
        return not _mock_mode
    
    # Get credentials path from settings
    cred_path = Path(__file__).parent.parent / settings.FIREBASE_CREDENTIALS_PATH
    
    # Check if credentials file exists
    if not cred_path.exists():
        logger.warning(
            f"⚠️ Firebase credentials not found at '{cred_path}'. "
            "Running in MOCK MODE - database operations will be simulated."
        )
        _mock_mode = True
        return False
    
    try:
        # Initialize Firebase Admin SDK
        cred = credentials.Certificate(str(cred_path))
        
        # Check if already initialized (happens during hot reload)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'projectId': settings.FIREBASE_PROJECT_ID,
            })
            logger.info(f"🔥 Firebase Admin SDK initialized for project: {settings.FIREBASE_PROJECT_ID}")
        
        # Get Firestore client
        _db = firestore.client()
        logger.info("✅ Connected to Firestore successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Firebase: {e}")
        logger.warning("Running in MOCK MODE - database operations will be simulated.")
        _mock_mode = True
        return False


def get_db() -> Optional[firestore.Client]:
    """
    Get Firestore database client.
    
    Returns:
        Firestore client or None if in mock mode.
    """
    global _db
    
    if _db is None and not _mock_mode:
        init_firebase()
    
    return _db


def is_mock_mode() -> bool:
    """Check if running in mock mode (no Firebase connection)."""
    return _mock_mode


# ============================================
# Mock Database for Development/Testing
# ============================================

class MockFirestoreClient:
    """
    Mock Firestore client for development without Firebase.
    Stores data in memory.
    """
    
    def __init__(self):
        self._collections: dict = {}
        logger.info("🧪 MockFirestoreClient initialized (in-memory storage)")
    
    def collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = MockCollection(name)
        return self._collections[name]


class MockCollection:
    """Mock Firestore collection."""
    
    def __init__(self, name: str):
        self.name = name
        self._documents: dict = {}
    
    def document(self, doc_id: str = None):
        if doc_id is None:
            doc_id = f"mock_{len(self._documents) + 1}"
        if doc_id not in self._documents:
            self._documents[doc_id] = MockDocument(doc_id, self)
        return self._documents[doc_id]
    
    def add(self, data: dict):
        doc_id = f"mock_{len(self._documents) + 1}"
        doc = self.document(doc_id)
        doc.set(data)
        return (None, doc)
    
    def get(self):
        return [doc for doc in self._documents.values()]
    
    def where(self, field: str, op: str, value):
        # Simple mock filtering
        return self


class MockDocument:
    """Mock Firestore document."""
    
    def __init__(self, doc_id: str, collection: MockCollection):
        self.id = doc_id
        self._collection = collection
        self._data: dict = {}
        self.exists = False
    
    def set(self, data: dict, merge: bool = False):
        if merge:
            self._data.update(data)
        else:
            self._data = data.copy()
        self.exists = True
    
    def update(self, data: dict):
        self._data.update(data)
    
    def get(self):
        return self
    
    def to_dict(self):
        return self._data.copy()
    
    def delete(self):
        self._data = {}
        self.exists = False


def get_mock_db() -> MockFirestoreClient:
    """Get mock database client for testing."""
    return MockFirestoreClient()


# ============================================
# Database Helper Functions
# ============================================

def get_database():
    """
    Get database client (real or mock).
    Use this in your services to automatically handle mock mode.
    """
    if _mock_mode:
        return get_mock_db()
    return get_db()
