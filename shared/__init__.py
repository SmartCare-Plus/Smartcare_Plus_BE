"""
SMARTCARE+ Shared Module

Common utilities used across all services.
"""

from .storage import LocalStorageManager, get_storage, StoredFile

__all__ = [
    'LocalStorageManager',
    'get_storage',
    'StoredFile',
]
