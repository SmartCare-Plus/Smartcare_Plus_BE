"""
SMARTCARE+ Shared Utilities

Logging, validators, and response models.
"""

import logging
import sys
from typing import Any, Optional, TypeVar, Generic
from datetime import datetime, timezone
from functools import wraps
import time

from pydantic import BaseModel
from fastapi import HTTPException


# ============================================
# Logging Configuration
# ============================================

def setup_logger(name: str = "smartcare", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a configured logger with console output.
    
    Usage:
        logger = setup_logger(__name__)
        logger.info("Hello from SMARTCARE+")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# Default logger for imports
logger = setup_logger("smartcare")


# ============================================
# Response Models
# ============================================

T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[T] = None
    timestamp: str = ""
    
    def __init__(self, **data):
        if "timestamp" not in data or not data["timestamp"]:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        super().__init__(**data)


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[dict] = None
    timestamp: str = ""
    
    def __init__(self, **data):
        if "timestamp" not in data or not data["timestamp"]:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()
        super().__init__(**data)


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated list response."""
    success: bool = True
    data: list[T]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


def success_response(data: Any = None, message: str = "Success") -> dict:
    """Create a success response dict."""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def error_response(error: str, error_code: str = None, details: dict = None) -> dict:
    """Create an error response dict."""
    return {
        "success": False,
        "error": error,
        "error_code": error_code,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================
# Validators
# ============================================

def validate_phone(phone: str) -> bool:
    """Validate phone number format."""
    import re
    # Basic international phone format
    pattern = r'^\+?[1-9]\d{1,14}$'
    return bool(re.match(pattern, phone.replace(" ", "").replace("-", "")))


def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_latitude(lat: float) -> bool:
    """Validate latitude range."""
    return -90 <= lat <= 90


def validate_longitude(lng: float) -> bool:
    """Validate longitude range."""
    return -180 <= lng <= 180


def validate_coordinates(lat: float, lng: float) -> bool:
    """Validate both latitude and longitude."""
    return validate_latitude(lat) and validate_longitude(lng)


# ============================================
# Decorators
# ============================================

def log_execution_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} executed in {elapsed:.2f}ms")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"{func.__name__} executed in {elapsed:.2f}ms")
        return result
    
    if hasattr(func, '__wrapped__'):
        return async_wrapper
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def handle_exceptions(func):
    """Decorator to catch exceptions and return proper HTTP errors."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unhandled error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unhandled error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# ============================================
# Utility Functions
# ============================================

def get_now() -> datetime:
    """Get current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def get_now_iso() -> str:
    """Get current UTC datetime as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(iso_string: str) -> datetime:
    """Parse ISO datetime string to datetime object."""
    return datetime.fromisoformat(iso_string.replace('Z', '+00:00'))


def safe_get(obj: dict, *keys, default=None):
    """Safely get nested dictionary value."""
    result = obj
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, default)
        else:
            return default
    return result


def chunks(lst: list, n: int):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
