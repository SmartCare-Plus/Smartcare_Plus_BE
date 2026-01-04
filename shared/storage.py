"""
SMARTCARE+ Storage Manager

Local file storage implementation (Firebase Storage fallback).
Handles video and image uploads to local filesystem.
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, BinaryIO, Tuple
from dataclasses import dataclass

from core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class StoredFile:
    """Represents a stored file with metadata."""
    file_id: str
    filename: str
    path: str
    url: str
    size_bytes: int
    content_type: str
    created_at: datetime


class LocalStorageManager:
    """
    Local file storage manager.
    
    Stores uploaded files to backend/media/ folder.
    Provides URLs for serving via FastAPI static files.
    """
    
    # Supported file types
    ALLOWED_VIDEO_TYPES = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}
    ALLOWED_IMAGE_TYPES = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    
    # Max file sizes (in bytes)
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
    MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB
    
    def __init__(self, base_path: str = None, base_url: str = "http://localhost:8000"):
        """
        Initialize local storage manager.
        
        Args:
            base_path: Base directory for file storage. Defaults to backend/media/
            base_url: Base URL for serving files. Defaults to localhost:8000
        """
        if base_path is None:
            # Default to backend/media/
            self.base_path = Path(__file__).parent.parent / "media"
        else:
            self.base_path = Path(base_path)
        
        self.base_url = base_url.rstrip('/')
        
        # Create subdirectories
        self._ensure_directories()
        
        logger.info(f"📁 LocalStorageManager initialized at: {self.base_path}")
    
    def _ensure_directories(self):
        """Create required directory structure."""
        directories = [
            self.base_path,
            self.base_path / "videos",
            self.base_path / "images",
            self.base_path / "temp",
            self.base_path / "physio",      # For gait analysis videos
            self.base_path / "nutrition",   # For food photos
            self.base_path / "guardian",    # For monitoring videos
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def _generate_file_id(self) -> str:
        """Generate unique file ID."""
        return str(uuid.uuid4())
    
    def _get_content_type(self, filename: str) -> str:
        """Determine content type from filename."""
        ext = Path(filename).suffix.lower()
        
        content_types = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        
        return content_types.get(ext, 'application/octet-stream')
    
    def _validate_file(self, filename: str, size: int, file_type: str = "video") -> Tuple[bool, str]:
        """
        Validate file before saving.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        ext = Path(filename).suffix.lower()
        
        if file_type == "video":
            if ext not in self.ALLOWED_VIDEO_TYPES:
                return False, f"Invalid video type. Allowed: {self.ALLOWED_VIDEO_TYPES}"
            if size > self.MAX_VIDEO_SIZE:
                return False, f"Video too large. Max size: {self.MAX_VIDEO_SIZE / 1024 / 1024}MB"
        
        elif file_type == "image":
            if ext not in self.ALLOWED_IMAGE_TYPES:
                return False, f"Invalid image type. Allowed: {self.ALLOWED_IMAGE_TYPES}"
            if size > self.MAX_IMAGE_SIZE:
                return False, f"Image too large. Max size: {self.MAX_IMAGE_SIZE / 1024 / 1024}MB"
        
        return True, ""
    
    async def save_video(
        self,
        file: BinaryIO,
        filename: str,
        category: str = "videos",
        user_id: str = None
    ) -> Optional[StoredFile]:
        """
        Save a video file to local storage.
        
        Args:
            file: File-like object with video data
            filename: Original filename
            category: Subdirectory (videos, physio, guardian)
            user_id: Optional user ID for organizing files
            
        Returns:
            StoredFile object or None on failure
        """
        try:
            # Read file content
            content = file.read()
            size = len(content)
            
            # Validate
            is_valid, error = self._validate_file(filename, size, "video")
            if not is_valid:
                logger.error(f"Video validation failed: {error}")
                return None
            
            # Generate unique filename
            file_id = self._generate_file_id()
            ext = Path(filename).suffix.lower()
            safe_filename = f"{file_id}{ext}"
            
            # Determine storage path
            if user_id:
                storage_dir = self.base_path / category / user_id
                storage_dir.mkdir(parents=True, exist_ok=True)
            else:
                storage_dir = self.base_path / category
            
            file_path = storage_dir / safe_filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Generate URL
            relative_path = file_path.relative_to(self.base_path)
            url = f"{self.base_url}/media/{relative_path.as_posix()}"
            
            stored_file = StoredFile(
                file_id=file_id,
                filename=filename,
                path=str(file_path),
                url=url,
                size_bytes=size,
                content_type=self._get_content_type(filename),
                created_at=datetime.utcnow()
            )
            
            logger.info(f"✅ Saved video: {filename} -> {file_path} ({size} bytes)")
            return stored_file
            
        except Exception as e:
            logger.error(f"❌ Failed to save video: {e}")
            return None
    
    async def save_image(
        self,
        file: BinaryIO,
        filename: str,
        category: str = "images",
        user_id: str = None
    ) -> Optional[StoredFile]:
        """
        Save an image file to local storage.
        
        Args:
            file: File-like object with image data
            filename: Original filename
            category: Subdirectory (images, nutrition)
            user_id: Optional user ID for organizing files
            
        Returns:
            StoredFile object or None on failure
        """
        try:
            # Read file content
            content = file.read()
            size = len(content)
            
            # Validate
            is_valid, error = self._validate_file(filename, size, "image")
            if not is_valid:
                logger.error(f"Image validation failed: {error}")
                return None
            
            # Generate unique filename
            file_id = self._generate_file_id()
            ext = Path(filename).suffix.lower()
            safe_filename = f"{file_id}{ext}"
            
            # Determine storage path
            if user_id:
                storage_dir = self.base_path / category / user_id
                storage_dir.mkdir(parents=True, exist_ok=True)
            else:
                storage_dir = self.base_path / category
            
            file_path = storage_dir / safe_filename
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Generate URL
            relative_path = file_path.relative_to(self.base_path)
            url = f"{self.base_url}/media/{relative_path.as_posix()}"
            
            stored_file = StoredFile(
                file_id=file_id,
                filename=filename,
                path=str(file_path),
                url=url,
                size_bytes=size,
                content_type=self._get_content_type(filename),
                created_at=datetime.utcnow()
            )
            
            logger.info(f"✅ Saved image: {filename} -> {file_path} ({size} bytes)")
            return stored_file
            
        except Exception as e:
            logger.error(f"❌ Failed to save image: {e}")
            return None
    
    def get_file(self, file_id: str, category: str = None) -> Optional[Path]:
        """
        Find a file by its ID.
        
        Args:
            file_id: The unique file ID
            category: Optional category to search in
            
        Returns:
            Path to file or None if not found
        """
        search_dirs = [self.base_path / category] if category else [
            self.base_path / "videos",
            self.base_path / "images",
            self.base_path / "physio",
            self.base_path / "nutrition",
            self.base_path / "guardian",
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            # Search recursively
            for file_path in search_dir.rglob(f"{file_id}*"):
                if file_path.is_file():
                    return file_path
        
        return None
    
    def delete_file(self, file_id: str, category: str = None) -> bool:
        """
        Delete a file by its ID.
        
        Returns:
            True if deleted, False otherwise
        """
        file_path = self.get_file(file_id, category)
        
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"🗑️ Deleted file: {file_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete file: {e}")
        
        return False
    
    def cleanup_temp(self, max_age_hours: int = 24):
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Delete files older than this many hours
        """
        temp_dir = self.base_path / "temp"
        if not temp_dir.exists():
            return
        
        cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        deleted_count = 0
        
        for file_path in temp_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete temp file: {e}")
        
        if deleted_count > 0:
            logger.info(f"🧹 Cleaned up {deleted_count} temp files")
    
    def get_storage_stats(self) -> dict:
        """Get storage usage statistics."""
        stats = {
            "total_files": 0,
            "total_size_bytes": 0,
            "by_category": {}
        }
        
        for category in ["videos", "images", "physio", "nutrition", "guardian"]:
            category_dir = self.base_path / category
            if category_dir.exists():
                files = list(category_dir.rglob("*"))
                file_count = sum(1 for f in files if f.is_file())
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                
                stats["by_category"][category] = {
                    "files": file_count,
                    "size_bytes": total_size
                }
                stats["total_files"] += file_count
                stats["total_size_bytes"] += total_size
        
        return stats


# Global storage instance
_storage_manager: Optional[LocalStorageManager] = None


def get_storage() -> LocalStorageManager:
    """Get the global storage manager instance."""
    global _storage_manager
    
    if _storage_manager is None:
        _storage_manager = LocalStorageManager()
    
    return _storage_manager
