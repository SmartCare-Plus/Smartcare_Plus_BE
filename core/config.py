"""
SMARTCARE+ Configuration

Environment variables and application settings.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    APP_NAME: str = "SMARTCARE+"
    DEBUG: bool = True
    
    # Firebase
    FIREBASE_PROJECT_ID: str = "smartcare-plus-c9617"
    FIREBASE_CREDENTIALS_PATH: str = "service-account.json"
    
    # Local Storage (Firebase Storage fallback)
    LOCAL_MEDIA_PATH: str = "media"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080", "http://10.0.2.2:8000"]
    
    # WebSocket
    WS_MAX_CONNECTIONS: int = 100
    WS_HEARTBEAT_INTERVAL: int = 30
    
    # Thread Pool
    THREAD_POOL_SIZE: int = 4
    VIDEO_FRAME_QUEUE_SIZE: int = 100
    
    # ML Models
    ML_MODELS_PATH: str = "ml_models"
    
<<<<<<< Updated upstream
=======
    # Email / SMTP
    SMTP_EMAIL: str = ""
    SMTP_PASSWORD: str = ""

    # Speech-to-text accessibility
    GEMINI_API_KEY: str = ""
    GEMINI_STT_MODEL: str = "gemini-2.5-flash"
    STT_PROVIDER: str = "vertex_ai"  # vertex_ai | gemini_api
    VERTEX_AI_CREDENTIALS_PATH: str = ""
    VERTEX_AI_PROJECT_ID: str = ""
    VERTEX_AI_LOCATION: str = "us-central1"
    STT_DEFAULT_LOCALE: str = "en-US"
    STT_MAX_AUDIO_MB: int = 8
    
>>>>>>> Stashed changes
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
