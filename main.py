"""
SMARTCARE+ Backend API
AI-Powered Elderly Care Ecosystem

FastAPI application entry point with WebSocket support and worker threads
for non-blocking video processing.
"""

import logging
<<<<<<< Updated upstream
=======
import os
>>>>>>> Stashed changes
import sys
import time
from contextlib import asynccontextmanager

<<<<<<< Updated upstream
=======
# Force unbuffered stdout/stderr - prevents logs from disappearing on Windows
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

>>>>>>> Stashed changes
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware

# ============================================
# Configure Root Logger First
# ============================================
<<<<<<< Updated upstream
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
=======
_root_handler = logging.StreamHandler(sys.stdout)
_root_handler.setLevel(logging.DEBUG)
_root_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

# Override emit to force flush after every log
_original_emit = _root_handler.emit
def _flushing_emit(record):
    _original_emit(record)
    _root_handler.flush()
_root_handler.emit = _flushing_emit

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[_root_handler]
>>>>>>> Stashed changes
)

# Service routers
from physio_service.router import router as physio_router
from nutrition_service.router import router as nutrition_router
from guardian_service.router import router as guardian_router
from core.users import router as users_router
from core.connections import router as connections_router
<<<<<<< Updated upstream
=======
from core.tasks import router as tasks_router
from core.reports import router as reports_router
from core.accessibility import router as accessibility_router
>>>>>>> Stashed changes

# Core utilities
from core.config import settings
from core.database import init_firebase, get_db, is_mock_mode
from core.websocket import connection_manager
from core.threading import video_worker_pool, ml_worker_pool
from core.notifications import fcm_service
<<<<<<< Updated upstream
from shared.utils import setup_logger
=======
from shared.utils import setup_logger, init_session_log

# Initialize session log file (clears on each server startup)
init_session_log()
>>>>>>> Stashed changes

# Setup logging
logger = setup_logger("smartcare.main", level=logging.DEBUG)
request_logger = setup_logger("smartcare.requests", level=logging.DEBUG)


# ============================================
# Request Logging Middleware
# ============================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests and responses with timing."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request with more details
        client_ip = request.client.host if request.client else "unknown"
        query_string = f"?{request.url.query}" if request.url.query else ""
        auth_header = request.headers.get("Authorization", "")
        has_auth = "🔐" if auth_header else "🔓"
        
        request_logger.info(f"")
        request_logger.info(f"➡️  {has_auth} {request.method} {request.url.path}{query_string}")
        request_logger.debug(f"    Client: {client_ip}")
        request_logger.debug(f"    User-Agent: {request.headers.get('User-Agent', 'unknown')[:50]}")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000
            
            # Log response with colors based on status
            if response.status_code < 300:
                status_emoji = "✅"
            elif response.status_code < 400:
                status_emoji = "↪️"
            elif response.status_code < 500:
                status_emoji = "⚠️"
            else:
                status_emoji = "❌"
            
            request_logger.info(
                f"{status_emoji} {request.method} {request.url.path} → {response.status_code} ({process_time:.1f}ms)"
            )
            
            return response
        except Exception as e:
            process_time = (time.time() - start_time) * 1000
            request_logger.error(f"💥 {request.method} {request.url.path} → ERROR: {type(e).__name__}: {str(e)} ({process_time:.1f}ms)")
            import traceback
            request_logger.error(traceback.format_exc())
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # ===== STARTUP =====
    logger.info("🚀 SMARTCARE+ API starting up...")
    
    # Initialize Firebase
    if init_firebase():
        logger.info("🔥 Firebase connected")
    else:
        logger.warning("⚠️ Running in MOCK MODE (no Firebase)")
    
    # Start WebSocket heartbeat
    await connection_manager.start_heartbeat()
    logger.info("💓 WebSocket heartbeat started")
    
    # Initialize FCM
    fcm_service.initialize()
    
    # Create media directory for local storage
    media_path = Path(__file__).parent / "media"
    media_path.mkdir(exist_ok=True)
    
<<<<<<< Updated upstream
=======
    # Pre-initialize ML models for faster first detection request
    logger.info("🧠 Pre-initializing ML models in background...")
    try:
        from guardian_service.router import initialize_models_async
        await initialize_models_async()
    except Exception as e:
        logger.warning(f"⚠️ ML model pre-init failed (will load on first request): {e}")
    
>>>>>>> Stashed changes
    logger.info("✅ SMARTCARE+ API ready!")
    
    yield  # Application runs here
    
    # ===== SHUTDOWN =====
    logger.info("👋 SMARTCARE+ API shutting down...")
    
    # Stop heartbeat
    await connection_manager.stop_heartbeat()
    
    # Shutdown thread pools
    video_worker_pool.shutdown(wait=True)
    ml_worker_pool.shutdown(wait=True)
    
<<<<<<< Updated upstream
=======
    # Shutdown guardian analysis thread pool  
    try:
        from guardian_service.router import analysis_thread_pool
        analysis_thread_pool.shutdown(wait=False)
    except Exception:
        pass
    
>>>>>>> Stashed changes
    logger.info("✅ Shutdown complete")


app = FastAPI(
    title="SMARTCARE+ API",
    description="AI-Powered Elderly Care Ecosystem - Backend Services",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Mount static files for media (local storage)
media_path = Path(__file__).parent / "media"
media_path.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=str(media_path)), name="media")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "smartcare-api",
        "firebase": "connected" if not is_mock_mode() else "mock",
        "websocket_connections": connection_manager.connection_count
    }


@app.get("/stats")
async def get_stats():
    """Get service statistics."""
    return {
        "websocket": connection_manager.get_stats(),
        "video_pool": video_worker_pool.get_stats(),
        "ml_pool": ml_worker_pool.get_stats(),
        "fcm": fcm_service.get_stats()
    }


# Include service routers
app.include_router(users_router, prefix="/api/users", tags=["Users"])
app.include_router(connections_router, prefix="/api/connections", tags=["Connections"])
app.include_router(physio_router, prefix="/api/physio", tags=["Physio Service"])
app.include_router(nutrition_router, prefix="/api/nutrition", tags=["Nutrition Service"])
app.include_router(guardian_router, prefix="/api/guardian", tags=["Guardian Service"])
<<<<<<< Updated upstream
=======
app.include_router(tasks_router, prefix="/api/tasks", tags=["Task Service"])
app.include_router(reports_router, prefix="/api/reports", tags=["Report Service"])
app.include_router(accessibility_router, prefix="/api/accessibility", tags=["Accessibility"])
>>>>>>> Stashed changes


if __name__ == "__main__":
    import uvicorn
<<<<<<< Updated upstream
=======
    # Set PYTHONUNBUFFERED before spawning reload worker to prevent log buffering
    os.environ["PYTHONUNBUFFERED"] = "1"
>>>>>>> Stashed changes
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
