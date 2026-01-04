"""
SMARTCARE+ Guardian Service - Video Stream WebSocket Handler

Owner: Madhushani
WebSocket handler for streaming pre-recorded videos as simulated CCTV.
"""

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class StreamState(Enum):
    """Video stream states."""
    IDLE = "idle"
    LOADING = "loading"
    STREAMING = "streaming"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Video stream configuration."""
    video_path: str
    target_fps: int = 15
    resize_width: int = 640
    resize_height: int = 480
    enable_pose_detection: bool = True
    enable_activity_detection: bool = True
    loop_video: bool = True


@dataclass
class StreamSession:
    """Active stream session."""
    session_id: str
    elderly_id: str
    camera_id: str
    config: StreamConfig
    state: StreamState = StreamState.IDLE
    
    # Video capture
    cap: Optional[Any] = None
    frame_count: int = 0
    current_frame: int = 0
    
    # Timing
    start_time: float = 0.0
    last_frame_time: float = 0.0
    
    # Activity tracking
    current_activity: str = "unknown"
    last_activity_change: float = 0.0
    
    # Alerts
    alerts: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO STREAM HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class VideoStreamHandler:
    """
    Handles video streaming with pose and activity detection.
    
    Uses pre-recorded videos to simulate live CCTV feeds for demo.
    """
    
    # Default video directory
    DEFAULT_VIDEO_DIR = Path(__file__).parent.parent.parent / "media" / "simulation_footage"
    
    # Activity detection (simplified rules)
    ACTIVITY_LABELS = ["walking", "sitting", "standing", "lying", "unknown"]
    
    def __init__(self, video_dir: Optional[str] = None):
        """
        Initialize stream handler.
        
        Args:
            video_dir: Directory containing video files
        """
        self.video_dir = Path(video_dir) if video_dir else self.DEFAULT_VIDEO_DIR
        self.sessions: Dict[str, StreamSession] = {}
        
        # Initialize pose detector if available
        self.pose_detector = None
        if MEDIAPIPE_AVAILABLE:
            self._init_pose_detector()
    
    def _init_pose_detector(self):
        """Initialize MediaPipe pose detector."""
        try:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # Lite model for speed
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✅ MediaPipe pose detector initialized for streaming")
        except Exception as e:
            print(f"⚠️ Failed to init MediaPipe: {e}")
            self.pose_detector = None
    
    def get_available_videos(self) -> Dict[str, Any]:
        """Get list of available video files."""
        videos = []
        
        if self.video_dir.exists():
            for video_path in self.video_dir.glob("*.mp4"):
                videos.append({
                    "name": video_path.stem,
                    "path": str(video_path),
                    "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2)
                })
            
            for video_path in self.video_dir.glob("*.avi"):
                videos.append({
                    "name": video_path.stem,
                    "path": str(video_path),
                    "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2)
                })
        
        return {
            "videos": videos,
            "directory": str(self.video_dir),
            "total": len(videos)
        }
    
    def create_session(
        self,
        session_id: str,
        elderly_id: str,
        camera_id: str,
        video_file: Optional[str] = None,
        config: Optional[StreamConfig] = None
    ) -> Dict[str, Any]:
        """
        Create a new streaming session.
        
        Args:
            session_id: Unique session ID
            elderly_id: Elderly person ID
            camera_id: Camera ID
            video_file: Video filename (optional)
            config: Stream configuration
        
        Returns:
            Session creation result
        """
        # Determine video path
        if video_file:
            video_path = self.video_dir / video_file
        else:
            # Use first available video
            videos = list(self.video_dir.glob("*.mp4"))
            if not videos:
                videos = list(self.video_dir.glob("*.avi"))
            
            if not videos:
                return {
                    "error": "No video files found",
                    "directory": str(self.video_dir)
                }
            video_path = videos[0]
        
        if not video_path.exists():
            return {"error": f"Video not found: {video_path}"}
        
        # Create config if not provided
        if config is None:
            config = StreamConfig(video_path=str(video_path))
        else:
            config.video_path = str(video_path)
        
        # Create session
        session = StreamSession(
            session_id=session_id,
            elderly_id=elderly_id,
            camera_id=camera_id,
            config=config
        )
        
        self.sessions[session_id] = session
        
        return {
            "status": "created",
            "session_id": session_id,
            "video": video_path.name,
            "config": {
                "fps": config.target_fps,
                "resolution": f"{config.resize_width}x{config.resize_height}",
                "pose_detection": config.enable_pose_detection,
                "loop": config.loop_video
            }
        }
    
    async def start_stream(self, session_id: str) -> Dict[str, Any]:
        """
        Start streaming video.
        
        Args:
            session_id: Session ID
        
        Returns:
            Stream start result
        """
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Open video file
        session.cap = cv2.VideoCapture(session.config.video_path)
        
        if not session.cap.isOpened():
            session.state = StreamState.ERROR
            return {"error": "Failed to open video file"}
        
        # Get video info
        session.frame_count = int(session.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = session.cap.get(cv2.CAP_PROP_FPS)
        
        session.state = StreamState.STREAMING
        session.start_time = time.time()
        
        return {
            "status": "streaming",
            "session_id": session_id,
            "video_info": {
                "total_frames": session.frame_count,
                "original_fps": original_fps,
                "target_fps": session.config.target_fps,
                "duration_seconds": session.frame_count / original_fps if original_fps > 0 else 0
            }
        }
    
    async def get_next_frame(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get next frame from video stream.
        
        Args:
            session_id: Session ID
        
        Returns:
            Frame data with optional pose landmarks
        """
        session = self.sessions.get(session_id)
        if not session or not session.cap:
            return None
        
        if session.state != StreamState.STREAMING:
            return None
        
        # Rate limiting
        current_time = time.time()
        min_interval = 1.0 / session.config.target_fps
        if current_time - session.last_frame_time < min_interval:
            await asyncio.sleep(min_interval - (current_time - session.last_frame_time))
        
        # Read frame
        ret, frame = session.cap.read()
        
        if not ret:
            if session.config.loop_video:
                # Loop back to beginning
                session.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = session.cap.read()
                if not ret:
                    return None
            else:
                session.state = StreamState.STOPPED
                return {"status": "ended", "session_id": session_id}
        
        session.current_frame += 1
        session.last_frame_time = time.time()
        
        # Resize frame
        frame = cv2.resize(frame, (session.config.resize_width, session.config.resize_height))
        
        result = {
            "frame_number": session.current_frame,
            "timestamp": time.time() - session.start_time,
        }
        
        # Pose detection
        landmarks = None
        if session.config.enable_pose_detection and self.pose_detector:
            landmarks = self._detect_pose(frame)
            if landmarks:
                result["pose"] = landmarks
                
                # Activity detection
                if session.config.enable_activity_detection:
                    activity = self._classify_activity(landmarks)
                    result["activity"] = activity
                    
                    if activity != session.current_activity:
                        session.current_activity = activity
                        session.last_activity_change = result["timestamp"]
        
        # Encode frame as base64 JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        result["frame"] = frame_base64
        
        return result
    
    def _detect_pose(self, frame: np.ndarray) -> Optional[Dict[int, Dict]]:
        """Detect pose landmarks in frame."""
        if not self.pose_detector:
            return None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.pose_detector.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            }
        
        return landmarks
    
    def _classify_activity(self, landmarks: Dict[int, Dict]) -> str:
        """Classify activity from pose landmarks."""
        # Simple rule-based classification
        
        # Get key landmarks
        left_hip = landmarks.get(23, {})
        right_hip = landmarks.get(24, {})
        left_knee = landmarks.get(25, {})
        right_knee = landmarks.get(26, {})
        left_shoulder = landmarks.get(11, {})
        right_shoulder = landmarks.get(12, {})
        
        if not all([left_hip, right_hip, left_knee, right_knee]):
            return "unknown"
        
        hip_y = (left_hip.get("y", 0) + right_hip.get("y", 0)) / 2
        shoulder_y = (left_shoulder.get("y", 0) + right_shoulder.get("y", 0)) / 2 if left_shoulder and right_shoulder else 0
        
        # Calculate knee angle
        def angle_at_point(a, b, c):
            ba = np.array([a.get("x", 0) - b.get("x", 0), a.get("y", 0) - b.get("y", 0)])
            bc = np.array([c.get("x", 0) - b.get("x", 0), c.get("y", 0) - b.get("y", 0)])
            cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        left_knee_angle = angle_at_point(left_hip, left_knee, landmarks.get(27, {}))
        right_knee_angle = angle_at_point(right_hip, right_knee, landmarks.get(28, {}))
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # Lying: very low shoulder position or horizontal
        if shoulder_y > 0.75:
            return "lying"
        
        # Sitting: hips high, bent knees
        if hip_y < 0.5 and avg_knee_angle < 130:
            return "sitting"
        
        # Standing: straight knees
        if avg_knee_angle > 150:
            return "standing"
        
        return "walking"
    
    def pause_stream(self, session_id: str) -> Dict[str, Any]:
        """Pause video stream."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        session.state = StreamState.PAUSED
        return {"status": "paused", "session_id": session_id}
    
    def resume_stream(self, session_id: str) -> Dict[str, Any]:
        """Resume paused stream."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        if session.state == StreamState.PAUSED:
            session.state = StreamState.STREAMING
            return {"status": "resumed", "session_id": session_id}
        
        return {"error": "Stream not paused"}
    
    def stop_stream(self, session_id: str) -> Dict[str, Any]:
        """Stop and clean up stream."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        if session.cap:
            session.cap.release()
        
        session.state = StreamState.STOPPED
        
        return {
            "status": "stopped",
            "session_id": session_id,
            "frames_streamed": session.current_frame,
            "duration_seconds": round(time.time() - session.start_time, 1)
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current session status."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "state": session.state.value,
            "elderly_id": session.elderly_id,
            "camera_id": session.camera_id,
            "current_frame": session.current_frame,
            "total_frames": session.frame_count,
            "current_activity": session.current_activity,
            "duration_seconds": round(time.time() - session.start_time, 1) if session.start_time > 0 else 0
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up session resources."""
        session = self.sessions.get(session_id)
        if session:
            if session.cap:
                session.cap.release()
            del self.sessions[session_id]
    
    def cleanup_all(self):
        """Clean up all sessions."""
        for session_id in list(self.sessions.keys()):
            self.cleanup_session(session_id)
        
        if self.pose_detector:
            self.pose_detector.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_handler: Optional[VideoStreamHandler] = None

def get_video_stream_handler() -> VideoStreamHandler:
    """Get or create video stream handler singleton."""
    global _handler
    if _handler is None:
        _handler = VideoStreamHandler()
    return _handler
