"""
SMARTCARE+ Physio Service - Face Pain Analyzer

Owner: Neelaka
MediaPipe Face Mesh-based pain/discomfort detection using Facial Action Units.

Uses geometric analysis of 478 face landmarks to detect:
- Brow lowering (AU4)
- Eye squinting/orbital tightening (AU6+7)
- Nose wrinkling/upper lip raising (AU9+10)
- Eye closure (AU43)
- Mouth tension

Pain indicators are stored for:
- Real-time feedback during exercise
- Exercise plan adaptation
- Weekly caregiver reports
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import os
import logging
import sys


def _setup_logger(name: str) -> logging.Logger:
    """Configure logger at DEBUG level."""
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        _logger.addHandler(handler)
    return _logger

logger = _setup_logger("smartcare.physio.face_pain")

# ── MediaPipe Face Landmarker imports ────────────────────────────────────────
FACE_MESH_AVAILABLE = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    FACE_MESH_AVAILABLE = True
    logger.info(f"✅ MediaPipe Face Landmarker API available (v{mp.__version__})")
except ImportError as e:
    logger.warning(f"⚠️ MediaPipe not available for face analysis: {e}")


# ── Face landmark indices for pain-related regions ───────────────────────────
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

class FaceLandmarks:
    """Key face landmark indices for pain detection (478 total landmarks)."""
    
    # Eyebrows (for brow lowering - AU4)
    LEFT_EYEBROW_TOP = [336, 296, 334, 293, 300]
    LEFT_EYEBROW_BOTTOM = [285, 295, 282, 283, 276]
    RIGHT_EYEBROW_TOP = [107, 66, 105, 63, 70]
    RIGHT_EYEBROW_BOTTOM = [55, 65, 52, 53, 46]
    
    # Eyes (for squinting/orbital tightening - AU6+7, closure - AU43)
    LEFT_EYE_TOP = [386, 374, 373, 390, 388]
    LEFT_EYE_BOTTOM = [263, 249, 390, 373, 374]
    RIGHT_EYE_TOP = [159, 145, 144, 163, 161]
    RIGHT_EYE_BOTTOM = [33, 7, 163, 144, 145]
    
    # Upper eyelid for eye closure detection
    LEFT_UPPER_EYELID = [386, 387, 388, 466, 263]
    LEFT_LOWER_EYELID = [374, 380, 381, 382, 362]
    RIGHT_UPPER_EYELID = [159, 160, 161, 246, 33]
    RIGHT_LOWER_EYELID = [145, 153, 154, 155, 133]
    
    # Nose (for wrinkling - AU9)
    NOSE_TIP = 4
    NOSE_BRIDGE = [168, 6, 197, 195, 5]
    NOSE_SIDES = [48, 278]  # Left and right nasolabial
    
    # Mouth (for tension, grimacing - AU10, AU25, AU26)
    UPPER_LIP_TOP = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
    LOWER_LIP_BOTTOM = [17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84]
    MOUTH_CORNERS = [61, 291]  # Left and right corners
    
    # Chin (for jaw tension)
    CHIN = [152, 175, 199, 200, 18]
    
    # Face contour reference points
    LEFT_CHEEK = [234]
    RIGHT_CHEEK = [454]
    FOREHEAD_CENTER = [10]


class PainLevel(Enum):
    """Pain severity levels."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class FacialPainResult:
    """Result of facial pain analysis."""
    pain_detected: bool
    pain_level: PainLevel
    confidence: float  # 0.0 - 1.0
    
    # Individual Action Unit scores (0-1)
    brow_lowering: float = 0.0       # AU4
    eye_squinting: float = 0.0       # AU6+7
    eye_closure: float = 0.0         # AU43
    nose_wrinkling: float = 0.0      # AU9
    mouth_tension: float = 0.0       # AU10/25
    
    # Metadata
    timestamp: float = 0.0
    face_detected: bool = True
    
    # Details for logging/reporting
    details: List[str] = field(default_factory=list)
    
    @property
    def pain_score(self) -> float:
        """Alias for confidence - the overall pain score (0-1)."""
        return self.confidence
    
    @property
    def action_units(self) -> Dict[str, float]:
        """Get action unit scores as a dictionary."""
        return {
            "brow_lowering": round(self.brow_lowering, 3),
            "eye_squinting": round(self.eye_squinting, 3),
            "eye_closure": round(self.eye_closure, 3),
            "nose_wrinkling": round(self.nose_wrinkling, 3),
            "mouth_tension": round(self.mouth_tension, 3),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pain_detected": self.pain_detected,
            "pain_level": self.pain_level.value,
            "confidence": round(self.confidence, 3),
            "pain_score": round(self.pain_score, 3),
            "action_units": self.action_units,
            "face_detected": self.face_detected,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class PainRecord:
    """A single pain detection event for storage."""
    session_id: str
    user_id: str
    timestamp: datetime
    pain_level: PainLevel
    confidence: float
    action_units: Dict[str, float]
    exercise_type: str
    rep_number: int
    set_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "pain_level": self.pain_level.value,
            "confidence": self.confidence,
            "action_units": self.action_units,
            "exercise_type": self.exercise_type,
            "rep_number": self.rep_number,
            "set_number": self.set_number,
        }


class FacePainAnalyzer:
    """
    Analyzes facial expressions for pain/discomfort indicators.
    
    Uses MediaPipe Face Landmarker (478 landmarks) to detect
    Facial Action Units associated with pain expression.
    """
    
    # Pain detection thresholds
    BROW_LOWERING_THRESHOLD = 0.15    # Distance decrease from baseline
    EYE_SQUINT_THRESHOLD = 0.3        # Eye aperture reduction
    EYE_CLOSURE_THRESHOLD = 0.1       # Near-closed eyes
    NOSE_WRINKLE_THRESHOLD = 0.2      # Nasolabial fold deepening
    MOUTH_TENSION_THRESHOLD = 0.25    # Lip compression/thinning
    
    # Combined pain thresholds
    MILD_PAIN_THRESHOLD = 0.25
    MODERATE_PAIN_THRESHOLD = 0.45
    SEVERE_PAIN_THRESHOLD = 0.65
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face pain analyzer.
        
        Args:
            model_path: Path to face_landmarker.task file
        """
        self.face_detector = None
        self.using_mock = True
        self._baseline_metrics: Optional[Dict[str, float]] = None
        self._frame_count = 0
        self._calibration_frames = 15  # Frames to establish baseline
        self._calibration_data: List[Dict[str, float]] = []
        
        # Pain history for this session
        self.pain_history: List[FacialPainResult] = []
        self.max_history = 100
        
        if FACE_MESH_AVAILABLE:
            self._init_face_landmarker(model_path)
    
    def _find_model_file(self) -> Optional[str]:
        """Find face landmarker model file."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths = [
            os.path.join(script_dir, "..", "..", "scripts", "face_landmarker.task"),
            os.path.join(script_dir, "..", "..", "ml_models", "physio", "face_landmarker.task"),
            os.path.join(script_dir, "..", "..", "ml_models", "face_landmarker.task"),
        ]
        for path in search_paths:
            resolved = os.path.abspath(path)
            if os.path.exists(resolved):
                logger.info(f"  📁 Found face model: {resolved}")
                return resolved
        return None
    
    def _init_face_landmarker(self, model_path: Optional[str] = None):
        """Initialize MediaPipe Face Landmarker."""
        try:
            if not model_path:
                model_path = self._find_model_file()
            
            if not model_path:
                logger.warning("⚠️ Face landmarker model not found")
                return
            
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,  # Get blendshapes for AU detection
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(options)
            self.using_mock = False
            logger.info("✅ Face Landmarker initialized for pain detection")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to init Face Landmarker: {e}")
            self.face_detector = None
            self.using_mock = True
    
    def analyze_pain(self, image: np.ndarray, timestamp_ms: float = 0) -> FacialPainResult:
        """
        Analyze facial expression for pain indicators.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            timestamp_ms: Frame timestamp
        
        Returns:
            FacialPainResult with pain detection results
        """
        if not FACE_MESH_AVAILABLE or self.face_detector is None:
            return self._generate_mock_result(timestamp_ms)
        
        try:
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
            # Detect face landmarks
            result = self.face_detector.detect(mp_image)
            
            if not result.face_landmarks or len(result.face_landmarks) == 0:
                logger.debug("  ❌ No face detected")
                return FacialPainResult(
                    pain_detected=False,
                    pain_level=PainLevel.NONE,
                    confidence=0.0,
                    face_detected=False,
                    timestamp=timestamp_ms,
                    details=["No face detected in frame"]
                )
            
            # Get landmarks for first face
            landmarks = result.face_landmarks[0]
            
            # Get blendshapes if available (more accurate for AU detection)
            blendshapes = None
            if result.face_blendshapes and len(result.face_blendshapes) > 0:
                blendshapes = {bs.category_name: bs.score for bs in result.face_blendshapes[0]}
            
            # Calculate pain indicators
            pain_result = self._analyze_pain_indicators(landmarks, blendshapes, timestamp_ms)
            
            # Store in history
            self.pain_history.append(pain_result)
            if len(self.pain_history) > self.max_history:
                self.pain_history.pop(0)
            
            return pain_result
            
        except Exception as e:
            logger.error(f"Face pain analysis error: {e}", exc_info=True)
            return FacialPainResult(
                pain_detected=False,
                pain_level=PainLevel.NONE,
                confidence=0.0,
                timestamp=timestamp_ms,
                details=[f"Analysis error: {str(e)}"]
            )
    
    def _analyze_pain_indicators(
        self, 
        landmarks: List, 
        blendshapes: Optional[Dict[str, float]],
        timestamp_ms: float
    ) -> FacialPainResult:
        """
        Analyze landmarks for pain-related Action Units.
        """
        details = []
        
        # If blendshapes available, use them (more accurate)
        if blendshapes:
            return self._analyze_from_blendshapes(blendshapes, timestamp_ms)
        
        # Otherwise, use geometric analysis
        # Convert landmarks to numpy array
        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        # Calculate individual AU scores
        brow_score = self._calculate_brow_lowering(lm_array)
        eye_squint_score = self._calculate_eye_squinting(lm_array)
        eye_closure_score = self._calculate_eye_closure(lm_array)
        nose_score = self._calculate_nose_wrinkling(lm_array)
        mouth_score = self._calculate_mouth_tension(lm_array)
        
        # Build details
        if brow_score > self.BROW_LOWERING_THRESHOLD:
            details.append(f"Brow furrowing ({brow_score:.0%})")
        if eye_squint_score > self.EYE_SQUINT_THRESHOLD:
            details.append(f"Eye squinting ({eye_squint_score:.0%})")
        if eye_closure_score > self.EYE_CLOSURE_THRESHOLD:
            details.append(f"Eyes closing ({eye_closure_score:.0%})")
        if nose_score > self.NOSE_WRINKLE_THRESHOLD:
            details.append(f"Nose wrinkling ({nose_score:.0%})")
        if mouth_score > self.MOUTH_TENSION_THRESHOLD:
            details.append(f"Mouth tension ({mouth_score:.0%})")
        
        # Combine scores with weights (based on pain expression research)
        # Brow lowering and mouth tension are strongest pain indicators
        combined_score = (
            brow_score * 0.25 +
            eye_squint_score * 0.20 +
            eye_closure_score * 0.15 +
            nose_score * 0.15 +
            mouth_score * 0.25
        )
        
        # Determine pain level
        if combined_score >= self.SEVERE_PAIN_THRESHOLD:
            pain_level = PainLevel.SEVERE
            pain_detected = True
        elif combined_score >= self.MODERATE_PAIN_THRESHOLD:
            pain_level = PainLevel.MODERATE
            pain_detected = True
        elif combined_score >= self.MILD_PAIN_THRESHOLD:
            pain_level = PainLevel.MILD
            pain_detected = True
        else:
            pain_level = PainLevel.NONE
            pain_detected = False
        
        logger.debug(
            f"  😣 Face pain: {pain_level.value} ({combined_score:.0%}) | "
            f"brow={brow_score:.2f} eye_sq={eye_squint_score:.2f} "
            f"eye_cl={eye_closure_score:.2f} nose={nose_score:.2f} mouth={mouth_score:.2f}"
        )
        
        return FacialPainResult(
            pain_detected=pain_detected,
            pain_level=pain_level,
            confidence=combined_score,
            brow_lowering=brow_score,
            eye_squinting=eye_squint_score,
            eye_closure=eye_closure_score,
            nose_wrinkling=nose_score,
            mouth_tension=mouth_score,
            face_detected=True,
            timestamp=timestamp_ms,
            details=details
        )
    
    def _analyze_from_blendshapes(
        self, 
        blendshapes: Dict[str, float], 
        timestamp_ms: float
    ) -> FacialPainResult:
        """
        Analyze pain using MediaPipe's built-in blendshapes.
        
        Blendshape names are based on ARKit face tracking:
        https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
        """
        details = []
        
        # Map blendshapes to pain-related AUs
        brow_score = max(
            blendshapes.get("browDownLeft", 0),
            blendshapes.get("browDownRight", 0),
            blendshapes.get("browInnerUp", 0) * 0.5  # Inner brow raise can indicate distress
        )
        
        eye_squint_score = max(
            blendshapes.get("eyeSquintLeft", 0),
            blendshapes.get("eyeSquintRight", 0)
        )
        
        eye_closure_score = max(
            blendshapes.get("eyeBlinkLeft", 0),
            blendshapes.get("eyeBlinkRight", 0)
        )
        
        nose_score = blendshapes.get("noseSneerLeft", 0) + blendshapes.get("noseSneerRight", 0)
        nose_score = min(nose_score, 1.0)
        
        # Mouth tension from multiple blendshapes
        mouth_score = max(
            blendshapes.get("mouthPressLeft", 0),
            blendshapes.get("mouthPressRight", 0),
            blendshapes.get("mouthStretchLeft", 0) * 0.7,
            blendshapes.get("mouthStretchRight", 0) * 0.7,
            (blendshapes.get("jawOpen", 0) > 0.5) * blendshapes.get("mouthFunnel", 0)  # Pain vocalization
        )
        
        # Build details
        if brow_score > 0.2:
            details.append(f"Brow furrowing ({brow_score:.0%})")
        if eye_squint_score > 0.25:
            details.append(f"Eye squinting ({eye_squint_score:.0%})")
        if eye_closure_score > 0.5:
            details.append(f"Eyes closing ({eye_closure_score:.0%})")
        if nose_score > 0.15:
            details.append(f"Nose wrinkling ({nose_score:.0%})")
        if mouth_score > 0.2:
            details.append(f"Mouth tension ({mouth_score:.0%})")
        
        # Combined weighted score
        combined_score = (
            brow_score * 0.25 +
            eye_squint_score * 0.20 +
            eye_closure_score * 0.15 +
            nose_score * 0.15 +
            mouth_score * 0.25
        )
        
        # Determine pain level
        if combined_score >= 0.50:
            pain_level = PainLevel.SEVERE
            pain_detected = True
        elif combined_score >= 0.35:
            pain_level = PainLevel.MODERATE
            pain_detected = True
        elif combined_score >= 0.20:
            pain_level = PainLevel.MILD
            pain_detected = True
        else:
            pain_level = PainLevel.NONE
            pain_detected = False
        
        logger.debug(
            f"  😣 Face pain (blendshapes): {pain_level.value} ({combined_score:.0%}) | "
            f"brow={brow_score:.2f} squint={eye_squint_score:.2f} "
            f"close={eye_closure_score:.2f} nose={nose_score:.2f} mouth={mouth_score:.2f}"
        )
        
        return FacialPainResult(
            pain_detected=pain_detected,
            pain_level=pain_level,
            confidence=combined_score,
            brow_lowering=brow_score,
            eye_squinting=eye_squint_score,
            eye_closure=eye_closure_score,
            nose_wrinkling=nose_score,
            mouth_tension=mouth_score,
            face_detected=True,
            timestamp=timestamp_ms,
            details=details
        )
    
    # ── Geometric AU calculation methods ─────────────────────────────────────
    
    def _calculate_brow_lowering(self, landmarks: np.ndarray) -> float:
        """Calculate brow lowering (AU4) from landmark geometry."""
        try:
            # Distance between eyebrow and eye
            left_brow_y = np.mean(landmarks[FaceLandmarks.LEFT_EYEBROW_BOTTOM, 1])
            left_eye_y = np.mean(landmarks[FaceLandmarks.LEFT_EYE_TOP, 1])
            left_dist = left_eye_y - left_brow_y
            
            right_brow_y = np.mean(landmarks[FaceLandmarks.RIGHT_EYEBROW_BOTTOM, 1])
            right_eye_y = np.mean(landmarks[FaceLandmarks.RIGHT_EYE_TOP, 1])
            right_dist = right_eye_y - right_brow_y
            
            # Smaller distance = more lowered brows
            avg_dist = (left_dist + right_dist) / 2
            
            # Normalize (typical range is 0.02-0.06 in normalized coords)
            score = max(0, (0.05 - avg_dist) / 0.05)
            return min(1.0, score)
        except:
            return 0.0
    
    def _calculate_eye_squinting(self, landmarks: np.ndarray) -> float:
        """Calculate eye squinting (AU6+7) from eye aperture."""
        try:
            # Left eye vertical aperture
            left_top = np.mean(landmarks[FaceLandmarks.LEFT_EYE_TOP, 1])
            left_bottom = np.mean(landmarks[FaceLandmarks.LEFT_EYE_BOTTOM, 1])
            left_aperture = left_bottom - left_top
            
            # Right eye vertical aperture
            right_top = np.mean(landmarks[FaceLandmarks.RIGHT_EYE_TOP, 1])
            right_bottom = np.mean(landmarks[FaceLandmarks.RIGHT_EYE_BOTTOM, 1])
            right_aperture = right_bottom - right_top
            
            avg_aperture = (left_aperture + right_aperture) / 2
            
            # Smaller aperture = more squinting
            # Normal aperture is ~0.02-0.04 in normalized coords
            score = max(0, (0.03 - avg_aperture) / 0.03)
            return min(1.0, score)
        except:
            return 0.0
    
    def _calculate_eye_closure(self, landmarks: np.ndarray) -> float:
        """Calculate eye closure (AU43)."""
        try:
            # Similar to squinting but with tighter threshold
            left_top = np.mean(landmarks[FaceLandmarks.LEFT_UPPER_EYELID, 1])
            left_bottom = np.mean(landmarks[FaceLandmarks.LEFT_LOWER_EYELID, 1])
            left_aperture = left_bottom - left_top
            
            right_top = np.mean(landmarks[FaceLandmarks.RIGHT_UPPER_EYELID, 1])
            right_bottom = np.mean(landmarks[FaceLandmarks.RIGHT_LOWER_EYELID, 1])
            right_aperture = right_bottom - right_top
            
            avg_aperture = (left_aperture + right_aperture) / 2
            
            # Very small aperture = closed eyes
            score = max(0, (0.015 - avg_aperture) / 0.015)
            return min(1.0, score)
        except:
            return 0.0
    
    def _calculate_nose_wrinkling(self, landmarks: np.ndarray) -> float:
        """Calculate nose wrinkling (AU9) from nasolabial fold."""
        try:
            # Distance from nose tip to mouth corners should decrease when wrinkling
            nose_tip = landmarks[FaceLandmarks.NOSE_TIP]
            left_corner = landmarks[FaceLandmarks.MOUTH_CORNERS[0]]
            right_corner = landmarks[FaceLandmarks.MOUTH_CORNERS[1]]
            
            left_dist = np.linalg.norm(nose_tip[:2] - left_corner[:2])
            right_dist = np.linalg.norm(nose_tip[:2] - right_corner[:2])
            
            avg_dist = (left_dist + right_dist) / 2
            
            # Smaller distance indicates nose wrinkling/upper lip raise
            score = max(0, (0.15 - avg_dist) / 0.15)
            return min(1.0, score)
        except:
            return 0.0
    
    def _calculate_mouth_tension(self, landmarks: np.ndarray) -> float:
        """Calculate mouth tension from lip compression."""
        try:
            # Vertical mouth aperture (lips pressed together = smaller)
            upper_lip_y = np.mean(landmarks[FaceLandmarks.UPPER_LIP_TOP[:5], 1])
            lower_lip_y = np.mean(landmarks[FaceLandmarks.LOWER_LIP_BOTTOM[:5], 1])
            vertical_aperture = lower_lip_y - upper_lip_y
            
            # Horizontal mouth width
            left_corner = landmarks[FaceLandmarks.MOUTH_CORNERS[0]]
            right_corner = landmarks[FaceLandmarks.MOUTH_CORNERS[1]]
            mouth_width = abs(right_corner[0] - left_corner[0])
            
            # Ratio of height to width (compressed lips = lower ratio)
            if mouth_width > 0:
                aspect_ratio = vertical_aperture / mouth_width
            else:
                aspect_ratio = 0.5
            
            # Low ratio = pressed lips = tension
            score = max(0, (0.3 - aspect_ratio) / 0.3)
            return min(1.0, score)
        except:
            return 0.0
    
    def _generate_mock_result(self, timestamp_ms: float) -> FacialPainResult:
        """Generate mock pain result for testing."""
        return FacialPainResult(
            pain_detected=False,
            pain_level=PainLevel.NONE,
            confidence=0.0,
            face_detected=False,
            timestamp=timestamp_ms,
            details=["Face mesh not available (mock mode)"]
        )
    
    def reset(self):
        """Reset analyzer state for new session."""
        self._baseline_metrics = None
        self._frame_count = 0
        self._calibration_data = []
        self.pain_history = []


# ── Global instance ──────────────────────────────────────────────────────────
_face_analyzer_instance: Optional[FacePainAnalyzer] = None


def get_face_pain_analyzer() -> FacePainAnalyzer:
    """Get or create global face pain analyzer instance."""
    global _face_analyzer_instance
    if _face_analyzer_instance is None:
        _face_analyzer_instance = FacePainAnalyzer()
    return _face_analyzer_instance
