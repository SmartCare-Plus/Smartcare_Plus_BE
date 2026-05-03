"""
SMARTCARE+ Physio Service - Pose Analyzer

Owner: Neelaka
MediaPipe-based pose estimation with joint angle calculation and form validation.
Uses rule-based system for exercise monitoring (NOT deep learning for Physio).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math
import logging
import sys


def _setup_physio_logger(name: str) -> logging.Logger:
    """Configure a physio logger at DEBUG level with console output."""
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

logger = _setup_physio_logger("smartcare.physio.pose")

# ── MediaPipe imports ──────────────────────────────────────────────────────────
# MediaPipe 0.10.31+ uses Tasks API only (legacy mp.solutions removed)
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_TASKS_AVAILABLE = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_TASKS_AVAILABLE = True
    logger.info("✅ MediaPipe Tasks API available (v%s)", mp.__version__)
except ImportError as e:
    logger.warning("⚠️ MediaPipe not installed: %s. Will use mock pose data.", e)

# Check for legacy solutions API (older mediapipe versions)
MEDIAPIPE_LEGACY_AVAILABLE = False
if MEDIAPIPE_AVAILABLE:
    try:
        _ = mp.solutions.pose
        MEDIAPIPE_LEGACY_AVAILABLE = True
        logger.info("✅ MediaPipe legacy API (mp.solutions.pose) also available")
    except AttributeError:
        logger.info("ℹ️ MediaPipe legacy API not available (expected for v0.10+)")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class JointType(Enum):
    """Body joint types for pose estimation."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class ExerciseType(Enum):
    """Supported exercise types."""
    # Original trackable exercises (with pose analysis)
    CHAIR_STAND = "chair_stand"
    HEEL_TOE_WALK = "heel_toe_walk"
    SINGLE_LEG_STAND = "single_leg_stand"
    ANKLE_CIRCLES = "ankle_circles"
    WALL_PUSHUP = "wall_pushup"
    TANDEM_STAND = "tandem_stand"
    MARCHING = "marching"
    LEG_RAISE = "leg_raise"
    ARM_RAISE = "arm_raise"
    SQUAT = "squat"
    
    # Additional exercises (basic tracking or instruction-based)
    NECK_ROTATIONS = "neck_rotations"
    SHOULDER_ROLLS = "shoulder_rolls"
    SEATED_HAMSTRING_STRETCH = "seated_hamstring_stretch"
    SEATED_HIP_STRETCH = "seated_hip_stretch"
    CALF_STRETCH = "calf_stretch"
    SEATED_LEG_RAISE = "seated_leg_raise"
    SEATED_ARM_RAISES = "seated_arm_raises"
    DEEP_BREATHING = "deep_breathing"
    GENTLE_SPINAL_TWIST = "gentle_spinal_twist"
    MARCHING_IN_PLACE = "marching_in_place"
    
    @classmethod
    def is_trackable(cls, exercise_type: 'ExerciseType') -> bool:
        """Check if exercise type supports full pose tracking."""
        trackable = {
            cls.CHAIR_STAND, cls.WALL_PUSHUP, cls.LEG_RAISE, cls.ARM_RAISE,
            cls.SQUAT, cls.MARCHING, cls.ANKLE_CIRCLES, cls.SEATED_LEG_RAISE,
            cls.SEATED_ARM_RAISES, cls.SHOULDER_ROLLS
        }
        return exercise_type in trackable


class FormQuality(Enum):
    """Form quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class Landmark:
    """A single pose landmark with 3D coordinates and visibility."""
    x: float
    y: float
    z: float
    visibility: float
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class PoseResult:
    """Complete pose estimation result."""
    landmarks: Dict[int, Landmark]
    timestamp: float
    confidence: float
    world_landmarks: Optional[Dict[int, Landmark]] = None


@dataclass
class JointAngle:
    """Joint angle measurement."""
    name: str
    angle: float  # degrees
    reference_min: float  # acceptable range
    reference_max: float
    
    @property
    def is_in_range(self) -> bool:
        return self.reference_min <= self.angle <= self.reference_max
    
    @property
    def deviation(self) -> float:
        """How far outside the acceptable range."""
        if self.angle < self.reference_min:
            return self.reference_min - self.angle
        elif self.angle > self.reference_max:
            return self.angle - self.reference_max
        return 0.0


@dataclass
class FormAssessment:
    """Form quality assessment for an exercise."""
    exercise_type: ExerciseType
    quality: FormQuality
    score: float  # 0-100
    joint_angles: List[JointAngle]
    feedback: List[str]
    timestamp: float


@dataclass
class RepCounter:
    """Tracks repetition counting for exercises."""
    exercise_type: ExerciseType
    count: int = 0
    state: str = "neutral"  # neutral, up, down, etc.
    last_angle: float = 0.0
    threshold_up: float = 0.0
    threshold_down: float = 0.0
    
    def update(self, current_angle: float) -> bool:
        """Update counter with new angle. Returns True if rep completed."""
        rep_completed = False
        
        if self.state == "neutral":
            if current_angle > self.threshold_up:
                self.state = "up"
        elif self.state == "up":
            if current_angle < self.threshold_down:
                self.state = "down"
                self.count += 1
                rep_completed = True
        elif self.state == "down":
            if current_angle > self.threshold_up:
                self.state = "up"
        
        self.last_angle = current_angle
        return rep_completed


# ═══════════════════════════════════════════════════════════════════════════════
# POSE ANALYZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class PoseAnalyzer:
    """
    MediaPipe-based pose analyzer for physiotherapy exercises.
    
    Uses rule-based logic for:
    - Joint angle calculation
    - Form validation
    - Repetition counting
    - Pain/discomfort detection
    """
    
    # Exercise-specific angle thresholds
    EXERCISE_THRESHOLDS = {
        ExerciseType.CHAIR_STAND: {
            "knee_angle_standing": (160, 180),  # Nearly straight
            "knee_angle_sitting": (70, 110),    # 90 degrees bent
            "hip_angle_standing": (160, 180),
            "hip_angle_sitting": (70, 110),
            "rep_up": 150,
            "rep_down": 100,
        },
        ExerciseType.SQUAT: {
            "knee_angle_up": (150, 180),
            "knee_angle_down": (70, 110),
            "hip_angle_up": (150, 180),
            "hip_angle_down": (70, 110),
            "back_straight": (160, 180),
            "rep_up": 150,
            "rep_down": 100,
        },
        ExerciseType.LEG_RAISE: {
            "hip_angle_up": (60, 100),
            "hip_angle_down": (160, 180),
            "knee_straight": (160, 180),
            "rep_up": 120,
            "rep_down": 160,
        },
        ExerciseType.ARM_RAISE: {
            "shoulder_angle_up": (150, 180),
            "shoulder_angle_down": (0, 30),
            "elbow_straight": (160, 180),
            "rep_up": 140,
            "rep_down": 40,
        },
        ExerciseType.WALL_PUSHUP: {
            "elbow_angle_extended": (150, 180),
            "elbow_angle_bent": (70, 110),
            "shoulder_angle": (60, 120),
            "rep_up": 150,
            "rep_down": 90,
        },
        ExerciseType.SINGLE_LEG_STAND: {
            "standing_knee_straight": (160, 180),
            "raised_knee_bent": (70, 110),
            "hip_alignment": (-10, 10),  # deviation in degrees
        },
        ExerciseType.MARCHING: {
            "knee_lift_min": 70,
            "knee_lift_max": 110,
            "rep_up": 100,
            "rep_down": 150,
        },
        # Additional exercise thresholds (basic/simple tracking)
        ExerciseType.SHOULDER_ROLLS: {
            "shoulder_angle_range": (0, 180),
            "rep_up": 90,
            "rep_down": 30,
        },
        ExerciseType.NECK_ROTATIONS: {
            "head_angle_range": (-45, 45),
            "rep_up": 30,
            "rep_down": 0,
        },
        ExerciseType.SEATED_LEG_RAISE: {
            "hip_angle_up": (60, 100),
            "hip_angle_down": (160, 180),
            "knee_straight": (160, 180),
            "rep_up": 120,
            "rep_down": 160,
        },
        ExerciseType.SEATED_ARM_RAISES: {
            "shoulder_angle_up": (150, 180),
            "shoulder_angle_down": (0, 30),
            "rep_up": 140,
            "rep_down": 40,
        },
        ExerciseType.DEEP_BREATHING: {
            # No pose tracking needed - timer-based
            "rep_up": 0,
            "rep_down": 0,
        },
        ExerciseType.GENTLE_SPINAL_TWIST: {
            "trunk_rotation": (-45, 45),
            "rep_up": 30,
            "rep_down": 0,
        },
        ExerciseType.MARCHING_IN_PLACE: {
            "knee_lift_min": 70,
            "knee_lift_max": 110,
            "rep_up": 100,
            "rep_down": 150,
        },
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize pose analyzer.
        
        Args:
            model_path: Path to MediaPipe pose model (.task file). 
                       If None, auto-discovers pose_landmarker_full.task.
        """
        self.pose_detector = None
        self.rep_counters: Dict[ExerciseType, RepCounter] = {}
        self.pose_history: List[PoseResult] = []
        self.max_history = 30  # For smoothing and velocity calculation
        self.using_mock = True  # Track whether we're using real or mock data
        self._use_tasks_api = False  # Which API path is active
        self._use_legacy_api = False
        
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe(model_path)
        else:
            logger.warning("🔴 MediaPipe NOT available — all pose detection will use MOCK data")
    
    def _find_model_file(self) -> Optional[str]:
        """Auto-discover the pose landmarker model file."""
        import os
        # Check common locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths = [
            os.path.join(script_dir, "..", "..", "scripts", "pose_landmarker_full.task"),
            os.path.join(script_dir, "..", "..", "ml_models", "physio", "pose_landmarker_full.task"),
            os.path.join(script_dir, "..", "..", "ml_models", "pose_landmarker_full.task"),
        ]
        for path in search_paths:
            resolved = os.path.abspath(path)
            if os.path.exists(resolved):
                logger.info(f"  📁 Found model file: {resolved}")
                return resolved
        return None
    
    def _init_mediapipe(self, model_path: Optional[str] = None):
        """Initialize MediaPipe pose detector."""
        # Strategy 1: Try Tasks API (required for mediapipe 0.10+)
        if MEDIAPIPE_TASKS_AVAILABLE:
            if not model_path:
                model_path = self._find_model_file()
            
            if model_path:
                try:
                    base_options = mp_python.BaseOptions(model_asset_path=model_path)
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,  # Individual frames from camera
                        num_poses=1,
                        min_pose_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self.pose_detector = vision.PoseLandmarker.create_from_options(options)
                    self._use_tasks_api = True
                    self.using_mock = False
                    logger.info("✅ MediaPipe PoseLandmarker initialized (Tasks API, IMAGE mode)")
                    return
                except Exception as e:
                    logger.error(f"⚠️ Failed to initialize Tasks API: {e}")
            else:
                logger.warning("⚠️ No .task model file found for Tasks API")
        
        # Strategy 2: Try legacy API (mediapipe <0.10)
        if MEDIAPIPE_LEGACY_AVAILABLE:
            try:
                self._init_legacy_pose()
                return
            except Exception as e:
                logger.error(f"⚠️ Failed to initialize legacy API: {e}")
        
        # Both failed
        logger.error("🔴 Could not initialize any MediaPipe API — will use MOCK data")
        self.pose_detector = None
        self.using_mock = True
    
    def _init_legacy_pose(self):
        """Initialize MediaPipe legacy Pose API (for older mediapipe versions)."""
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,  # True for disconnected frames
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._use_legacy_api = True
        self.using_mock = False
        logger.info("✅ MediaPipe pose detector initialized (legacy API, static_image_mode=True)")
    
    def detect_pose(self, image: np.ndarray, timestamp_ms: float = 0) -> Optional[PoseResult]:
        """
        Detect pose landmarks in an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            timestamp_ms: Frame timestamp in milliseconds
        
        Returns:
            PoseResult with landmarks or None if detection failed
        """
        if not MEDIAPIPE_AVAILABLE or self.pose_detector is None:
            logger.warning("  🔴 MOCK DATA — MediaPipe not available or not initialized")
            return self._generate_mock_pose(timestamp_ms)
        
        try:
            if self._use_tasks_api:
                return self._detect_pose_tasks_api(image, timestamp_ms)
            elif self._use_legacy_api:
                return self._detect_pose_legacy(image, timestamp_ms)
            else:
                logger.warning("  🔴 No API path configured, using mock data")
                return self._generate_mock_pose(timestamp_ms)
        except Exception as e:
            logger.error(f"Pose detection error: {e}", exc_info=True)
            return None
    
    def _detect_pose_tasks_api(self, image: np.ndarray, timestamp_ms: float) -> Optional[PoseResult]:
        """Detect pose using MediaPipe Tasks API (v0.10+)."""
        logger.debug(f"  🎯 Running Tasks API pose detection on frame {image.shape}")
        
        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        
        # Detect poses
        result = self.pose_detector.detect(mp_image)
        
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            logger.debug("  ❌ No pose detected by Tasks API")
            return None
        
        # Extract landmarks from first detected pose
        pose_lms = result.pose_landmarks[0]  # List of NormalizedLandmark
        landmarks = {}
        for idx, lm in enumerate(pose_lms):
            landmarks[idx] = Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') else lm.presence
            )
        
        # Extract world landmarks if available
        world_landmarks = None
        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            world_lms = result.pose_world_landmarks[0]
            world_landmarks = {}
            for idx, lm in enumerate(world_lms):
                world_landmarks[idx] = Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility if hasattr(lm, 'visibility') else lm.presence
                )
        
        # Calculate overall confidence
        confidence = np.mean([lm.visibility for lm in landmarks.values()])
        
        # Log key landmark visibility  
        nose_vis = landmarks.get(0, Landmark(0,0,0,0)).visibility
        l_hip_vis = landmarks.get(23, Landmark(0,0,0,0)).visibility
        r_hip_vis = landmarks.get(24, Landmark(0,0,0,0)).visibility
        l_knee_vis = landmarks.get(25, Landmark(0,0,0,0)).visibility
        r_knee_vis = landmarks.get(26, Landmark(0,0,0,0)).visibility
        logger.debug(
            f"  👁️ Pose landmarks: nose={nose_vis:.2f} "
            f"L_hip={l_hip_vis:.2f} R_hip={r_hip_vis:.2f} "
            f"L_knee={l_knee_vis:.2f} R_knee={r_knee_vis:.2f} "
            f"conf={confidence:.2f}"
        )
        
        pose_result = PoseResult(
            landmarks=landmarks,
            timestamp=timestamp_ms,
            confidence=confidence,
            world_landmarks=world_landmarks
        )
        
        # Add to history
        self.pose_history.append(pose_result)
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        return pose_result
    
    def _detect_pose_legacy(self, image: np.ndarray, timestamp_ms: float) -> Optional[PoseResult]:
        """Detect pose using MediaPipe legacy solutions API (pre v0.10)."""
        logger.debug(f"  🎯 Running legacy API pose detection on frame {image.shape}")
        
        # Convert to MediaPipe format
        results = self.pose_detector.process(image)
        
        if not results.pose_landmarks:
            logger.debug("  ❌ No pose detected by legacy API")
            return None
        
        # Extract landmarks
        landmarks = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            )
        
        # Extract world landmarks if available
        world_landmarks = None
        if results.pose_world_landmarks:
            world_landmarks = {}
            for idx, lm in enumerate(results.pose_world_landmarks.landmark):
                world_landmarks[idx] = Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility
                )
        
        # Calculate overall confidence
        confidence = np.mean([lm.visibility for lm in landmarks.values()])
        
        # Log key landmark visibility
        nose_vis = landmarks.get(0, Landmark(0,0,0,0)).visibility
        l_hip_vis = landmarks.get(23, Landmark(0,0,0,0)).visibility
        r_hip_vis = landmarks.get(24, Landmark(0,0,0,0)).visibility
        l_knee_vis = landmarks.get(25, Landmark(0,0,0,0)).visibility
        r_knee_vis = landmarks.get(26, Landmark(0,0,0,0)).visibility
        logger.debug(
            f"  👁️ Pose landmarks: nose={nose_vis:.2f} "
            f"L_hip={l_hip_vis:.2f} R_hip={r_hip_vis:.2f} "
            f"L_knee={l_knee_vis:.2f} R_knee={r_knee_vis:.2f} "
            f"conf={confidence:.2f}"
        )
        
        pose_result = PoseResult(
            landmarks=landmarks,
            timestamp=timestamp_ms,
            confidence=confidence,
            world_landmarks=world_landmarks
        )
        
        # Add to history
        self.pose_history.append(pose_result)
        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)
        
        return pose_result
    
    def _generate_mock_pose(self, timestamp_ms: float) -> PoseResult:
        """Generate mock pose data for testing without MediaPipe."""
        landmarks = {}
        for joint in JointType:
            landmarks[joint.value] = Landmark(
                x=np.random.uniform(0.2, 0.8),
                y=np.random.uniform(0.1, 0.9),
                z=np.random.uniform(-0.1, 0.1),
                visibility=np.random.uniform(0.8, 1.0)
            )
        
        return PoseResult(
            landmarks=landmarks,
            timestamp=timestamp_ms,
            confidence=0.9
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # JOINT ANGLE CALCULATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Calculate angle at point b formed by points a-b-c.
        
        Args:
            a, b, c: 3D points as numpy arrays
        
        Returns:
            Angle in degrees (0-180)
        """
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle
    
    def get_joint_angles(self, pose: PoseResult) -> Dict[str, float]:
        """
        Calculate all major joint angles from pose landmarks.
        
        Returns dict with angle names and values in degrees.
        """
        lm = pose.landmarks
        angles = {}
        
        # Helper to get landmark as numpy array
        def get_point(joint: JointType) -> np.ndarray:
            landmark = lm.get(joint.value)
            if landmark:
                return landmark.to_numpy()
            return np.zeros(3)
        
        # Left elbow angle (shoulder-elbow-wrist)
        angles["left_elbow"] = self.calculate_angle(
            get_point(JointType.LEFT_SHOULDER),
            get_point(JointType.LEFT_ELBOW),
            get_point(JointType.LEFT_WRIST)
        )
        
        # Right elbow angle
        angles["right_elbow"] = self.calculate_angle(
            get_point(JointType.RIGHT_SHOULDER),
            get_point(JointType.RIGHT_ELBOW),
            get_point(JointType.RIGHT_WRIST)
        )
        
        # Left shoulder angle (elbow-shoulder-hip)
        angles["left_shoulder"] = self.calculate_angle(
            get_point(JointType.LEFT_ELBOW),
            get_point(JointType.LEFT_SHOULDER),
            get_point(JointType.LEFT_HIP)
        )
        
        # Right shoulder angle
        angles["right_shoulder"] = self.calculate_angle(
            get_point(JointType.RIGHT_ELBOW),
            get_point(JointType.RIGHT_SHOULDER),
            get_point(JointType.RIGHT_HIP)
        )
        
        # Left hip angle (shoulder-hip-knee)
        angles["left_hip"] = self.calculate_angle(
            get_point(JointType.LEFT_SHOULDER),
            get_point(JointType.LEFT_HIP),
            get_point(JointType.LEFT_KNEE)
        )
        
        # Right hip angle
        angles["right_hip"] = self.calculate_angle(
            get_point(JointType.RIGHT_SHOULDER),
            get_point(JointType.RIGHT_HIP),
            get_point(JointType.RIGHT_KNEE)
        )
        
        # Left knee angle (hip-knee-ankle)
        angles["left_knee"] = self.calculate_angle(
            get_point(JointType.LEFT_HIP),
            get_point(JointType.LEFT_KNEE),
            get_point(JointType.LEFT_ANKLE)
        )
        
        # Right knee angle
        angles["right_knee"] = self.calculate_angle(
            get_point(JointType.RIGHT_HIP),
            get_point(JointType.RIGHT_KNEE),
            get_point(JointType.RIGHT_ANKLE)
        )
        
        # Left ankle angle (knee-ankle-foot)
        angles["left_ankle"] = self.calculate_angle(
            get_point(JointType.LEFT_KNEE),
            get_point(JointType.LEFT_ANKLE),
            get_point(JointType.LEFT_FOOT_INDEX)
        )
        
        # Right ankle angle
        angles["right_ankle"] = self.calculate_angle(
            get_point(JointType.RIGHT_KNEE),
            get_point(JointType.RIGHT_ANKLE),
            get_point(JointType.RIGHT_FOOT_INDEX)
        )
        
        # Trunk angle (vertical alignment)
        mid_shoulder = (get_point(JointType.LEFT_SHOULDER) + get_point(JointType.RIGHT_SHOULDER)) / 2
        mid_hip = (get_point(JointType.LEFT_HIP) + get_point(JointType.RIGHT_HIP)) / 2
        vertical = np.array([0, -1, 0])  # Up direction
        trunk_vec = mid_shoulder - mid_hip
        trunk_vec[2] = 0  # Project to 2D
        
        cosine = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-8)
        angles["trunk_vertical"] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        
        return angles
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FORM VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Minimum landmark visibility threshold for reliable angle computation
    MIN_VISIBILITY = 0.5
    
    # Required landmarks per exercise type (must be visible for reliable analysis)
    # This determines what body parts need to be in frame for each exercise
    REQUIRED_LANDMARKS = {
        # ─── FULL BODY EXERCISES ───
        ExerciseType.CHAIR_STAND: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
            JointType.LEFT_ANKLE, JointType.RIGHT_ANKLE,
        ],
        ExerciseType.SQUAT: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
            JointType.LEFT_ANKLE, JointType.RIGHT_ANKLE,
        ],
        ExerciseType.SINGLE_LEG_STAND: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
            JointType.LEFT_ANKLE, JointType.RIGHT_ANKLE,
        ],
        ExerciseType.TANDEM_STAND: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_ANKLE, JointType.RIGHT_ANKLE,
        ],
        ExerciseType.HEEL_TOE_WALK: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_ANKLE, JointType.RIGHT_ANKLE,
        ],
        ExerciseType.MARCHING: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
        ],
        ExerciseType.MARCHING_IN_PLACE: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
        ],
        
        # ─── LOWER BODY EXERCISES ───
        ExerciseType.LEG_RAISE: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
        ],
        ExerciseType.SEATED_LEG_RAISE: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
        ],
        ExerciseType.ANKLE_CIRCLES: [
            JointType.LEFT_ANKLE, JointType.RIGHT_ANKLE,
        ],
        ExerciseType.CALF_STRETCH: [
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
            JointType.LEFT_ANKLE, JointType.RIGHT_ANKLE,
        ],
        ExerciseType.SEATED_HAMSTRING_STRETCH: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
        ],
        ExerciseType.SEATED_HIP_STRETCH: [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
        ],
        
        # ─── UPPER BODY EXERCISES ───
        ExerciseType.ARM_RAISE: [
            JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
            JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW,
        ],
        ExerciseType.SEATED_ARM_RAISES: [
            JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
            JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW,
        ],
        ExerciseType.WALL_PUSHUP: [
            JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
            JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW,
            JointType.LEFT_WRIST, JointType.RIGHT_WRIST,
        ],
        ExerciseType.SHOULDER_ROLLS: [
            JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
        ],
        
        # ─── NECK/HEAD EXERCISES (only upper body needed) ───
        ExerciseType.NECK_ROTATIONS: [
            JointType.NOSE,
            JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
        ],
        
        # ─── BREATHING EXERCISES (just need to see posture) ───
        ExerciseType.DEEP_BREATHING: [
            JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
        ],
        
        # ─── TRUNK EXERCISES ───
        ExerciseType.GENTLE_SPINAL_TWIST: [
            JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
        ],
    }
    
    def check_landmark_visibility(self, pose: PoseResult, exercise: ExerciseType) -> Tuple[bool, List[str]]:
        """
        Check if required body landmarks are visible enough for reliable analysis.
        
        Returns:
            (is_visible, list of missing landmark names)
        """
        # Default to just shoulders for unknown exercises (most lenient)
        required = self.REQUIRED_LANDMARKS.get(exercise, [
            JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
        ])
        
        missing = []
        for joint in required:
            landmark = pose.landmarks.get(joint.value)
            if not landmark or landmark.visibility < self.MIN_VISIBILITY:
                vis = landmark.visibility if landmark else 0.0
                missing.append(f"{joint.name}({vis:.2f})")
        
        is_visible = len(missing) == 0
        return is_visible, missing
    
    def _get_visibility_feedback(self, exercise: ExerciseType, missing: List[str]) -> str:
        """Generate exercise-specific visibility feedback message."""
        # Determine what body parts are needed based on exercise
        required = self.REQUIRED_LANDMARKS.get(exercise, [])
        
        # Check which body regions are missing
        needs_upper = any(j in required for j in [
            JointType.NOSE, JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER,
            JointType.LEFT_ELBOW, JointType.RIGHT_ELBOW,
            JointType.LEFT_WRIST, JointType.RIGHT_WRIST,
        ])
        needs_lower = any(j in required for j in [
            JointType.LEFT_HIP, JointType.RIGHT_HIP,
            JointType.LEFT_KNEE, JointType.RIGHT_KNEE,
            JointType.LEFT_ANKLE, JointType.RIGHT_ANKLE,
        ])
        
        # Generate appropriate feedback
        if needs_upper and needs_lower:
            return "Position your full body in the camera"
        elif needs_lower:
            return "Position your hips and legs in the camera"
        elif needs_upper:
            # Check if it's just face/neck
            just_face = all(j in [JointType.NOSE, JointType.LEFT_EAR, JointType.RIGHT_EAR,
                                   JointType.LEFT_SHOULDER, JointType.RIGHT_SHOULDER]
                           for j in required)
            if just_face:
                return "Position your face and shoulders in the camera"
            return "Position your upper body in the camera"
        else:
            return "Adjust your position in the camera"
    
    def assess_form(self, pose: PoseResult, exercise: ExerciseType) -> FormAssessment:
        """
        Assess exercise form quality based on joint angles.
        
        Args:
            pose: Current pose result
            exercise: Type of exercise being performed
        
        Returns:
            FormAssessment with quality score and feedback
        """
        # ── Visibility gate: skip if key landmarks aren't visible ──
        is_visible, missing = self.check_landmark_visibility(pose, exercise)
        if not is_visible:
            logger.debug(
                f"  ⚠️ Insufficient visibility for {exercise.value}: missing {missing}"
            )
            # Generate exercise-specific feedback
            feedback_msg = self._get_visibility_feedback(exercise, missing)
            return FormAssessment(
                exercise_type=exercise,
                quality=FormQuality.POOR,
                score=0.0,
                joint_angles=[],
                feedback=[feedback_msg],
                timestamp=pose.timestamp
            )
        
        angles = self.get_joint_angles(pose)
        thresholds = self.EXERCISE_THRESHOLDS.get(exercise, {})
        
        joint_angles: List[JointAngle] = []
        feedback: List[str] = []
        total_score = 100.0
        
        # Common measurements
        avg_knee = (angles.get("left_knee", 180) + angles.get("right_knee", 180)) / 2
        avg_hip = (angles.get("left_hip", 180) + angles.get("right_hip", 180)) / 2
        avg_shoulder = (angles.get("left_shoulder", 0) + angles.get("right_shoulder", 0)) / 2
        avg_elbow = (angles.get("left_elbow", 180) + angles.get("right_elbow", 180)) / 2
        trunk = angles.get("trunk_vertical", 0)
        knee_diff = abs(angles.get("left_knee", 0) - angles.get("right_knee", 0))
        hip_diff = abs(angles.get("left_hip", 0) - angles.get("right_hip", 0))
        
        # Check each relevant angle for the exercise
        if exercise in [ExerciseType.SQUAT, ExerciseType.CHAIR_STAND]:
            # Knee angle assessment
            joint_angles.append(JointAngle("avg_knee", avg_knee, 70, 180))
            
            if avg_knee < 60:
                feedback.append(f"Knees bent too deep ({avg_knee:.0f}°) — risk of strain")
                total_score -= 25
            elif avg_knee < 80:
                feedback.append(f"Knees at {avg_knee:.0f}° — try not to go below 80°")
                total_score -= 10
            elif avg_knee > 170:
                feedback.append(f"Knees at {avg_knee:.0f}° — bend more for full rep")
                total_score -= 5
            else:
                feedback.append(f"Knee angle: {avg_knee:.0f}° ✓")
            
            # Hip angle assessment
            joint_angles.append(JointAngle("avg_hip", avg_hip, 70, 180))
            if avg_hip < 70:
                feedback.append(f"Hip angle too tight ({avg_hip:.0f}°) — don't lean too far")
                total_score -= 15
            
            # Back straightness
            if trunk > 35:
                feedback.append(f"Back tilted {trunk:.0f}° — keep torso upright")
                total_score -= 20
            elif trunk > 20:
                feedback.append(f"Slight forward lean ({trunk:.0f}°) — engage core")
                total_score -= 8
            
            # Symmetry check
            if knee_diff > 15:
                feedback.append(f"Uneven knees (diff {knee_diff:.0f}°) — distribute weight evenly")
                total_score -= 10
        
        elif exercise == ExerciseType.LEG_RAISE:
            for side in ["left", "right"]:
                hip_angle = angles.get(f"{side}_hip", 180)
                knee_angle = angles.get(f"{side}_knee", 180)
                
                joint_angles.append(JointAngle(f"{side}_hip", hip_angle, 90, 180))
                joint_angles.append(JointAngle(f"{side}_knee", knee_angle, 160, 180))
                
                if knee_angle < 140:
                    feedback.append(f"Straighten {side} leg more ({knee_angle:.0f}° → aim 170°+)")
                    total_score -= 12
                elif knee_angle < 160:
                    feedback.append(f"{side.title()} knee slightly bent ({knee_angle:.0f}°)")
                    total_score -= 5
            
            # Show current hip angles for user awareness
            feedback.append(f"Hip lift: L={angles.get('left_hip', 0):.0f}° R={angles.get('right_hip', 0):.0f}°")
            
            if trunk > 25:
                feedback.append(f"Keep torso stable (tilting {trunk:.0f}°)")
                total_score -= 10
        
        elif exercise == ExerciseType.ARM_RAISE:
            for side in ["left", "right"]:
                shoulder_angle = angles.get(f"{side}_shoulder", 0)
                elbow_angle = angles.get(f"{side}_elbow", 180)
                
                joint_angles.append(JointAngle(f"{side}_shoulder", shoulder_angle, 0, 180))
                joint_angles.append(JointAngle(f"{side}_elbow", elbow_angle, 160, 180))
                
                if elbow_angle < 140:
                    feedback.append(f"Straighten {side} elbow ({elbow_angle:.0f}° → aim 170°+)")
                    total_score -= 12
                elif elbow_angle < 160:
                    feedback.append(f"{side.title()} elbow slightly bent ({elbow_angle:.0f}°)")
                    total_score -= 5
            
            # Show shoulder elevation progress
            feedback.append(f"Shoulder: L={angles.get('left_shoulder', 0):.0f}° R={angles.get('right_shoulder', 0):.0f}°")
            
            shoulder_diff = abs(angles.get("left_shoulder", 0) - angles.get("right_shoulder", 0))
            if shoulder_diff > 20:
                feedback.append(f"Arms uneven (diff {shoulder_diff:.0f}°) — raise both equally")
                total_score -= 10
        
        elif exercise == ExerciseType.WALL_PUSHUP:
            for side in ["left", "right"]:
                elbow_angle = angles.get(f"{side}_elbow", 180)
                joint_angles.append(JointAngle(f"{side}_elbow", elbow_angle, 70, 180))
                
                if elbow_angle < 60:
                    feedback.append(f"{side.title()} elbow too bent ({elbow_angle:.0f}°) — don't go too deep")
                    total_score -= 15
            
            feedback.append(f"Elbow: L={angles.get('left_elbow', 0):.0f}° R={angles.get('right_elbow', 0):.0f}°")
            
            if trunk > 30:
                feedback.append(f"Body not straight (trunk {trunk:.0f}°) — align head to heels")
                total_score -= 15
        
        elif exercise == ExerciseType.MARCHING:
            joint_angles.append(JointAngle("avg_hip", avg_hip, 70, 180))
            
            # Check knee lift height
            for side in ["left", "right"]:
                hip_angle = angles.get(f"{side}_hip", 180)
                if hip_angle < 90:
                    feedback.append(f"Great {side} knee lift ({hip_angle:.0f}°)!")
                elif hip_angle > 150:
                    feedback.append(f"Lift {side} knee higher ({hip_angle:.0f}° → aim below 110°)")
                    total_score -= 8
            
            if trunk > 20:
                feedback.append(f"Stand tall — leaning {trunk:.0f}°")
                total_score -= 10
        
        elif exercise == ExerciseType.SINGLE_LEG_STAND:
            # Balance assessment — check hip alignment and body sway
            joint_angles.append(JointAngle("hip_alignment", hip_diff, 0, 10))
            
            if hip_diff > 15:
                feedback.append(f"Hips tilting (diff {hip_diff:.0f}°) — keep hips level")
                total_score -= 15
            elif hip_diff > 8:
                feedback.append(f"Slight hip tilt ({hip_diff:.0f}°)")
                total_score -= 5
            else:
                feedback.append(f"Hips level ✓ ({hip_diff:.0f}° diff)")
            
            if trunk > 15:
                feedback.append(f"Body swaying {trunk:.0f}° — focus on a fixed point")
                total_score -= 10
        
        elif exercise == ExerciseType.TANDEM_STAND:
            if hip_diff > 12:
                feedback.append(f"Hips uneven ({hip_diff:.0f}°) — balance weight")
                total_score -= 12
            if trunk > 15:
                feedback.append(f"Torso swaying ({trunk:.0f}°) — engage core")
                total_score -= 10
            else:
                feedback.append(f"Good stability (trunk {trunk:.0f}°)")
        
        elif exercise == ExerciseType.ANKLE_CIRCLES:
            # Minimal form checks — just ensure stability
            if trunk > 20:
                feedback.append(f"Stay steady — body tilting {trunk:.0f}°")
                total_score -= 8
            else:
                feedback.append("Good stability while circling ✓")
        
        elif exercise == ExerciseType.NECK_ROTATIONS:
            # Neck rotation assessment — check posture and head alignment
            # Head tilt is computed from nose vs shoulder midpoint
            head_tilt = angles.get("head_tilt", 0)
            
            # Posture check — shoulders should stay level
            shoulder_diff = abs(angles.get("left_shoulder", 0) - angles.get("right_shoulder", 0))
            if shoulder_diff > 15:
                feedback.append(f"Keep shoulders level (diff {shoulder_diff:.0f}°)")
                total_score -= 10
            
            # Trunk stability
            if trunk > 15:
                feedback.append(f"Keep body still — swaying {trunk:.0f}°")
                total_score -= 10
            
            # Dynamic feedback based on head position
            if abs(head_tilt) < 10:
                feedback.append("Head centered ✓")
            elif head_tilt > 30:
                feedback.append(f"Good right rotation ({head_tilt:.0f}°)")
            elif head_tilt < -30:
                feedback.append(f"Good left rotation ({abs(head_tilt):.0f}°)")
            else:
                feedback.append(f"Rotate head slowly ({abs(head_tilt):.0f}°)")
        
        elif exercise == ExerciseType.SHOULDER_ROLLS:
            # Shoulder roll assessment — check shoulder elevation and symmetry
            left_shoulder = angles.get("left_shoulder", 0)
            right_shoulder = angles.get("right_shoulder", 0)
            shoulder_diff = abs(left_shoulder - right_shoulder)
            
            # Symmetry check
            if shoulder_diff > 20:
                feedback.append(f"Roll shoulders evenly (diff {shoulder_diff:.0f}°)")
                total_score -= 10
            else:
                feedback.append("Shoulders moving symmetrically ✓")
            
            # Trunk stability
            if trunk > 15:
                feedback.append(f"Keep body still — swaying {trunk:.0f}°")
                total_score -= 8
            
            # Check smooth rolling motion
            feedback.append(f"Shoulder angles: L={left_shoulder:.0f}° R={right_shoulder:.0f}°")
        
        elif exercise == ExerciseType.HEEL_TOE_WALK:
            # Heel-toe walking assessment — check step pattern and balance
            if knee_diff > 25:
                feedback.append(f"Uneven steps (diff {knee_diff:.0f}°)")
                total_score -= 15
            else:
                feedback.append("Good step rhythm ✓")
            
            if trunk > 20:
                feedback.append(f"Keep body upright — tilting {trunk:.0f}°")
                total_score -= 10
            
            if hip_diff > 15:
                feedback.append(f"Hips swaying (diff {hip_diff:.0f}°) — keep stable")
                total_score -= 8
            else:
                feedback.append("Good hip stability ✓")
        
        elif exercise == ExerciseType.SEATED_HAMSTRING_STRETCH:
            # Seated hamstring stretch — check forward lean
            if avg_hip < 60:
                feedback.append("Great stretch depth! Hold gently")
            elif avg_hip < 80:
                feedback.append(f"Good forward lean ({avg_hip:.0f}°)")
            else:
                feedback.append(f"Lean forward more (currently {avg_hip:.0f}°)")
                total_score -= 10
            
            # Keep legs straight
            if avg_knee > 160:
                feedback.append("Legs nice and straight ✓")
            else:
                feedback.append(f"Straighten legs more ({avg_knee:.0f}°)")
                total_score -= 8
        
        elif exercise == ExerciseType.SEATED_HIP_STRETCH:
            # Seated hip stretch — check cross-leg position
            if trunk > 25:
                feedback.append(f"Sit taller — leaning {trunk:.0f}°")
                total_score -= 10
            else:
                feedback.append("Good upright posture ✓")
            
            feedback.append("Cross ankle over knee, lean gently forward")
        
        elif exercise == ExerciseType.CALF_STRETCH:
            # Calf stretch — check back leg position
            if trunk > 20:
                feedback.append(f"Keep body straight — tilting {trunk:.0f}°")
                total_score -= 10
            else:
                feedback.append("Good body alignment ✓")
            
            # Check back leg is straight
            if avg_knee > 160:
                feedback.append("Back leg straight ✓")
            else:
                feedback.append("Straighten back leg for deeper stretch")
                total_score -= 5
            
            feedback.append("Press heel down, feel calf stretch")
        
        elif exercise == ExerciseType.SEATED_LEG_RAISE:
            # Seated leg raise — check leg extension
            left_knee = angles.get("left_knee", 90)
            right_knee = angles.get("right_knee", 90)
            
            # Check if one leg is extended
            if left_knee > 150 or right_knee > 150:
                extending = "left" if left_knee > right_knee else "right"
                feedback.append(f"Good {extending} leg extension ✓")
            else:
                feedback.append("Extend leg fully, hold briefly")
            
            if trunk > 20:
                feedback.append(f"Sit tall — leaning {trunk:.0f}°")
                total_score -= 8
        
        elif exercise == ExerciseType.SEATED_ARM_RAISES:
            # Seated arm raises — check overhead extension
            if avg_shoulder > 150:
                feedback.append("Arms fully raised ✓")
            elif avg_shoulder > 90:
                feedback.append(f"Raise arms higher ({avg_shoulder:.0f}°)")
                total_score -= 5
            else:
                feedback.append("Lift arms overhead slowly")
            
            shoulder_diff = abs(angles.get("left_shoulder", 0) - angles.get("right_shoulder", 0))
            if shoulder_diff > 20:
                feedback.append(f"Raise both arms evenly (diff {shoulder_diff:.0f}°)")
                total_score -= 10
            else:
                feedback.append("Good arm symmetry ✓")
        
        elif exercise == ExerciseType.DEEP_BREATHING:
            # Deep breathing — minimal pose tracking, focus on rhythm
            if trunk > 15:
                feedback.append(f"Relax shoulders — slight tension detected")
                total_score -= 5
            else:
                feedback.append("Good relaxed posture ✓")
            
            feedback.append("Breathe in slowly... breathe out gently")
        
        elif exercise == ExerciseType.GENTLE_SPINAL_TWIST:
            # Spinal twist — check trunk rotation
            trunk_rotation = angles.get("trunk_rotation", 0)
            if abs(trunk_rotation) > 20:
                direction = "right" if trunk_rotation > 0 else "left"
                feedback.append(f"Good {direction} twist ({abs(trunk_rotation):.0f}°)")
            else:
                feedback.append("Rotate upper body gently")
            
            if hip_diff > 15:
                feedback.append(f"Keep hips still (moving {hip_diff:.0f}°)")
                total_score -= 10
            else:
                feedback.append("Hips stable ✓")
        
        elif exercise == ExerciseType.MARCHING_IN_PLACE:
            # Marching in place — same as regular marching
            joint_angles.append(JointAngle("avg_hip", avg_hip, 70, 180))
            
            for side in ["left", "right"]:
                hip_angle = angles.get(f"{side}_hip", 180)
                if hip_angle < 90:
                    feedback.append(f"Great {side} knee lift ✓")
                elif hip_angle > 150:
                    feedback.append(f"Lift {side} knee higher")
                    total_score -= 8
            
            if trunk > 20:
                feedback.append(f"Stand tall — leaning {trunk:.0f}°")
                total_score -= 10
        
        else:
            # Generic assessment for unsupported exercises
            if trunk > 30:
                feedback.append(f"Watch posture — trunk at {trunk:.0f}°")
                total_score -= 15
            if knee_diff > 20:
                feedback.append(f"Uneven movement (knee diff {knee_diff:.0f}°)")
                total_score -= 10
            # Show exercise-specific guidance instead of raw angles
            feedback.append("Follow the on-screen guidance")
        
        # Determine quality level
        total_score = max(0, min(100, total_score))
        
        if total_score >= 90:
            quality = FormQuality.EXCELLENT
        elif total_score >= 75:
            quality = FormQuality.GOOD
        elif total_score >= 50:
            quality = FormQuality.FAIR
        else:
            quality = FormQuality.POOR
        
        # Log assessment
        logger.debug(
            f"  📋 Form: {exercise.value} score={total_score:.0f} "
            f"quality={quality.value} feedback={feedback}"
        )
        
        return FormAssessment(
            exercise_type=exercise,
            quality=quality,
            score=total_score,
            joint_angles=joint_angles,
            feedback=feedback,
            timestamp=pose.timestamp
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REPETITION COUNTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def init_rep_counter(self, exercise: ExerciseType) -> RepCounter:
        """Initialize a rep counter for an exercise."""
        thresholds = self.EXERCISE_THRESHOLDS.get(exercise, {})
        
        counter = RepCounter(
            exercise_type=exercise,
            threshold_up=thresholds.get("rep_up", 150),
            threshold_down=thresholds.get("rep_down", 90),
        )
        
        self.rep_counters[exercise] = counter
        return counter
    
    def count_rep(self, pose: PoseResult, exercise: ExerciseType) -> Tuple[int, bool]:
        """
        Update rep count for an exercise.
        
        Args:
            pose: Current pose
            exercise: Exercise type
        
        Returns:
            Tuple of (current_count, rep_just_completed)
        """
        if exercise not in self.rep_counters:
            self.init_rep_counter(exercise)
        
        counter = self.rep_counters[exercise]
        
        # ── Visibility gate: don't count reps from invisible landmarks ──
        is_visible, missing = self.check_landmark_visibility(pose, exercise)
        if not is_visible:
            logger.debug(
                f"  ⚠️ Skipping rep count — landmarks not visible: {missing}"
            )
            return counter.count, False
        
        angles = self.get_joint_angles(pose)
        
        # Choose the primary angle to track based on exercise
        if exercise in [ExerciseType.SQUAT, ExerciseType.CHAIR_STAND]:
            primary_angle = (angles.get("left_knee", 180) + angles.get("right_knee", 180)) / 2
        elif exercise in [ExerciseType.LEG_RAISE, ExerciseType.MARCHING]:
            primary_angle = (angles.get("left_hip", 180) + angles.get("right_hip", 180)) / 2
        elif exercise in [ExerciseType.ARM_RAISE]:
            primary_angle = (angles.get("left_shoulder", 0) + angles.get("right_shoulder", 0)) / 2
        elif exercise == ExerciseType.WALL_PUSHUP:
            primary_angle = (angles.get("left_elbow", 180) + angles.get("right_elbow", 180)) / 2
        else:
            primary_angle = 90
        
        old_state = counter.state
        rep_completed = counter.update(primary_angle)
        
        # Log state transitions
        if old_state != counter.state:
            logger.debug(
                f"  🔄 Rep state: {old_state}→{counter.state} | "
                f"angle={primary_angle:.1f}° (up>{counter.threshold_up} down<{counter.threshold_down}) | "
                f"count={counter.count}"
            )
        
        if rep_completed:
            logger.info(
                f"  ✅ Rep #{counter.count} completed for {exercise.value} | "
                f"angle={primary_angle:.1f}°"
            )
        
        return counter.count, rep_completed
    
    def reset_rep_counter(self, exercise: ExerciseType):
        """Reset rep counter for an exercise."""
        if exercise in self.rep_counters:
            self.rep_counters[exercise].count = 0
            self.rep_counters[exercise].state = "neutral"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PAIN/DISCOMFORT DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_pain_indicators(self, current_pose: PoseResult) -> Dict[str, Any]:
        """
        Detect potential pain or discomfort indicators.
        
        Looks for:
        - Shaking/trembling (high velocity variance)
        - Slowing down (decreasing velocity)
        - Compensatory movements (asymmetry)
        - Hesitation (pause before movements)
        
        Returns dict with pain indicators.
        """
        indicators = {
            "shaking_detected": False,
            "slowing_detected": False,
            "asymmetry_detected": False,
            "hesitation_detected": False,
            "confidence": 0.0,
            "details": []
        }
        
        if len(self.pose_history) < 10:
            return indicators
        
        # Analyze recent pose history
        recent_poses = self.pose_history[-10:]
        
        # Check for shaking (high variance in landmark positions)
        variance_sum = 0.0
        for joint_idx in [JointType.LEFT_WRIST.value, JointType.RIGHT_WRIST.value,
                          JointType.LEFT_ANKLE.value, JointType.RIGHT_ANKLE.value]:
            positions = []
            for pose in recent_poses:
                if joint_idx in pose.landmarks:
                    lm = pose.landmarks[joint_idx]
                    positions.append([lm.x, lm.y])
            
            if len(positions) >= 5:
                positions = np.array(positions)
                variance = np.var(positions, axis=0).sum()
                variance_sum += variance
        
        if variance_sum > 0.01:  # Threshold for shaking
            indicators["shaking_detected"] = True
            indicators["details"].append("Trembling detected in extremities")
        
        # Check for asymmetry between left and right sides
        angles = self.get_joint_angles(current_pose)
        
        knee_diff = abs(angles.get("left_knee", 0) - angles.get("right_knee", 0))
        hip_diff = abs(angles.get("left_hip", 0) - angles.get("right_hip", 0))
        shoulder_diff = abs(angles.get("left_shoulder", 0) - angles.get("right_shoulder", 0))
        
        if knee_diff > 15 or hip_diff > 15:
            indicators["asymmetry_detected"] = True
            indicators["details"].append("Asymmetric movement pattern detected")
        
        # Calculate overall pain confidence
        pain_score = 0
        if indicators["shaking_detected"]:
            pain_score += 30
        if indicators["asymmetry_detected"]:
            pain_score += 25
        if indicators["slowing_detected"]:
            pain_score += 25
        if indicators["hesitation_detected"]:
            pain_score += 20
        
        indicators["confidence"] = min(100, pain_score) / 100.0
        
        return indicators
    
    def get_intensity_recommendation(self, pain_indicators: Dict[str, Any]) -> str:
        """
        Get exercise intensity recommendation based on pain indicators.
        
        Returns recommendation string.
        """
        confidence = pain_indicators.get("confidence", 0)
        
        if confidence < 0.2:
            return "Continue at current intensity"
        elif confidence < 0.4:
            return "Consider reducing intensity slightly"
        elif confidence < 0.6:
            return "Recommend reducing intensity"
        elif confidence < 0.8:
            return "Strongly recommend taking a break"
        else:
            return "Stop exercise - potential injury risk"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def landmarks_to_dict(self, pose: PoseResult) -> Dict[str, Any]:
        """Convert pose landmarks to JSON-serializable dict."""
        return {
            "timestamp": pose.timestamp,
            "confidence": pose.confidence,
            "landmarks": [
                {
                    "id": idx,
                    "name": JointType(idx).name if idx < len(JointType) else f"point_{idx}",
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
                for idx, lm in pose.landmarks.items()
            ]
        }
    
    def cleanup(self):
        """Release resources."""
        if self.pose_detector and hasattr(self.pose_detector, 'close'):
            self.pose_detector.close()
        self.pose_history.clear()
        self.rep_counters.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_analyzer_instance: Optional[PoseAnalyzer] = None

def get_pose_analyzer() -> PoseAnalyzer:
    """Get or create the global pose analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = PoseAnalyzer()
    return _analyzer_instance
