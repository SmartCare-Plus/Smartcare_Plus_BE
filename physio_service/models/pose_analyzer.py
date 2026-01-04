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

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Using mock pose data.")


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
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize pose analyzer.
        
        Args:
            model_path: Path to MediaPipe pose model (uses default if None)
        """
        self.pose_detector = None
        self.rep_counters: Dict[ExerciseType, RepCounter] = {}
        self.pose_history: List[PoseResult] = []
        self.max_history = 30  # For smoothing and velocity calculation
        
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe(model_path)
    
    def _init_mediapipe(self, model_path: Optional[str] = None):
        """Initialize MediaPipe pose detector."""
        try:
            if model_path:
                # Use custom model
                base_options = mp_python.BaseOptions(model_asset_path=model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.pose_detector = vision.PoseLandmarker.create_from_options(options)
            else:
                # Use MediaPipe legacy API for simplicity
                self.mp_pose = mp.solutions.pose
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            print("✅ MediaPipe pose detector initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize MediaPipe: {e}")
            self.pose_detector = None
    
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
            return self._generate_mock_pose(timestamp_ms)
        
        try:
            # Convert to MediaPipe format
            results = self.pose_detector.process(image)
            
            if not results.pose_landmarks:
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
            
            result = PoseResult(
                landmarks=landmarks,
                timestamp=timestamp_ms,
                confidence=confidence,
                world_landmarks=world_landmarks
            )
            
            # Add to history
            self.pose_history.append(result)
            if len(self.pose_history) > self.max_history:
                self.pose_history.pop(0)
            
            return result
            
        except Exception as e:
            print(f"Pose detection error: {e}")
            return None
    
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
    
    def assess_form(self, pose: PoseResult, exercise: ExerciseType) -> FormAssessment:
        """
        Assess exercise form quality based on joint angles.
        
        Args:
            pose: Current pose result
            exercise: Type of exercise being performed
        
        Returns:
            FormAssessment with quality score and feedback
        """
        angles = self.get_joint_angles(pose)
        thresholds = self.EXERCISE_THRESHOLDS.get(exercise, {})
        
        joint_angles: List[JointAngle] = []
        feedback: List[str] = []
        total_score = 100.0
        
        # Check each relevant angle for the exercise
        if exercise == ExerciseType.SQUAT:
            # Check knee angles
            for side in ["left", "right"]:
                knee_angle = angles.get(f"{side}_knee", 180)
                ja = JointAngle(
                    name=f"{side}_knee",
                    angle=knee_angle,
                    reference_min=70,
                    reference_max=180
                )
                joint_angles.append(ja)
                
                if knee_angle < 70:
                    feedback.append(f"Don't bend {side} knee too deeply")
                    total_score -= 15
            
            # Check back straightness
            trunk = angles.get("trunk_vertical", 0)
            if trunk > 30:
                feedback.append("Keep your back straighter")
                total_score -= 20
        
        elif exercise == ExerciseType.CHAIR_STAND:
            # Similar checks for chair stand
            avg_knee = (angles.get("left_knee", 180) + angles.get("right_knee", 180)) / 2
            ja = JointAngle(
                name="avg_knee",
                angle=avg_knee,
                reference_min=70,
                reference_max=180
            )
            joint_angles.append(ja)
            
            if avg_knee < 80:
                feedback.append("Don't sit too low")
                total_score -= 10
        
        elif exercise == ExerciseType.LEG_RAISE:
            for side in ["left", "right"]:
                hip_angle = angles.get(f"{side}_hip", 180)
                knee_angle = angles.get(f"{side}_knee", 180)
                
                joint_angles.append(JointAngle(f"{side}_hip", hip_angle, 90, 180))
                joint_angles.append(JointAngle(f"{side}_knee", knee_angle, 160, 180))
                
                if knee_angle < 150:
                    feedback.append(f"Keep {side} leg straighter")
                    total_score -= 10
        
        elif exercise == ExerciseType.ARM_RAISE:
            for side in ["left", "right"]:
                shoulder_angle = angles.get(f"{side}_shoulder", 0)
                elbow_angle = angles.get(f"{side}_elbow", 180)
                
                joint_angles.append(JointAngle(f"{side}_shoulder", shoulder_angle, 0, 180))
                joint_angles.append(JointAngle(f"{side}_elbow", elbow_angle, 160, 180))
                
                if elbow_angle < 150:
                    feedback.append(f"Straighten {side} elbow more")
                    total_score -= 10
        
        elif exercise == ExerciseType.WALL_PUSHUP:
            for side in ["left", "right"]:
                elbow_angle = angles.get(f"{side}_elbow", 180)
                joint_angles.append(JointAngle(f"{side}_elbow", elbow_angle, 70, 180))
        
        # Determine quality level
        total_score = max(0, min(100, total_score))
        
        if total_score >= 90:
            quality = FormQuality.EXCELLENT
            if not feedback:
                feedback.append("Excellent form! Keep it up!")
        elif total_score >= 75:
            quality = FormQuality.GOOD
            if not feedback:
                feedback.append("Good form")
        elif total_score >= 50:
            quality = FormQuality.FAIR
        else:
            quality = FormQuality.POOR
        
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
        
        rep_completed = counter.update(primary_angle)
        
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
