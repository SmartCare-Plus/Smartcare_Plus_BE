"""
SMARTCARE+ Guardian Service - TUG Test Handler

Owner: Madhushani
Timed Up and Go (TUG) test processing for mobility assessment.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class TUGPhase(Enum):
    """Phases of the TUG test."""
    NOT_STARTED = "not_started"
    SIT_TO_STAND = "sit_to_stand"
    WALK_OUT = "walk_out"
    TURN_AROUND = "turn_around"
    WALK_BACK = "walk_back"
    TURN_AND_SIT = "turn_and_sit"
    COMPLETED = "completed"


class MobilityLevel(Enum):
    """Mobility assessment levels based on TUG time."""
    NORMAL = "normal"  # < 10 seconds
    MILD_IMPAIRMENT = "mild_impairment"  # 10-19 seconds
    MODERATE_IMPAIRMENT = "moderate_impairment"  # 20-29 seconds
    SEVERE_IMPAIRMENT = "severe_impairment"  # >= 30 seconds


class ActivityType(Enum):
    """Detected activity types."""
    SITTING = "sitting"
    STANDING = "standing"
    WALKING = "walking"
    TURNING = "turning"
    LYING = "lying"
    FALLING = "falling"
    UNKNOWN = "unknown"


@dataclass
class TUGPhaseResult:
    """Result for a single TUG phase."""
    phase: TUGPhase
    duration_seconds: float
    start_time: float
    end_time: float
    quality_score: float = 1.0  # 0-1, lower if issues detected
    notes: List[str] = field(default_factory=list)


@dataclass
class TUGTestResult:
    """Complete TUG test result."""
    test_id: str
    user_id: str
    total_time_seconds: float
    mobility_level: MobilityLevel
    fall_risk_score: float  # 0-100
    
    # Phase breakdown
    phases: List[TUGPhaseResult] = field(default_factory=list)
    
    # Individual phase times
    sit_to_stand_time: float = 0.0
    walk_out_time: float = 0.0
    turn_time: float = 0.0
    walk_back_time: float = 0.0
    turn_and_sit_time: float = 0.0
    
    # Quality indicators
    used_armrest: bool = False
    needed_assistance: bool = False
    hesitations_detected: int = 0
    stumbles_detected: int = 0
    
    # Interpretation
    interpretation: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    # Timestamps
    start_time: float = 0.0
    end_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "user_id": self.user_id,
            "total_time_seconds": round(self.total_time_seconds, 1),
            "mobility_level": self.mobility_level.value,
            "fall_risk_score": round(self.fall_risk_score, 1),
            "phases": {
                "sit_to_stand": round(self.sit_to_stand_time, 1),
                "walk_out": round(self.walk_out_time, 1),
                "turn_around": round(self.turn_time, 1),
                "walk_back": round(self.walk_back_time, 1),
                "turn_and_sit": round(self.turn_and_sit_time, 1),
            },
            "quality": {
                "used_armrest": self.used_armrest,
                "needed_assistance": self.needed_assistance,
                "hesitations": self.hesitations_detected,
                "stumbles": self.stumbles_detected,
            },
            "interpretation": self.interpretation,
            "recommendations": self.recommendations,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVITY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class ActivityClassifier:
    """
    Rule-based activity classification from pose landmarks.
    
    Uses joint positions and angles to determine current activity.
    """
    
    # Thresholds for activity detection (normalized coordinates)
    SITTING_HIP_Y_THRESHOLD = 0.6  # Hips above this = likely sitting
    STANDING_KNEE_ANGLE = 160  # Knee angle when standing (degrees)
    WALKING_VELOCITY_THRESHOLD = 0.02  # Movement per frame
    
    def __init__(self):
        """Initialize classifier."""
        self.position_history: List[Dict] = []
        self.max_history = 30
        self.last_activity = ActivityType.UNKNOWN
    
    def classify(self, landmarks: Dict[int, Any], timestamp: float) -> ActivityType:
        """
        Classify current activity from pose landmarks.
        
        Args:
            landmarks: Dict of landmark ID -> {x, y, z, visibility}
            timestamp: Frame timestamp
        
        Returns:
            Detected ActivityType
        """
        # Extract key landmarks
        left_hip = landmarks.get(23, {})
        right_hip = landmarks.get(24, {})
        left_knee = landmarks.get(25, {})
        right_knee = landmarks.get(26, {})
        left_ankle = landmarks.get(27, {})
        right_ankle = landmarks.get(28, {})
        left_shoulder = landmarks.get(11, {})
        right_shoulder = landmarks.get(12, {})
        nose = landmarks.get(0, {})
        
        if not all([left_hip, right_hip, left_knee, right_knee]):
            return ActivityType.UNKNOWN
        
        # Calculate key metrics
        hip_y = (left_hip.get("y", 0) + right_hip.get("y", 0)) / 2
        shoulder_y = (left_shoulder.get("y", 0) + right_shoulder.get("y", 0)) / 2 if left_shoulder and right_shoulder else 0
        
        # Store position for velocity calculation
        current_pos = {
            "hip_x": (left_hip.get("x", 0) + right_hip.get("x", 0)) / 2,
            "hip_y": hip_y,
            "timestamp": timestamp
        }
        self.position_history.append(current_pos)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        # Calculate velocity
        velocity = self._calculate_velocity()
        
        # Calculate knee angles
        left_knee_angle = self._calculate_angle(
            left_hip, left_knee, left_ankle
        )
        right_knee_angle = self._calculate_angle(
            right_hip, right_knee, right_ankle
        )
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        # Calculate trunk angle (vertical alignment)
        trunk_angle = self._calculate_trunk_angle(landmarks)
        
        # Classification logic
        
        # Lying down: very low shoulder Y or horizontal trunk
        if shoulder_y > 0.75 or trunk_angle > 70:
            return ActivityType.LYING
        
        # Sitting: hips high in frame, bent knees
        if hip_y < 0.5 and avg_knee_angle < 130:
            return ActivityType.SITTING
        
        # Standing vs Walking
        if avg_knee_angle > 150:
            if velocity > self.WALKING_VELOCITY_THRESHOLD:
                # Check if turning (lateral movement)
                if self._is_turning():
                    return ActivityType.TURNING
                return ActivityType.WALKING
            else:
                return ActivityType.STANDING
        
        # Transition states
        if 130 <= avg_knee_angle <= 150:
            if velocity > 0.01:
                return ActivityType.WALKING
            # Could be mid-sit-to-stand
            if self.last_activity == ActivityType.SITTING:
                return ActivityType.STANDING  # Transitioning
        
        self.last_activity = ActivityType.UNKNOWN
        return ActivityType.UNKNOWN
    
    def _calculate_velocity(self) -> float:
        """Calculate movement velocity from history."""
        if len(self.position_history) < 5:
            return 0.0
        
        recent = self.position_history[-5:]
        dx = recent[-1]["hip_x"] - recent[0]["hip_x"]
        dy = recent[-1]["hip_y"] - recent[0]["hip_y"]
        dt = recent[-1]["timestamp"] - recent[0]["timestamp"]
        
        if dt > 0:
            return np.sqrt(dx**2 + dy**2) / dt
        return 0.0
    
    def _is_turning(self) -> bool:
        """Detect if person is turning based on position history."""
        if len(self.position_history) < 10:
            return False
        
        recent = self.position_history[-10:]
        x_positions = [p["hip_x"] for p in recent]
        
        # Check for direction change
        directions = []
        for i in range(1, len(x_positions)):
            diff = x_positions[i] - x_positions[i-1]
            if abs(diff) > 0.005:
                directions.append(1 if diff > 0 else -1)
        
        if len(directions) >= 3:
            # Direction change indicates turning
            changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])
            return changes >= 2
        
        return False
    
    def _calculate_angle(self, a: Dict, b: Dict, c: Dict) -> float:
        """Calculate angle at point b."""
        if not all([a, b, c]):
            return 180.0
        
        ax, ay = a.get("x", 0), a.get("y", 0)
        bx, by = b.get("x", 0), b.get("y", 0)
        cx, cy = c.get("x", 0), c.get("y", 0)
        
        ba = np.array([ax - bx, ay - by])
        bc = np.array([cx - bx, cy - by])
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        
        return angle
    
    def _calculate_trunk_angle(self, landmarks: Dict) -> float:
        """Calculate trunk angle from vertical."""
        left_shoulder = landmarks.get(11, {})
        right_shoulder = landmarks.get(12, {})
        left_hip = landmarks.get(23, {})
        right_hip = landmarks.get(24, {})
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.0
        
        shoulder_mid = np.array([
            (left_shoulder.get("x", 0) + right_shoulder.get("x", 0)) / 2,
            (left_shoulder.get("y", 0) + right_shoulder.get("y", 0)) / 2
        ])
        
        hip_mid = np.array([
            (left_hip.get("x", 0) + right_hip.get("x", 0)) / 2,
            (left_hip.get("y", 0) + right_hip.get("y", 0)) / 2
        ])
        
        trunk_vec = shoulder_mid - hip_mid
        vertical = np.array([0, -1])  # Up in image coordinates
        
        cosine = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        
        return angle
    
    def reset(self):
        """Reset classifier state."""
        self.position_history.clear()
        self.last_activity = ActivityType.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════════
# TUG TEST HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class TUGTestHandler:
    """
    Handles Timed Up and Go (TUG) test processing.
    
    The TUG test measures:
    1. Time to stand from sitting
    2. Walk 3 meters
    3. Turn around
    4. Walk back
    5. Turn and sit down
    
    Total time predicts fall risk and mobility impairment.
    """
    
    # TUG interpretation thresholds (seconds)
    NORMAL_THRESHOLD = 10
    MILD_THRESHOLD = 20
    MODERATE_THRESHOLD = 30
    
    def __init__(self):
        """Initialize TUG handler."""
        self.activity_classifier = ActivityClassifier()
        self.current_test: Optional[Dict] = None
        self.is_running = False
    
    def start_test(self, test_id: str, user_id: str) -> Dict[str, Any]:
        """
        Start a new TUG test.
        
        Args:
            test_id: Unique test ID
            user_id: User ID
        
        Returns:
            Test start confirmation
        """
        self.current_test = {
            "test_id": test_id,
            "user_id": user_id,
            "start_time": time.time(),
            "current_phase": TUGPhase.NOT_STARTED,
            "phases": [],
            "phase_start_time": None,
            "hesitations": 0,
            "stumbles": 0,
        }
        self.is_running = True
        self.activity_classifier.reset()
        
        return {
            "status": "started",
            "test_id": test_id,
            "message": "TUG test started. Please remain seated until instructed to begin."
        }
    
    def process_frame(self, landmarks: Dict[int, Any], timestamp: float) -> Dict[str, Any]:
        """
        Process a video frame during TUG test.
        
        Args:
            landmarks: Pose landmarks
            timestamp: Frame timestamp
        
        Returns:
            Current test status and phase info
        """
        if not self.is_running or not self.current_test:
            return {"error": "No active test"}
        
        # Classify current activity
        activity = self.activity_classifier.classify(landmarks, timestamp)
        
        # Update phase based on activity sequence
        phase_result = self._update_phase(activity, timestamp)
        
        response = {
            "test_id": self.current_test["test_id"],
            "current_phase": self.current_test["current_phase"].value,
            "activity": activity.value,
            "elapsed_time": round(timestamp - self.current_test["start_time"], 1),
        }
        
        if phase_result:
            response["phase_completed"] = phase_result["phase"]
            response["phase_time"] = phase_result["duration"]
        
        # Check if test completed
        if self.current_test["current_phase"] == TUGPhase.COMPLETED:
            response["test_completed"] = True
        
        return response
    
    def _update_phase(self, activity: ActivityType, timestamp: float) -> Optional[Dict]:
        """Update test phase based on detected activity."""
        current_phase = self.current_test["current_phase"]
        phase_result = None
        
        # Phase transition logic
        if current_phase == TUGPhase.NOT_STARTED:
            if activity == ActivityType.SITTING:
                # Ready to start
                pass
            elif activity in [ActivityType.STANDING, ActivityType.WALKING]:
                # Started standing up
                self._start_phase(TUGPhase.SIT_TO_STAND, timestamp)
        
        elif current_phase == TUGPhase.SIT_TO_STAND:
            if activity == ActivityType.STANDING:
                # Completed standing
                phase_result = self._end_phase(timestamp)
                self._start_phase(TUGPhase.WALK_OUT, timestamp)
            elif activity == ActivityType.WALKING:
                # Started walking directly
                phase_result = self._end_phase(timestamp)
                self._start_phase(TUGPhase.WALK_OUT, timestamp)
        
        elif current_phase == TUGPhase.WALK_OUT:
            if activity == ActivityType.TURNING:
                # Reached marker, turning
                phase_result = self._end_phase(timestamp)
                self._start_phase(TUGPhase.TURN_AROUND, timestamp)
        
        elif current_phase == TUGPhase.TURN_AROUND:
            if activity == ActivityType.WALKING:
                # Completed turn, walking back
                phase_result = self._end_phase(timestamp)
                self._start_phase(TUGPhase.WALK_BACK, timestamp)
        
        elif current_phase == TUGPhase.WALK_BACK:
            if activity == ActivityType.TURNING:
                # Near chair, turning
                phase_result = self._end_phase(timestamp)
                self._start_phase(TUGPhase.TURN_AND_SIT, timestamp)
        
        elif current_phase == TUGPhase.TURN_AND_SIT:
            if activity == ActivityType.SITTING:
                # Completed test
                phase_result = self._end_phase(timestamp)
                self.current_test["current_phase"] = TUGPhase.COMPLETED
                self.current_test["end_time"] = timestamp
        
        # Detect hesitations (velocity drops during walking)
        if activity == ActivityType.STANDING and current_phase in [TUGPhase.WALK_OUT, TUGPhase.WALK_BACK]:
            self.current_test["hesitations"] += 1
        
        return phase_result
    
    def _start_phase(self, phase: TUGPhase, timestamp: float):
        """Start a new phase."""
        self.current_test["current_phase"] = phase
        self.current_test["phase_start_time"] = timestamp
    
    def _end_phase(self, timestamp: float) -> Dict:
        """End current phase and return result."""
        phase = self.current_test["current_phase"]
        start_time = self.current_test["phase_start_time"]
        duration = timestamp - start_time if start_time else 0
        
        result = {
            "phase": phase.value,
            "duration": round(duration, 2),
            "start_time": start_time,
            "end_time": timestamp
        }
        
        self.current_test["phases"].append(TUGPhaseResult(
            phase=phase,
            duration_seconds=duration,
            start_time=start_time,
            end_time=timestamp
        ))
        
        return result
    
    def complete_test(self) -> TUGTestResult:
        """
        Complete the test and generate results.
        
        Returns:
            Complete TUGTestResult
        """
        if not self.current_test:
            raise ValueError("No active test")
        
        test = self.current_test
        
        # Calculate total time
        total_time = test.get("end_time", time.time()) - test["start_time"]
        
        # Determine mobility level
        if total_time < self.NORMAL_THRESHOLD:
            mobility = MobilityLevel.NORMAL
            interpretation = "Normal mobility. Low fall risk."
        elif total_time < self.MILD_THRESHOLD:
            mobility = MobilityLevel.MILD_IMPAIRMENT
            interpretation = "Mild mobility impairment. Consider balance exercises."
        elif total_time < self.MODERATE_THRESHOLD:
            mobility = MobilityLevel.MODERATE_IMPAIRMENT
            interpretation = "Moderate mobility impairment. Fall prevention measures recommended."
        else:
            mobility = MobilityLevel.SEVERE_IMPAIRMENT
            interpretation = "Severe mobility impairment. Immediate intervention needed."
        
        # Calculate fall risk score
        fall_risk = self._calculate_fall_risk(total_time, test)
        
        # Get phase times
        phase_times = self._get_phase_times(test["phases"])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(mobility, total_time, test)
        
        result = TUGTestResult(
            test_id=test["test_id"],
            user_id=test["user_id"],
            total_time_seconds=total_time,
            mobility_level=mobility,
            fall_risk_score=fall_risk,
            phases=test["phases"],
            sit_to_stand_time=phase_times.get(TUGPhase.SIT_TO_STAND, 0),
            walk_out_time=phase_times.get(TUGPhase.WALK_OUT, 0),
            turn_time=phase_times.get(TUGPhase.TURN_AROUND, 0),
            walk_back_time=phase_times.get(TUGPhase.WALK_BACK, 0),
            turn_and_sit_time=phase_times.get(TUGPhase.TURN_AND_SIT, 0),
            hesitations_detected=test["hesitations"],
            interpretation=interpretation,
            recommendations=recommendations,
            start_time=test["start_time"],
            end_time=test.get("end_time", time.time())
        )
        
        # Clean up
        self.is_running = False
        self.current_test = None
        
        return result
    
    def _calculate_fall_risk(self, total_time: float, test: Dict) -> float:
        """Calculate fall risk score (0-100)."""
        risk = 0.0
        
        # Base risk from time
        if total_time >= 30:
            risk = 80
        elif total_time >= 20:
            risk = 50 + (total_time - 20) * 3
        elif total_time >= 14:
            risk = 25 + (total_time - 14) * 4
        elif total_time >= 10:
            risk = 10 + (total_time - 10) * 4
        else:
            risk = total_time
        
        # Add for hesitations
        risk += test["hesitations"] * 5
        
        # Add for stumbles
        risk += test["stumbles"] * 10
        
        return min(100, risk)
    
    def _get_phase_times(self, phases: List[TUGPhaseResult]) -> Dict[TUGPhase, float]:
        """Extract times for each phase."""
        times = {}
        for phase in phases:
            times[phase.phase] = phase.duration_seconds
        return times
    
    def _generate_recommendations(
        self, 
        mobility: MobilityLevel, 
        total_time: float,
        test: Dict
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if mobility == MobilityLevel.SEVERE_IMPAIRMENT:
            recommendations.extend([
                "Assistive device strongly recommended",
                "Consider physical therapy evaluation",
                "Home safety modifications needed",
                "Avoid walking unassisted"
            ])
        elif mobility == MobilityLevel.MODERATE_IMPAIRMENT:
            recommendations.extend([
                "Consider using a cane or walker",
                "Balance training exercises recommended",
                "Remove home fall hazards",
                "Regular exercise program"
            ])
        elif mobility == MobilityLevel.MILD_IMPAIRMENT:
            recommendations.extend([
                "Daily balance exercises",
                "Strengthen legs with chair stands",
                "Practice heel-to-toe walking"
            ])
        else:
            recommendations.extend([
                "Maintain current activity level",
                "Continue regular exercise",
                "Annual reassessment recommended"
            ])
        
        # Phase-specific recommendations
        phases = test.get("phases", [])
        for phase in phases:
            if phase.phase == TUGPhase.SIT_TO_STAND and phase.duration_seconds > 2.5:
                recommendations.append("Practice sit-to-stand exercises")
            elif phase.phase in [TUGPhase.TURN_AROUND, TUGPhase.TURN_AND_SIT]:
                if phase.duration_seconds > 3:
                    recommendations.append("Work on turning stability")
        
        if test["hesitations"] > 0:
            recommendations.append("Address walking confidence with supervised practice")
        
        return recommendations
    
    def cancel_test(self):
        """Cancel current test."""
        self.is_running = False
        self.current_test = None
        self.activity_classifier.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_handler: Optional[TUGTestHandler] = None

def get_tug_handler() -> TUGTestHandler:
    """Get or create TUG handler singleton."""
    global _handler
    if _handler is None:
        _handler = TUGTestHandler()
    return _handler
