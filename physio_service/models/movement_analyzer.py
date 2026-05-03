"""
SMARTCARE+ Physio Service - Movement Analyzer

Owner: Neelaka
Three-layer movement analysis for enhanced exercise monitoring:
  1. Angle-based analysis (current approach, improved)
  2. Velocity & smoothness analysis
  3. Temporal pattern analysis (DTW-based)

This module provides advanced metrics similar to the HybridFallDetector approach.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque
from enum import Enum
import math
import time
import logging
import sys


def _setup_logger(name: str) -> logging.Logger:
    """Configure logger with console output."""
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


logger = _setup_logger("smartcare.physio.movement")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class BasePosture(Enum):
    """Base body postures for exercise validation."""
    STANDING = "standing"
    SITTING = "sitting"
    LYING_SUPINE = "lying_supine"    # On back
    LYING_PRONE = "lying_prone"      # On stomach
    QUADRUPED = "quadruped"          # On all fours (cat-cow position)
    KNEELING = "kneeling"
    UNKNOWN = "unknown"


class MovementPhase(Enum):
    """Exercise movement phases."""
    NEUTRAL = "ready"
    ASCENDING = "ascending"     # Moving toward peak (e.g., standing up, raising arm)
    PEAK = "peak"              # At top of movement
    DESCENDING = "descending"  # Moving toward start (e.g., sitting, lowering arm)
    HOLD = "hold"              # Static hold position
    TRANSITION = "transition"  # Between phases


class ExercisePhase(Enum):
    """Phases within an exercise rep for reference pose matching."""
    START = "start"          # Starting position (e.g., sitting for chair stand)
    MIDDLE = "middle"        # Mid-movement (transition)
    END = "end"              # End position (e.g., standing for chair stand)
    HOLD = "hold"            # Hold position (for static exercises)


@dataclass
class ReferencePose:
    """
    Reference pose defining ideal joint angles for an exercise phase.
    Used to guide patients to correct form without needing video reference.
    """
    exercise: str
    phase: ExercisePhase
    
    # Primary joint angles (degrees) - the key angles to monitor
    primary_angles: Dict[str, float] = field(default_factory=dict)
    
    # Acceptable tolerance for each angle (degrees)
    tolerances: Dict[str, float] = field(default_factory=dict)
    
    # Body alignment requirements
    alignment: Dict[str, str] = field(default_factory=dict)  # e.g., {"back": "straight", "knees": "over_toes"}
    
    # Visual cues for patient
    visual_cues: List[str] = field(default_factory=list)
    
    # Common mistakes to avoid
    common_mistakes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exercise": self.exercise,
            "phase": self.phase.value,
            "primary_angles": self.primary_angles,
            "tolerances": self.tolerances,
            "alignment": self.alignment,
            "visual_cues": self.visual_cues,
            "common_mistakes": self.common_mistakes,
        }


@dataclass
class FormDeviation:
    """
    Deviation between user's current pose and reference pose.
    Provides specific guidance on how to correct form.
    """
    joint: str
    current_angle: float
    target_angle: float
    deviation: float           # Difference in degrees
    tolerance: float           # Acceptable range
    is_within_tolerance: bool
    correction_direction: str  # "more", "less", or "correct"
    correction_hint: str       # Human-readable instruction
    severity: str              # "good", "minor", "major"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint": self.joint,
            "current_angle": round(self.current_angle, 1),
            "target_angle": round(self.target_angle, 1),
            "deviation": round(self.deviation, 1),
            "tolerance": round(self.tolerance, 1),
            "is_within_tolerance": self.is_within_tolerance,
            "correction_direction": self.correction_direction,
            "correction_hint": self.correction_hint,
            "severity": self.severity,
        }


@dataclass
class ReferenceMatchResult:
    """Result of comparing user pose to reference pose."""
    current_phase: ExercisePhase
    reference_pose: Optional[ReferencePose]
    overall_match_score: float          # 0-100%
    deviations: List[FormDeviation] = field(default_factory=list)
    priority_correction: str = ""       # Most important fix
    visual_feedback: List[str] = field(default_factory=list)
    is_form_acceptable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.current_phase.value,
            "match_score": round(self.overall_match_score, 1),
            "deviations": [d.to_dict() for d in self.deviations],
            "priority_correction": self.priority_correction,
            "visual_feedback": self.visual_feedback,
            "is_form_acceptable": self.is_form_acceptable,
        }


@dataclass
class PostureValidation:
    """Result of posture validation for exercise matching."""
    detected_posture: BasePosture = BasePosture.UNKNOWN
    expected_postures: List[str] = field(default_factory=list)
    is_valid: bool = False
    confidence: float = 0.0
    mismatch_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_posture": self.detected_posture.value,
            "expected_postures": self.expected_postures,
            "is_valid": self.is_valid,
            "confidence": round(self.confidence, 3),
            "mismatch_reason": self.mismatch_reason,
        }


@dataclass
class VelocityMetrics:
    """Velocity and smoothness metrics for movement analysis."""
    # Velocity (degrees/second)
    current_velocity: float = 0.0
    peak_velocity: float = 0.0
    avg_velocity: float = 0.0
    
    # Smoothness metrics
    jerk: float = 0.0           # Rate of acceleration change (lower = smoother)
    smoothness_score: float = 100.0  # 0-100 scale
    
    # Tempo consistency
    rep_duration: float = 0.0   # Duration of current/last rep
    avg_rep_duration: float = 0.0
    tempo_variance: float = 0.0  # Consistency of rep timing
    tempo_score: float = 100.0   # 0-100 scale
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_velocity": round(self.current_velocity, 2),
            "peak_velocity": round(self.peak_velocity, 2),
            "avg_velocity": round(self.avg_velocity, 2),
            "jerk": round(self.jerk, 3),
            "smoothness_score": round(self.smoothness_score, 1),
            "rep_duration": round(self.rep_duration, 2),
            "avg_rep_duration": round(self.avg_rep_duration, 2),
            "tempo_variance": round(self.tempo_variance, 2),
            "tempo_score": round(self.tempo_score, 1),
        }


@dataclass
class PainIndicators:
    """Enhanced pain and discomfort detection."""
    # Detection flags
    shaking_detected: bool = False
    slowing_detected: bool = False
    asymmetry_detected: bool = False
    hesitation_detected: bool = False
    rom_reduction_detected: bool = False  # Range of motion reduction
    
    # Confidence scores (0-1)
    shaking_confidence: float = 0.0
    slowing_confidence: float = 0.0
    asymmetry_confidence: float = 0.0
    hesitation_confidence: float = 0.0
    rom_reduction_confidence: float = 0.0
    
    # Overall
    overall_confidence: float = 0.0
    details: List[str] = field(default_factory=list)
    recommendation: str = "continue"  # continue, reduce_intensity, take_break, stop
    
    # Specifics
    affected_joints: List[str] = field(default_factory=list)
    rom_current: float = 0.0  # Current range of motion (degrees)
    rom_baseline: float = 0.0  # Baseline ROM from start of session
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "shaking": {"detected": self.shaking_detected, "confidence": round(self.shaking_confidence, 3)},
            "slowing": {"detected": self.slowing_detected, "confidence": round(self.slowing_confidence, 3)},
            "asymmetry": {"detected": self.asymmetry_detected, "confidence": round(self.asymmetry_confidence, 3)},
            "hesitation": {"detected": self.hesitation_detected, "confidence": round(self.hesitation_confidence, 3)},
            "rom_reduction": {
                "detected": self.rom_reduction_detected, 
                "confidence": round(self.rom_reduction_confidence, 3),
                "current": round(self.rom_current, 1),
                "baseline": round(self.rom_baseline, 1),
            },
            "overall_confidence": round(self.overall_confidence, 3),
            "details": self.details,
            "recommendation": self.recommendation,
            "affected_joints": self.affected_joints,
        }


@dataclass
class RepDetectionResult:
    """Result from 3-layer rep detection."""
    rep_completed: bool = False
    confidence: float = 0.0
    current_phase: MovementPhase = MovementPhase.NEUTRAL
    
    # Layer scores (0-1)
    angle_score: float = 0.0     # Layer 1: Angle-based
    velocity_score: float = 0.0  # Layer 2: Velocity-based
    pattern_score: float = 0.0   # Layer 3: Pattern-based
    
    # Timing
    phase_duration: float = 0.0
    rep_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rep_completed": self.rep_completed,
            "confidence": round(self.confidence, 3),
            "phase": self.current_phase.value,
            "scores": {
                "angle": round(self.angle_score, 3),
                "velocity": round(self.velocity_score, 3),
                "pattern": round(self.pattern_score, 3),
            },
            "phase_duration": round(self.phase_duration, 2),
            "rep_duration": round(self.rep_duration, 2),
        }


@dataclass
class MovementAnalysisResult:
    """Complete movement analysis output."""
    # Posture validation (Layer 0 - must pass before rep counting)
    posture_validation: PostureValidation = field(default_factory=PostureValidation)
    
    # Rep detection
    rep_result: RepDetectionResult = field(default_factory=RepDetectionResult)
    
    # Velocity/smoothness
    velocity_metrics: VelocityMetrics = field(default_factory=VelocityMetrics)
    
    # Pain detection
    pain_indicators: PainIndicators = field(default_factory=PainIndicators)
    
    # Form guidance (Reference skeleton matching)
    form_guidance: Dict[str, Any] = field(default_factory=dict)
    
    # Adaptive feedback
    feedback_priority: List[str] = field(default_factory=list)  # Prioritized feedback
    current_focus: str = ""  # Single most important thing to focus on
    adaptation_suggestion: str = ""  # Suggested modification
    
    # Timing
    processing_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "posture_validation": self.posture_validation.to_dict(),
            "rep": self.rep_result.to_dict(),
            "velocity": self.velocity_metrics.to_dict(),
            "pain": self.pain_indicators.to_dict(),
            "form_guidance": self.form_guidance,
            "feedback": {
                "priority_list": self.feedback_priority[:3],  # Top 3 only
                "current_focus": self.current_focus,
                "adaptation": self.adaptation_suggestion,
            },
            "processing_time_ms": round(self.processing_time_ms, 1),
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MOVEMENT ANALYZER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MovementAnalyzer:
    """
    Three-layer movement analyzer for enhanced exercise monitoring.
    
    Layer 1: Angle-based analysis (thresholds)
    Layer 2: Velocity & smoothness analysis
    Layer 3: Temporal pattern matching
    """
    
    # History buffer sizes
    ANGLE_HISTORY_SIZE = 60     # ~2 seconds at 30fps
    VELOCITY_HISTORY_SIZE = 30  # ~1 second
    REP_TIMING_HISTORY = 10     # Last 10 reps for tempo analysis
    
    # Smoothness thresholds
    JERK_THRESHOLD_GOOD = 500     # deg/s³ - below this is smooth
    JERK_THRESHOLD_POOR = 2000    # deg/s³ - above this is jerky
    
    # Pain detection thresholds
    SHAKING_VARIANCE_THRESHOLD = 15.0    # degrees² variance
    SLOWING_RATIO_THRESHOLD = 0.6        # 60% of initial velocity = slowing
    ASYMMETRY_THRESHOLD = 20.0           # 20° difference between sides
    HESITATION_PAUSE_THRESHOLD = 0.8     # 0.8 seconds of minimal movement
    ROM_REDUCTION_THRESHOLD = 0.15       # 15% reduction from baseline
    
    # Velocity thresholds (degrees/second)
    MIN_MOVEMENT_VELOCITY = 5.0    # Below this = static/hesitating
    NORMAL_VELOCITY_RANGE = (20, 100)  # Normal exercise range
    
    # Expected postures for each exercise type
    EXPECTED_POSTURES: Dict[str, List[BasePosture]] = {
        "chair_stand": [BasePosture.SITTING, BasePosture.STANDING],
        "squat": [BasePosture.STANDING],
        "leg_raise": [BasePosture.STANDING, BasePosture.LYING_SUPINE],
        "arm_raise": [BasePosture.STANDING, BasePosture.SITTING],
        "seated_arm_raises": [BasePosture.SITTING],
        "wall_pushup": [BasePosture.STANDING],
        "marching": [BasePosture.STANDING],
        "single_leg_stand": [BasePosture.STANDING],
        "seated_leg_raise": [BasePosture.SITTING],
        "shoulder_rolls": [BasePosture.STANDING, BasePosture.SITTING],
        "hip_flexion": [BasePosture.STANDING],
        "knee_extension": [BasePosture.SITTING],
        "ankle_rotation": [BasePosture.SITTING],
        "balance_exercise": [BasePosture.STANDING],
        "generic": [BasePosture.STANDING, BasePosture.SITTING],  # Allow any upright posture
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REFERENCE POSES - Ideal joint angles for each exercise phase
    # These enable form guidance without needing reference videos
    # ═══════════════════════════════════════════════════════════════════════════
    
    REFERENCE_POSES: Dict[str, Dict[str, ReferencePose]] = {
        # ─────────────────────────────────────────────────────────────────────
        # CHAIR STAND: Sitting → Standing → Sitting
        # ─────────────────────────────────────────────────────────────────────
        "chair_stand": {
            "start": ReferencePose(
                exercise="chair_stand",
                phase=ExercisePhase.START,
                primary_angles={
                    "knee": 90.0,        # Knees at 90° when sitting
                    "hip": 90.0,         # Hips at 90° when sitting
                    "back": 90.0,        # Torso upright
                },
                tolerances={"knee": 15.0, "hip": 15.0, "back": 15.0},
                alignment={"back": "upright", "feet": "flat_on_floor", "arms": "crossed_on_chest"},
                visual_cues=[
                    "Sit at edge of chair",
                    "Feet flat, shoulder-width apart",
                    "Cross arms over chest",
                    "Keep back straight",
                ],
                common_mistakes=[
                    "Leaning too far forward",
                    "Feet too close together",
                    "Using hands to push up",
                ],
            ),
            "end": ReferencePose(
                exercise="chair_stand",
                phase=ExercisePhase.END,
                primary_angles={
                    "knee": 175.0,       # Knees nearly straight
                    "hip": 175.0,        # Hips extended
                    "back": 180.0,       # Torso vertical
                },
                tolerances={"knee": 10.0, "hip": 10.0, "back": 10.0},
                alignment={"back": "straight", "hips": "fully_extended", "head": "looking_forward"},
                visual_cues=[
                    "Stand fully upright",
                    "Hips pushed forward",
                    "Knees straight (not locked)",
                    "Look straight ahead",
                ],
                common_mistakes=[
                    "Not standing fully upright",
                    "Knees still bent",
                    "Leaning forward",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # SQUAT: Standing → Squat → Standing
        # ─────────────────────────────────────────────────────────────────────
        "squat": {
            "start": ReferencePose(
                exercise="squat",
                phase=ExercisePhase.START,
                primary_angles={
                    "knee": 175.0,
                    "hip": 175.0,
                    "back": 180.0,
                },
                tolerances={"knee": 10.0, "hip": 10.0, "back": 10.0},
                alignment={"feet": "shoulder_width", "toes": "slightly_out", "back": "neutral"},
                visual_cues=[
                    "Stand with feet shoulder-width apart",
                    "Toes pointing slightly outward",
                    "Arms extended forward or crossed",
                ],
                common_mistakes=[
                    "Feet too narrow",
                    "Knees caving inward",
                ],
            ),
            "middle": ReferencePose(
                exercise="squat",
                phase=ExercisePhase.MIDDLE,
                primary_angles={
                    "knee": 90.0,        # Deep squat
                    "hip": 90.0,
                    "back": 160.0,       # Slight forward lean is OK
                },
                tolerances={"knee": 15.0, "hip": 15.0, "back": 20.0},
                alignment={"knees": "over_toes", "back": "neutral", "weight": "on_heels"},
                visual_cues=[
                    "Lower until thighs parallel to ground",
                    "Keep knees over toes",
                    "Weight on heels",
                    "Chest up, back neutral",
                ],
                common_mistakes=[
                    "Knees going past toes",
                    "Rounding lower back",
                    "Heels lifting off ground",
                    "Knees caving inward",
                ],
            ),
            "end": ReferencePose(
                exercise="squat",
                phase=ExercisePhase.END,
                primary_angles={
                    "knee": 175.0,
                    "hip": 175.0,
                    "back": 180.0,
                },
                tolerances={"knee": 10.0, "hip": 10.0, "back": 10.0},
                alignment={"hips": "fully_extended", "glutes": "squeezed"},
                visual_cues=[
                    "Stand fully upright",
                    "Squeeze glutes at top",
                ],
                common_mistakes=[
                    "Not fully extending hips",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # ARM RAISE: Arms down → Arms up → Arms down
        # ─────────────────────────────────────────────────────────────────────
        "arm_raise": {
            "start": ReferencePose(
                exercise="arm_raise",
                phase=ExercisePhase.START,
                primary_angles={
                    "shoulder": 10.0,    # Arms at sides
                    "elbow": 175.0,      # Arms straight
                },
                tolerances={"shoulder": 15.0, "elbow": 10.0},
                alignment={"arms": "at_sides", "palms": "facing_body"},
                visual_cues=[
                    "Stand tall with arms at sides",
                    "Palms facing your body",
                    "Shoulders relaxed",
                ],
                common_mistakes=[
                    "Shoulders raised/tensed",
                    "Arms bent at elbows",
                ],
            ),
            "end": ReferencePose(
                exercise="arm_raise",
                phase=ExercisePhase.END,
                primary_angles={
                    "shoulder": 170.0,   # Arms overhead
                    "elbow": 175.0,      # Arms straight
                },
                tolerances={"shoulder": 15.0, "elbow": 10.0},
                alignment={"arms": "overhead", "biceps": "by_ears"},
                visual_cues=[
                    "Raise arms straight overhead",
                    "Biceps by your ears",
                    "Keep arms straight",
                    "Palms facing each other",
                ],
                common_mistakes=[
                    "Arms not fully raised",
                    "Elbows bent",
                    "Arching lower back",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # SEATED ARM RAISES
        # ─────────────────────────────────────────────────────────────────────
        "seated_arm_raises": {
            "start": ReferencePose(
                exercise="seated_arm_raises",
                phase=ExercisePhase.START,
                primary_angles={
                    "shoulder": 10.0,
                    "elbow": 175.0,
                    "hip": 90.0,
                },
                tolerances={"shoulder": 15.0, "elbow": 10.0, "hip": 10.0},
                alignment={"back": "against_chair", "feet": "flat"},
                visual_cues=[
                    "Sit upright with back against chair",
                    "Arms at sides",
                    "Feet flat on floor",
                ],
                common_mistakes=[
                    "Slouching",
                    "Leaning forward",
                ],
            ),
            "end": ReferencePose(
                exercise="seated_arm_raises",
                phase=ExercisePhase.END,
                primary_angles={
                    "shoulder": 170.0,
                    "elbow": 175.0,
                    "hip": 90.0,
                },
                tolerances={"shoulder": 15.0, "elbow": 10.0, "hip": 10.0},
                alignment={"arms": "overhead", "back": "straight"},
                visual_cues=[
                    "Raise arms straight overhead",
                    "Keep back against chair",
                    "Don't arch lower back",
                ],
                common_mistakes=[
                    "Leaning back",
                    "Arching lower back",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # LEG RAISE (Standing)
        # ─────────────────────────────────────────────────────────────────────
        "leg_raise": {
            "start": ReferencePose(
                exercise="leg_raise",
                phase=ExercisePhase.START,
                primary_angles={
                    "hip": 175.0,        # Standing straight
                    "knee": 175.0,       # Leg straight
                },
                tolerances={"hip": 10.0, "knee": 10.0},
                alignment={"standing_leg": "straight", "hands": "on_support"},
                visual_cues=[
                    "Stand next to chair/wall for support",
                    "Both legs straight",
                    "Hold support lightly",
                ],
                common_mistakes=[
                    "Leaning on support too heavily",
                ],
            ),
            "end": ReferencePose(
                exercise="leg_raise",
                phase=ExercisePhase.END,
                primary_angles={
                    "hip": 90.0,         # Leg raised to 90°
                    "knee": 175.0,       # Keep leg straight
                },
                tolerances={"hip": 20.0, "knee": 15.0},
                alignment={"raised_leg": "straight", "standing_leg": "stable"},
                visual_cues=[
                    "Raise leg forward to hip height",
                    "Keep raised leg straight",
                    "Don't lean back",
                    "Keep standing leg stable",
                ],
                common_mistakes=[
                    "Bending raised leg",
                    "Leaning back excessively",
                    "Swinging leg (use control)",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # SEATED LEG RAISE
        # ─────────────────────────────────────────────────────────────────────
        "seated_leg_raise": {
            "start": ReferencePose(
                exercise="seated_leg_raise",
                phase=ExercisePhase.START,
                primary_angles={
                    "knee": 90.0,        # Sitting with feet on floor
                    "hip": 90.0,
                },
                tolerances={"knee": 15.0, "hip": 10.0},
                alignment={"back": "against_chair", "feet": "on_floor"},
                visual_cues=[
                    "Sit with back against chair",
                    "Feet flat on floor",
                    "Grip sides of chair lightly",
                ],
                common_mistakes=[
                    "Slouching forward",
                ],
            ),
            "end": ReferencePose(
                exercise="seated_leg_raise",
                phase=ExercisePhase.END,
                primary_angles={
                    "knee": 175.0,       # Leg extended straight
                    "hip": 90.0,         # Still sitting
                },
                tolerances={"knee": 10.0, "hip": 10.0},
                alignment={"raised_leg": "parallel_to_floor", "back": "straight"},
                visual_cues=[
                    "Extend one leg straight out",
                    "Leg parallel to floor",
                    "Hold for 1-2 seconds",
                    "Keep back against chair",
                ],
                common_mistakes=[
                    "Not fully extending leg",
                    "Leaning back",
                    "Raising leg too high",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # WALL PUSHUP
        # ─────────────────────────────────────────────────────────────────────
        "wall_pushup": {
            "start": ReferencePose(
                exercise="wall_pushup",
                phase=ExercisePhase.START,
                primary_angles={
                    "elbow": 175.0,      # Arms extended
                    "shoulder": 90.0,    # Arms at shoulder height
                },
                tolerances={"elbow": 10.0, "shoulder": 15.0},
                alignment={"body": "straight_line", "hands": "shoulder_width"},
                visual_cues=[
                    "Stand arm's length from wall",
                    "Hands on wall at shoulder height",
                    "Feet shoulder-width apart",
                    "Body in straight line",
                ],
                common_mistakes=[
                    "Standing too close to wall",
                    "Hands too high or low",
                ],
            ),
            "middle": ReferencePose(
                exercise="wall_pushup",
                phase=ExercisePhase.MIDDLE,
                primary_angles={
                    "elbow": 90.0,       # Bent at 90°
                    "shoulder": 60.0,    # Shoulders engaged
                },
                tolerances={"elbow": 15.0, "shoulder": 20.0},
                alignment={"chest": "toward_wall", "body": "straight"},
                visual_cues=[
                    "Bend elbows to bring chest toward wall",
                    "Keep body straight (don't sag)",
                    "Elbows at 45° angle from body",
                ],
                common_mistakes=[
                    "Sagging at hips",
                    "Flaring elbows out wide",
                    "Not going deep enough",
                ],
            ),
            "end": ReferencePose(
                exercise="wall_pushup",
                phase=ExercisePhase.END,
                primary_angles={
                    "elbow": 175.0,
                    "shoulder": 90.0,
                },
                tolerances={"elbow": 10.0, "shoulder": 15.0},
                alignment={"arms": "fully_extended"},
                visual_cues=[
                    "Push back to starting position",
                    "Fully extend arms",
                ],
                common_mistakes=[
                    "Not fully extending arms",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # MARCHING IN PLACE
        # ─────────────────────────────────────────────────────────────────────
        "marching": {
            "start": ReferencePose(
                exercise="marching",
                phase=ExercisePhase.START,
                primary_angles={
                    "knee": 175.0,       # Standing straight
                    "hip": 175.0,
                },
                tolerances={"knee": 10.0, "hip": 10.0},
                alignment={"posture": "upright", "arms": "at_sides"},
                visual_cues=[
                    "Stand tall",
                    "Arms at sides or bent at elbows",
                    "Feet together",
                ],
                common_mistakes=[],
            ),
            "end": ReferencePose(
                exercise="marching",
                phase=ExercisePhase.END,
                primary_angles={
                    "knee": 90.0,        # Knee raised to 90°
                    "hip": 90.0,         # Hip flexed to 90°
                },
                tolerances={"knee": 15.0, "hip": 15.0},
                alignment={"raised_knee": "toward_chest", "standing_leg": "straight"},
                visual_cues=[
                    "Lift knee toward chest",
                    "Thigh parallel to floor",
                    "Keep standing leg straight",
                    "Swing opposite arm forward",
                ],
                common_mistakes=[
                    "Not lifting knee high enough",
                    "Leaning back",
                    "Losing balance",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # SINGLE LEG STAND
        # ─────────────────────────────────────────────────────────────────────
        "single_leg_stand": {
            "start": ReferencePose(
                exercise="single_leg_stand",
                phase=ExercisePhase.START,
                primary_angles={
                    "knee": 175.0,
                    "hip": 175.0,
                },
                tolerances={"knee": 10.0, "hip": 10.0},
                alignment={"both_feet": "on_floor"},
                visual_cues=[
                    "Stand near support (chair/wall)",
                    "Both feet on floor",
                    "Stand tall",
                ],
                common_mistakes=[],
            ),
            "hold": ReferencePose(
                exercise="single_leg_stand",
                phase=ExercisePhase.HOLD,
                primary_angles={
                    "standing_knee": 175.0,
                    "raised_knee": 90.0,     # Slight bend
                    "hip": 90.0,             # Slight hip flexion
                },
                tolerances={"standing_knee": 10.0, "raised_knee": 30.0, "hip": 20.0},
                alignment={"hips": "level", "standing_leg": "straight"},
                visual_cues=[
                    "Lift one foot off floor",
                    "Keep hips level",
                    "Hold for 10-30 seconds",
                    "Use support if needed",
                ],
                common_mistakes=[
                    "Dropping hip on raised side",
                    "Bending standing knee",
                    "Looking down (look forward)",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # SHOULDER ROLLS
        # ─────────────────────────────────────────────────────────────────────
        "shoulder_rolls": {
            "start": ReferencePose(
                exercise="shoulder_rolls",
                phase=ExercisePhase.START,
                primary_angles={
                    "shoulder": 10.0,    # Arms relaxed at sides
                },
                tolerances={"shoulder": 15.0},
                alignment={"shoulders": "relaxed", "arms": "at_sides"},
                visual_cues=[
                    "Stand or sit with good posture",
                    "Arms relaxed at sides",
                    "Shoulders down and back",
                ],
                common_mistakes=[
                    "Tensed shoulders",
                ],
            ),
            "middle": ReferencePose(
                exercise="shoulder_rolls",
                phase=ExercisePhase.MIDDLE,
                primary_angles={
                    "shoulder": 30.0,    # Shoulders elevated during roll
                },
                tolerances={"shoulder": 20.0},
                alignment={"shoulders": "elevated_and_rotating"},
                visual_cues=[
                    "Roll shoulders up, back, and down",
                    "Make smooth circular motion",
                    "Keep arms relaxed",
                ],
                common_mistakes=[
                    "Jerky movements",
                    "Moving arms instead of shoulders",
                ],
            ),
        },
        
        # ─────────────────────────────────────────────────────────────────────
        # GENERIC (fallback)
        # ─────────────────────────────────────────────────────────────────────
        "generic": {
            "start": ReferencePose(
                exercise="generic",
                phase=ExercisePhase.START,
                primary_angles={},
                tolerances={},
                alignment={"posture": "upright"},
                visual_cues=["Maintain good posture"],
                common_mistakes=[],
            ),
        },
    }
    
    def __init__(self, exercise_type: str = "generic"):
        """
        Initialize movement analyzer.
        
        Args:
            exercise_type: Type of exercise being monitored
        """
        self.exercise_type = exercise_type
        
        # Posture tracking
        self.posture_history: Deque[BasePosture] = deque(maxlen=30)  # Last ~1 second
        self.posture_valid_frames = 0
        self.posture_invalid_frames = 0
        
        # Angle history: {joint_name: deque of (timestamp, angle)}
        self.angle_history: Dict[str, Deque[Tuple[float, float]]] = {}
        
        # Velocity history (derived from angles)
        self.velocity_history: Deque[Tuple[float, float]] = deque(maxlen=self.VELOCITY_HISTORY_SIZE)
        self.acceleration_history: Deque[Tuple[float, float]] = deque(maxlen=self.VELOCITY_HISTORY_SIZE)
        
        # Rep timing history
        self.rep_durations: Deque[float] = deque(maxlen=self.REP_TIMING_HISTORY)
        self.rep_start_time: float = 0.0
        self.phase_start_time: float = 0.0
        
        # Movement phase tracking
        self.current_phase = MovementPhase.NEUTRAL
        self.phase_angles: List[float] = []  # Angles during current phase
        
        # ROM baseline tracking
        self.rom_baseline: Dict[str, float] = {}  # joint -> max ROM observed
        self.rom_current: Dict[str, float] = {}   # joint -> current rep ROM
        self.baseline_established = False
        self.baseline_reps = 3  # First 3 reps establish baseline
        self.reps_for_baseline = 0
        
        # Initial velocities for slowing detection
        self.initial_velocities: Deque[float] = deque(maxlen=5)
        self.velocity_baseline = 0.0
        
        # Last analysis time
        self.last_analysis_time = 0.0
        self.last_primary_angle = 0.0
        
        logger.info(f"MovementAnalyzer initialized for exercise: {exercise_type}")
    
    def reset(self):
        """Reset analyzer state for new session."""
        self.angle_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self.rep_durations.clear()
        self.rep_start_time = 0.0
        self.phase_start_time = 0.0
        self.current_phase = MovementPhase.NEUTRAL
        self.phase_angles.clear()
        self.rom_baseline.clear()
        self.rom_current.clear()
        self.baseline_established = False
        self.reps_for_baseline = 0
        self.initial_velocities.clear()
        self.velocity_baseline = 0.0
        self.last_analysis_time = 0.0
        self.last_primary_angle = 0.0
        self.posture_history.clear()
        self.posture_valid_frames = 0
        self.posture_invalid_frames = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # POSTURE DETECTION & VALIDATION (Layer 0)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def detect_posture(self, landmarks: List[Any]) -> BasePosture:
        """
        Detect the base posture from landmarks.
        
        Args:
            landmarks: List of pose landmarks with x, y, z, visibility
        
        Returns:
            Detected BasePosture
        """
        if not landmarks or len(landmarks) < 25:
            return BasePosture.UNKNOWN
        
        try:
            # Extract key landmark Y positions (normalized 0-1, top to bottom)
            # MediaPipe landmark indices
            NOSE = 0
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_KNEE = 25
            RIGHT_KNEE = 26
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            LEFT_WRIST = 15
            RIGHT_WRIST = 16
            
            nose_y = landmarks[NOSE].y
            avg_shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
            avg_hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
            avg_knee_y = (landmarks[LEFT_KNEE].y + landmarks[RIGHT_KNEE].y) / 2
            avg_ankle_y = (landmarks[LEFT_ANKLE].y + landmarks[RIGHT_ANKLE].y) / 2
            avg_wrist_y = (landmarks[LEFT_WRIST].y + landmarks[RIGHT_WRIST].y) / 2
            
            # Calculate relative positions
            head_to_hip = avg_hip_y - nose_y  # Positive = head above hips (normal)
            hip_to_knee = avg_knee_y - avg_hip_y
            knee_to_ankle = avg_ankle_y - avg_knee_y
            
            # QUADRUPED detection: Head and hips at similar height, wrists near ground
            # Person on all fours - hips, shoulders, wrists at similar Y
            if (abs(avg_shoulder_y - avg_hip_y) < 0.15 and 
                abs(avg_wrist_y - avg_hip_y) < 0.2 and
                head_to_hip < 0.15):
                return BasePosture.QUADRUPED
            
            # LYING detection: Body mostly horizontal
            if abs(nose_y - avg_ankle_y) < 0.15:
                # Check if face up or face down
                if avg_shoulder_y > avg_hip_y:
                    return BasePosture.LYING_SUPINE
                else:
                    return BasePosture.LYING_PRONE
            
            # STANDING: Vertical posture, hips above knees, knees above ankles
            if (head_to_hip > 0.1 and hip_to_knee > 0.05 and knee_to_ankle > 0.05):
                return BasePosture.STANDING
            
            # SITTING: Hips and knees at similar Y level (both "bent")
            if (head_to_hip > 0.1 and abs(hip_to_knee) < 0.1):
                return BasePosture.SITTING
            
            # KNEELING: Knees at ankle level, torso vertical
            if (abs(avg_knee_y - avg_ankle_y) < 0.1 and head_to_hip > 0.15):
                return BasePosture.KNEELING
            
            return BasePosture.UNKNOWN
            
        except Exception as e:
            logger.debug(f"Posture detection error: {e}")
            return BasePosture.UNKNOWN
    
    def validate_posture(self, detected_posture: BasePosture) -> PostureValidation:
        """
        Validate if detected posture matches expected postures for this exercise.
        
        Args:
            detected_posture: The detected base posture
        
        Returns:
            PostureValidation result
        """
        expected = self.EXPECTED_POSTURES.get(self.exercise_type, [BasePosture.STANDING, BasePosture.SITTING])
        expected_names = [p.value for p in expected]
        
        is_valid = detected_posture in expected
        
        # Track posture history for stability
        self.posture_history.append(detected_posture)
        if is_valid:
            self.posture_valid_frames += 1
        else:
            self.posture_invalid_frames += 1
        
        # Calculate confidence based on history
        total_frames = self.posture_valid_frames + self.posture_invalid_frames
        if total_frames > 0:
            valid_ratio = self.posture_valid_frames / total_frames
        else:
            valid_ratio = 0.0
        
        # Determine mismatch reason
        mismatch_reason = ""
        if not is_valid:
            mismatch_reason = (
                f"Detected '{detected_posture.value}' posture, but '{self.exercise_type}' "
                f"requires {expected_names}. Please adjust your position."
            )
            logger.warning(f"Posture mismatch: {mismatch_reason}")
        
        return PostureValidation(
            detected_posture=detected_posture,
            expected_postures=expected_names,
            is_valid=is_valid,
            confidence=valid_ratio,
            mismatch_reason=mismatch_reason,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REFERENCE POSE MATCHING (Form Guidance without Video)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_current_exercise_phase(self, primary_angle: float, threshold_up: float, threshold_down: float) -> ExercisePhase:
        """
        Determine current phase of the exercise based on primary angle.
        
        Args:
            primary_angle: Current primary joint angle
            threshold_up: Upper threshold angle
            threshold_down: Lower threshold angle
        
        Returns:
            Current ExercisePhase
        """
        # Calculate midpoint
        mid_point = (threshold_up + threshold_down) / 2
        
        # Map movement phase to exercise phase
        if self.current_phase == MovementPhase.NEUTRAL:
            if primary_angle < threshold_down + 10:
                return ExercisePhase.START
            elif primary_angle > threshold_up - 10:
                return ExercisePhase.END
            else:
                return ExercisePhase.MIDDLE
        elif self.current_phase == MovementPhase.ASCENDING:
            return ExercisePhase.MIDDLE
        elif self.current_phase == MovementPhase.PEAK:
            return ExercisePhase.END
        elif self.current_phase == MovementPhase.DESCENDING:
            return ExercisePhase.MIDDLE
        else:
            return ExercisePhase.START
    
    def get_reference_pose(self, phase: ExercisePhase) -> Optional[ReferencePose]:
        """
        Get the reference pose for current exercise and phase.
        
        Args:
            phase: Current exercise phase
        
        Returns:
            ReferencePose if available, None otherwise
        """
        exercise_refs = self.REFERENCE_POSES.get(self.exercise_type, {})
        
        # Map phase to reference key
        phase_key_map = {
            ExercisePhase.START: "start",
            ExercisePhase.MIDDLE: "middle",
            ExercisePhase.END: "end",
            ExercisePhase.HOLD: "hold",
        }
        
        phase_key = phase_key_map.get(phase, "start")
        
        # Try exact phase, then fall back to start
        ref_pose = exercise_refs.get(phase_key)
        if ref_pose is None and phase_key != "start":
            ref_pose = exercise_refs.get("start")
        
        return ref_pose
    
    def compare_to_reference(
        self,
        current_angles: Dict[str, float],
        phase: ExercisePhase
    ) -> ReferenceMatchResult:
        """
        Compare user's current pose to the reference pose for this exercise phase.
        
        Args:
            current_angles: Dictionary of current joint angles
            phase: Current exercise phase
        
        Returns:
            ReferenceMatchResult with deviations and guidance
        """
        ref_pose = self.get_reference_pose(phase)
        
        if ref_pose is None:
            # No reference pose available
            return ReferenceMatchResult(
                current_phase=phase,
                reference_pose=None,
                overall_match_score=100.0,  # Assume correct if no reference
                is_form_acceptable=True,
            )
        
        deviations: List[FormDeviation] = []
        total_score = 0.0
        joint_count = 0
        priority_issues: List[Tuple[float, str]] = []  # (severity, hint)
        
        # Map common angle names
        angle_name_map = {
            "avg_knee": "knee",
            "left_knee": "knee",
            "right_knee": "knee",
            "avg_hip": "hip",
            "left_hip": "hip",
            "right_hip": "hip",
            "avg_shoulder": "shoulder",
            "left_shoulder": "shoulder",
            "right_shoulder": "shoulder",
            "avg_elbow": "elbow",
            "left_elbow": "elbow",
            "right_elbow": "elbow",
            "back_angle": "back",
            "trunk_angle": "back",
        }
        
        # Check each primary angle in reference
        for joint, target_angle in ref_pose.primary_angles.items():
            # Find matching current angle
            current_angle = None
            for angle_name, value in current_angles.items():
                mapped_name = angle_name_map.get(angle_name, angle_name)
                if mapped_name == joint or joint in angle_name:
                    current_angle = value
                    break
            
            if current_angle is None:
                continue  # Skip if angle not available
            
            tolerance = ref_pose.tolerances.get(joint, 15.0)
            deviation = abs(current_angle - target_angle)
            is_within = deviation <= tolerance
            
            # Determine correction direction and severity
            if is_within:
                direction = "correct"
                severity = "good"
                hint = f"{joint.title()} angle is good"
            else:
                severity = "minor" if deviation <= tolerance * 2 else "major"
                
                if current_angle < target_angle:
                    direction = "more"
                    hint = self._get_correction_hint(joint, "more", target_angle - current_angle)
                else:
                    direction = "less"
                    hint = self._get_correction_hint(joint, "less", current_angle - target_angle)
                
                priority_issues.append((deviation, hint))
            
            # Calculate score for this joint (100 = perfect, 0 = very wrong)
            if deviation <= tolerance:
                joint_score = 100.0
            else:
                # Score decreases as deviation increases beyond tolerance
                excess = deviation - tolerance
                joint_score = max(0.0, 100.0 - (excess * 2))  # -2 points per degree
            
            total_score += joint_score
            joint_count += 1
            
            deviations.append(FormDeviation(
                joint=joint,
                current_angle=current_angle,
                target_angle=target_angle,
                deviation=deviation,
                tolerance=tolerance,
                is_within_tolerance=is_within,
                correction_direction=direction,
                correction_hint=hint,
                severity=severity,
            ))
        
        # Calculate overall score
        overall_score = total_score / joint_count if joint_count > 0 else 100.0
        
        # Determine if form is acceptable (all joints within tolerance or score > 80)
        is_acceptable = all(d.is_within_tolerance for d in deviations) or overall_score >= 80.0
        
        # Get priority correction (worst deviation)
        priority_correction = ""
        if priority_issues:
            priority_issues.sort(reverse=True)  # Highest deviation first
            priority_correction = priority_issues[0][1]
        
        # Build visual feedback
        visual_feedback = ref_pose.visual_cues.copy() if ref_pose else []
        if not is_acceptable and ref_pose and ref_pose.common_mistakes:
            visual_feedback.append(f"⚠️ Watch out for: {ref_pose.common_mistakes[0]}")
        
        return ReferenceMatchResult(
            current_phase=phase,
            reference_pose=ref_pose,
            overall_match_score=overall_score,
            deviations=deviations,
            priority_correction=priority_correction,
            visual_feedback=visual_feedback,
            is_form_acceptable=is_acceptable,
        )
    
    def _get_correction_hint(self, joint: str, direction: str, amount: float) -> str:
        """Generate human-readable correction hint."""
        hints = {
            ("knee", "more"): f"Straighten your knees more ({amount:.0f}° to go)",
            ("knee", "less"): f"Bend your knees more ({amount:.0f}° too straight)",
            ("hip", "more"): f"Open your hips more ({amount:.0f}° to go)",
            ("hip", "less"): f"Bend at the hips more ({amount:.0f}° too open)",
            ("shoulder", "more"): f"Raise your arms higher ({amount:.0f}° to go)",
            ("shoulder", "less"): f"Lower your arms ({amount:.0f}° too high)",
            ("elbow", "more"): f"Straighten your elbows ({amount:.0f}° to go)",
            ("elbow", "less"): f"Bend your elbows more ({amount:.0f}° too straight)",
            ("back", "more"): f"Straighten your back ({amount:.0f}° to go)",
            ("back", "less"): f"Your back is too arched ({amount:.0f}° adjustment needed)",
        }
        return hints.get((joint, direction), f"Adjust your {joint} by {amount:.0f}°")
    
    def get_form_guidance(
        self,
        current_angles: Dict[str, float],
        primary_angle: float,
        threshold_up: float,
        threshold_down: float
    ) -> Dict[str, Any]:
        """
        Get comprehensive form guidance for the current exercise state.
        
        This is the main method for form guidance - use this to help patients
        exercise correctly without needing reference videos.
        
        Args:
            current_angles: Dictionary of all current joint angles
            primary_angle: The primary angle being tracked
            threshold_up: Upper threshold
            threshold_down: Lower threshold
        
        Returns:
            Dictionary with form guidance including:
            - current_phase: Which phase of the exercise
            - match_score: How well user matches ideal form (0-100)
            - is_acceptable: Whether form is good enough
            - priority_fix: The one thing to focus on
            - all_feedback: List of all corrections needed
            - visual_cues: What to show the user
            - reference_angles: The ideal angles to display
        """
        # Determine current phase
        phase = self.get_current_exercise_phase(primary_angle, threshold_up, threshold_down)
        
        # Compare to reference
        match_result = self.compare_to_reference(current_angles, phase)
        
        # Get reference pose for display
        ref_pose = self.get_reference_pose(phase)
        reference_angles = ref_pose.primary_angles if ref_pose else {}
        
        # Build feedback list
        feedback_list = []
        for dev in match_result.deviations:
            if not dev.is_within_tolerance:
                feedback_list.append({
                    "joint": dev.joint,
                    "hint": dev.correction_hint,
                    "severity": dev.severity,
                    "deviation": dev.deviation,
                })
        
        return {
            "current_phase": phase.value,
            "match_score": round(match_result.overall_match_score, 1),
            "is_acceptable": match_result.is_form_acceptable,
            "priority_fix": match_result.priority_correction,
            "all_feedback": feedback_list,
            "visual_cues": match_result.visual_feedback,
            "reference_angles": reference_angles,
            "deviations": [d.to_dict() for d in match_result.deviations],
        }
    
    def add_angle_sample(self, joint_name: str, angle: float, timestamp: float):
        """Add a joint angle measurement."""
        if joint_name not in self.angle_history:
            self.angle_history[joint_name] = deque(maxlen=self.ANGLE_HISTORY_SIZE)
        self.angle_history[joint_name].append((timestamp, angle))
    
    def analyze(
        self,
        angles: Dict[str, float],
        primary_angle_name: str,
        threshold_up: float,
        threshold_down: float,
        timestamp: float = None,
        landmarks: Optional[List[Any]] = None
    ) -> MovementAnalysisResult:
        """
        Perform complete 4-layer movement analysis.
        
        Layer 0: Posture validation (must pass before rep counting)
        Layer 1: Angle-based analysis
        Layer 2: Velocity & smoothness analysis
        Layer 3: Temporal pattern matching
        
        Args:
            angles: Current joint angles {name: degrees}
            primary_angle_name: Main angle to track for reps (e.g., "avg_knee")
            threshold_up: Angle threshold for "up" phase
            threshold_down: Angle threshold for "down" phase
            timestamp: Current time (uses time.time() if None)
            landmarks: Optional pose landmarks for posture validation
        
        Returns:
            MovementAnalysisResult with all metrics including posture validation
        """
        start_time = time.time()
        if timestamp is None:
            timestamp = start_time
        
        # Create result
        result = MovementAnalysisResult()
        
        # ═══════════════════════════════════════════════════════════════════════
        # LAYER 0: Posture validation (MUST PASS for valid rep counting)
        # ═══════════════════════════════════════════════════════════════════════
        if landmarks is not None:
            detected_posture = self.detect_posture(landmarks)
            result.posture_validation = self.validate_posture(detected_posture)
            
            # If posture is invalid, we still track angles but flag the result
            if not result.posture_validation.is_valid:
                logger.warning(
                    f"POSTURE MISMATCH: Expected {result.posture_validation.expected_postures}, "
                    f"detected {detected_posture.value} for {self.exercise_type}"
                )
                # Update feedback to indicate posture issue
                result.current_focus = f"Wrong position! Please {self._get_posture_correction()}"
                result.feedback_priority = [result.current_focus]
        else:
            # No landmarks provided - skip posture validation (backward compatibility)
            result.posture_validation = PostureValidation(
                detected_posture=BasePosture.UNKNOWN,
                expected_postures=self._get_expected_posture_names(),
                is_valid=True,  # Assume valid if no landmarks provided
                confidence=0.0,
                mismatch_reason="",
            )
        
        # Add all angles to history
        for name, angle in angles.items():
            self.add_angle_sample(name, angle, timestamp)
        
        primary_angle = angles.get(primary_angle_name, 0)
        dt = timestamp - self.last_analysis_time if self.last_analysis_time > 0 else 0.033  # ~30fps
        
        # ═══════════════════════════════════════════════════════════════════════
        # LAYER 1: Angle-based rep detection
        # ═══════════════════════════════════════════════════════════════════════
        layer1_result = self._analyze_angles(primary_angle, threshold_up, threshold_down, timestamp)
        
        # ═══════════════════════════════════════════════════════════════════════
        # LAYER 2: Velocity & smoothness analysis
        # ═══════════════════════════════════════════════════════════════════════
        velocity_metrics, layer2_score = self._analyze_velocity(primary_angle, dt, timestamp)
        result.velocity_metrics = velocity_metrics
        
        # ═══════════════════════════════════════════════════════════════════════
        # LAYER 3: Temporal pattern analysis
        # ═══════════════════════════════════════════════════════════════════════
        layer3_score = self._analyze_pattern()
        
        # ═══════════════════════════════════════════════════════════════════════
        # Combine layers for rep detection
        # IMPORTANT: Only count rep if posture is valid!
        # ═══════════════════════════════════════════════════════════════════════
        result.rep_result = self._combine_rep_detection(
            layer1_result, layer2_score, layer3_score, timestamp
        )
        
        # Invalidate rep if posture doesn't match
        if landmarks is not None and not result.posture_validation.is_valid:
            if result.rep_result.rep_completed:
                logger.warning("Rep NOT counted due to posture mismatch!")
                result.rep_result.rep_completed = False
                result.rep_result.confidence *= 0.1  # Drastically reduce confidence
        
        # ═══════════════════════════════════════════════════════════════════════
        # Enhanced pain detection
        # ═══════════════════════════════════════════════════════════════════════
        result.pain_indicators = self._analyze_pain_indicators(angles, timestamp)
        
        # ═══════════════════════════════════════════════════════════════════════
        # Form guidance (Reference skeleton matching)
        # Provides feedback on how to match ideal form without reference video
        # ═══════════════════════════════════════════════════════════════════════
        result.form_guidance = self.get_form_guidance(
            current_angles=angles,
            primary_angle=primary_angle,
            threshold_up=threshold_up,
            threshold_down=threshold_down
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # Adaptive feedback
        # ═══════════════════════════════════════════════════════════════════════
        if result.posture_validation.is_valid or landmarks is None:
            result.feedback_priority, result.current_focus, result.adaptation_suggestion = \
                self._generate_adaptive_feedback(result, angles)
            
            # If form is not ideal, add form guidance to feedback
            if result.form_guidance and not result.form_guidance.get("is_acceptable", True):
                priority_fix = result.form_guidance.get("priority_fix", "")
                if priority_fix and priority_fix not in result.feedback_priority:
                    result.feedback_priority.insert(0, priority_fix)
                    if not result.current_focus or result.current_focus == "":
                        result.current_focus = priority_fix
        
        # Track timing
        self.last_analysis_time = timestamp
        self.last_primary_angle = primary_angle
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _get_expected_posture_names(self) -> List[str]:
        """Get expected posture names for current exercise."""
        expected = self.EXPECTED_POSTURES.get(self.exercise_type, [BasePosture.STANDING])
        return [p.value for p in expected]
    
    def _get_posture_correction(self) -> str:
        """Get correction instruction for posture."""
        expected = self.EXPECTED_POSTURES.get(self.exercise_type, [BasePosture.STANDING])
        if BasePosture.STANDING in expected and BasePosture.SITTING in expected:
            return "stand near a chair or sit down"
        elif BasePosture.STANDING in expected:
            return "stand upright"
        elif BasePosture.SITTING in expected:
            return "sit down on a chair"
        elif BasePosture.LYING_SUPINE in expected:
            return "lie on your back"
        else:
            return "adjust your position"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LAYER 1: ANGLE-BASED ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_angles(
        self,
        primary_angle: float,
        threshold_up: float,
        threshold_down: float,
        timestamp: float
    ) -> Tuple[bool, float, MovementPhase]:
        """
        Layer 1: Angle-based rep detection with hysteresis.
        
        Returns: (rep_completed, confidence, new_phase)
        """
        rep_completed = False
        confidence = 0.0
        prev_phase = self.current_phase
        
        # Track phase angles
        self.phase_angles.append(primary_angle)
        
        # State machine with hysteresis
        if self.current_phase == MovementPhase.NEUTRAL:
            if primary_angle > threshold_up:
                self.current_phase = MovementPhase.ASCENDING
                self.phase_start_time = timestamp
                self.phase_angles = [primary_angle]
            elif primary_angle < threshold_down:
                self.current_phase = MovementPhase.DESCENDING
                self.phase_start_time = timestamp
                self.phase_angles = [primary_angle]
        
        elif self.current_phase == MovementPhase.ASCENDING:
            # Detect peak (angle starts decreasing)
            if len(self.phase_angles) >= 3:
                recent = self.phase_angles[-3:]
                if recent[-1] < recent[-2] < recent[-3]:
                    self.current_phase = MovementPhase.PEAK
                    self._record_rom("ascending", max(self.phase_angles))
        
        elif self.current_phase == MovementPhase.PEAK:
            if primary_angle < threshold_down:
                self.current_phase = MovementPhase.DESCENDING
                self.phase_start_time = timestamp
                self.phase_angles = [primary_angle]
        
        elif self.current_phase == MovementPhase.DESCENDING:
            # Detect bottom (angle starts increasing) OR crosses threshold
            if primary_angle < threshold_down:
                if len(self.phase_angles) >= 3:
                    recent = self.phase_angles[-3:]
                    if recent[-1] > recent[-2] > recent[-3]:
                        # Rep completed!
                        rep_completed = True
                        confidence = 1.0
                        self._record_rom("descending", min(self.phase_angles))
                        self._record_rep_timing(timestamp)
                        self.current_phase = MovementPhase.NEUTRAL
                        self.phase_angles = []
            elif primary_angle > threshold_up:
                # Started ascending again
                self.current_phase = MovementPhase.ASCENDING
                self.phase_start_time = timestamp
                self.phase_angles = [primary_angle]
        
        # Calculate confidence based on how clearly we're in a phase
        mid_point = (threshold_up + threshold_down) / 2
        range_size = abs(threshold_up - threshold_down)
        deviation = abs(primary_angle - mid_point)
        confidence = min(1.0, deviation / (range_size / 2)) if range_size > 0 else 0.5
        
        return (rep_completed, confidence, self.current_phase)
    
    def _record_rom(self, direction: str, extreme_angle: float):
        """Record range of motion for a phase."""
        key = f"{self.exercise_type}_{direction}"
        
        if not self.baseline_established:
            # Record for baseline
            if key not in self.rom_baseline:
                self.rom_baseline[key] = extreme_angle
            else:
                # Update baseline to max observed
                if direction == "ascending":
                    self.rom_baseline[key] = max(self.rom_baseline[key], extreme_angle)
                else:
                    self.rom_baseline[key] = min(self.rom_baseline[key], extreme_angle)
        
        # Always record current
        self.rom_current[key] = extreme_angle
    
    def _record_rep_timing(self, timestamp: float):
        """Record rep timing for tempo analysis."""
        if self.rep_start_time > 0:
            duration = timestamp - self.rep_start_time
            self.rep_durations.append(duration)
            
            # Track baseline reps
            self.reps_for_baseline += 1
            if self.reps_for_baseline >= self.baseline_reps:
                self.baseline_established = True
        
        self.rep_start_time = timestamp
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LAYER 2: VELOCITY & SMOOTHNESS ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_velocity(
        self,
        current_angle: float,
        dt: float,
        timestamp: float
    ) -> Tuple[VelocityMetrics, float]:
        """
        Layer 2: Velocity and smoothness analysis.
        
        Returns: (VelocityMetrics, layer2_score)
        """
        metrics = VelocityMetrics()
        
        if dt <= 0:
            return metrics, 0.5
        
        # Calculate velocity (deg/s)
        velocity = (current_angle - self.last_primary_angle) / dt if dt > 0 else 0
        self.velocity_history.append((timestamp, velocity))
        
        # Calculate acceleration
        if len(self.velocity_history) >= 2:
            prev_v = self.velocity_history[-2][1]
            acceleration = (velocity - prev_v) / dt
            self.acceleration_history.append((timestamp, acceleration))
        
        # Calculate jerk (rate of acceleration change) - smoothness metric
        if len(self.acceleration_history) >= 2:
            prev_a = self.acceleration_history[-2][1]
            current_a = self.acceleration_history[-1][1]
            jerk = abs(current_a - prev_a) / dt
            metrics.jerk = jerk
            
            # Smoothness score (lower jerk = smoother)
            if jerk < self.JERK_THRESHOLD_GOOD:
                metrics.smoothness_score = 100
            elif jerk > self.JERK_THRESHOLD_POOR:
                metrics.smoothness_score = 30
            else:
                # Linear interpolation
                metrics.smoothness_score = 100 - 70 * (jerk - self.JERK_THRESHOLD_GOOD) / \
                    (self.JERK_THRESHOLD_POOR - self.JERK_THRESHOLD_GOOD)
        
        # Current velocity metrics
        metrics.current_velocity = abs(velocity)
        
        # Peak and average from history
        if self.velocity_history:
            velocities = [abs(v) for _, v in self.velocity_history]
            metrics.peak_velocity = max(velocities)
            metrics.avg_velocity = sum(velocities) / len(velocities)
            
            # Track initial velocities for slowing detection
            if len(self.initial_velocities) < 5 and metrics.avg_velocity > self.MIN_MOVEMENT_VELOCITY:
                self.initial_velocities.append(metrics.avg_velocity)
                if len(self.initial_velocities) == 5:
                    self.velocity_baseline = sum(self.initial_velocities) / len(self.initial_velocities)
        
        # Tempo analysis
        if self.rep_durations:
            metrics.rep_duration = self.rep_durations[-1] if self.rep_durations else 0
            metrics.avg_rep_duration = sum(self.rep_durations) / len(self.rep_durations)
            
            if len(self.rep_durations) >= 2:
                metrics.tempo_variance = np.std(list(self.rep_durations))
                # Tempo score: lower variance = better consistency
                # Target: <0.5s variance = excellent, >2s = poor
                if metrics.tempo_variance < 0.5:
                    metrics.tempo_score = 100
                elif metrics.tempo_variance > 2.0:
                    metrics.tempo_score = 30
                else:
                    metrics.tempo_score = 100 - 47 * (metrics.tempo_variance - 0.5) / 1.5
        
        # Layer 2 score: combination of smoothness and tempo
        layer2_score = (metrics.smoothness_score * 0.6 + metrics.tempo_score * 0.4) / 100
        
        return metrics, layer2_score
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LAYER 3: TEMPORAL PATTERN ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_pattern(self) -> float:
        """
        Layer 3: Temporal pattern analysis.
        
        Uses angle history to compare current movement to ideal pattern.
        Returns pattern_score (0-1).
        """
        # For now, simple pattern consistency check
        # TODO: Could add DTW comparison to ideal reference pattern
        
        if len(self.rep_durations) < 2:
            return 0.5  # Not enough data
        
        # Check if rep timing is consistent
        avg_duration = sum(self.rep_durations) / len(self.rep_durations)
        if avg_duration <= 0:
            return 0.5
        
        # Score based on relative variance
        variance = np.var(list(self.rep_durations))
        cv = math.sqrt(variance) / avg_duration  # Coefficient of variation
        
        # CV < 0.1 = very consistent, CV > 0.5 = very inconsistent
        if cv < 0.1:
            return 1.0
        elif cv > 0.5:
            return 0.3
        else:
            return 1.0 - 0.7 * (cv - 0.1) / 0.4
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINE REP DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _combine_rep_detection(
        self,
        layer1_result: Tuple[bool, float, MovementPhase],
        layer2_score: float,
        layer3_score: float,
        timestamp: float
    ) -> RepDetectionResult:
        """Combine all layers for final rep detection."""
        rep_completed, angle_confidence, phase = layer1_result
        
        result = RepDetectionResult()
        result.rep_completed = rep_completed
        result.current_phase = phase
        result.angle_score = angle_confidence
        result.velocity_score = layer2_score
        result.pattern_score = layer3_score
        
        # Weighted combination (Layer 1 is primary, others support)
        result.confidence = (
            angle_confidence * 0.6 +
            layer2_score * 0.25 +
            layer3_score * 0.15
        )
        
        # Phase duration
        result.phase_duration = timestamp - self.phase_start_time if self.phase_start_time > 0 else 0
        
        # Rep duration (if rep just completed)
        if rep_completed and self.rep_durations:
            result.rep_duration = self.rep_durations[-1]
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENHANCED PAIN DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _analyze_pain_indicators(
        self,
        angles: Dict[str, float],
        timestamp: float
    ) -> PainIndicators:
        """Enhanced pain and discomfort detection."""
        indicators = PainIndicators()
        
        # ─── 1. SHAKING DETECTION ─────────────────────────────────────────────
        # Look for high variance in angle history (trembling)
        shaking_score = self._detect_shaking()
        indicators.shaking_confidence = shaking_score
        indicators.shaking_detected = shaking_score > 0.4
        if indicators.shaking_detected:
            indicators.details.append("Trembling/shaking detected in movement")
        
        # ─── 2. SLOWING DETECTION ─────────────────────────────────────────────
        # Compare current velocity to baseline
        slowing_score = self._detect_slowing()
        indicators.slowing_confidence = slowing_score
        indicators.slowing_detected = slowing_score > 0.5
        if indicators.slowing_detected:
            indicators.details.append("Movement slowing down significantly")
        
        # ─── 3. ASYMMETRY DETECTION ───────────────────────────────────────────
        asymmetry_score, affected = self._detect_asymmetry(angles)
        indicators.asymmetry_confidence = asymmetry_score
        indicators.asymmetry_detected = asymmetry_score > 0.4
        if indicators.asymmetry_detected:
            indicators.details.append(f"Asymmetric movement: {', '.join(affected)}")
            indicators.affected_joints.extend(affected)
        
        # ─── 4. HESITATION DETECTION ──────────────────────────────────────────
        hesitation_score = self._detect_hesitation(timestamp)
        indicators.hesitation_confidence = hesitation_score
        indicators.hesitation_detected = hesitation_score > 0.5
        if indicators.hesitation_detected:
            indicators.details.append("Hesitation/pause detected before movement")
        
        # ─── 5. ROM REDUCTION DETECTION ───────────────────────────────────────
        rom_score = self._detect_rom_reduction()
        indicators.rom_reduction_confidence = rom_score
        indicators.rom_reduction_detected = rom_score > 0.4
        if indicators.rom_reduction_detected:
            indicators.details.append("Range of motion decreasing from baseline")
            
            # Add ROM details
            if self.rom_baseline and self.rom_current:
                baseline_vals = list(self.rom_baseline.values())
                current_vals = list(self.rom_current.values())
                if baseline_vals and current_vals:
                    indicators.rom_baseline = sum(baseline_vals) / len(baseline_vals)
                    indicators.rom_current = sum(current_vals) / len(current_vals)
        
        # ─── OVERALL CONFIDENCE & RECOMMENDATION ──────────────────────────────
        # Weighted combination
        indicators.overall_confidence = (
            indicators.shaking_confidence * 0.25 +
            indicators.slowing_confidence * 0.20 +
            indicators.asymmetry_confidence * 0.20 +
            indicators.hesitation_confidence * 0.15 +
            indicators.rom_reduction_confidence * 0.20
        )
        
        # Generate recommendation
        if indicators.overall_confidence > 0.7:
            indicators.recommendation = "stop_exercise"
        elif indicators.overall_confidence > 0.5:
            indicators.recommendation = "take_break"
        elif indicators.overall_confidence > 0.3:
            indicators.recommendation = "reduce_intensity"
        else:
            indicators.recommendation = "continue"
        
        return indicators
    
    def _detect_shaking(self) -> float:
        """Detect shaking/trembling from angle variance."""
        if not self.angle_history:
            return 0.0
        
        total_variance = 0.0
        joints_checked = 0
        
        for joint_name, history in self.angle_history.items():
            if len(history) < 10:
                continue
            
            # Get recent angles
            recent = [angle for _, angle in list(history)[-10:]]
            
            # Calculate short-term variance (shaking = high frequency, low amplitude)
            variance = np.var(recent)
            
            # Normalize: higher variance = more shaking
            normalized = min(1.0, variance / self.SHAKING_VARIANCE_THRESHOLD)
            total_variance += normalized
            joints_checked += 1
        
        return total_variance / joints_checked if joints_checked > 0 else 0.0
    
    def _detect_slowing(self) -> float:
        """Detect movement slowing (fatigue/pain)."""
        if not self.velocity_history or len(self.velocity_history) < 10:
            return 0.0
        
        if self.velocity_baseline <= 0:
            return 0.0
        
        # Current velocity
        recent_velocities = [abs(v) for _, v in list(self.velocity_history)[-5:]]
        current_avg = sum(recent_velocities) / len(recent_velocities) if recent_velocities else 0
        
        # Compare to baseline
        ratio = current_avg / self.velocity_baseline if self.velocity_baseline > 0 else 1.0
        
        # If current is less than threshold ratio of baseline, we're slowing
        if ratio < self.SLOWING_RATIO_THRESHOLD:
            return 1.0 - ratio / self.SLOWING_RATIO_THRESHOLD
        
        return 0.0
    
    def _detect_asymmetry(self, angles: Dict[str, float]) -> Tuple[float, List[str]]:
        """Detect left/right asymmetry."""
        pairs = [
            ("left_knee", "right_knee"),
            ("left_hip", "right_hip"),
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
        ]
        
        total_asymmetry = 0.0
        affected = []
        pairs_checked = 0
        
        for left, right in pairs:
            if left in angles and right in angles:
                diff = abs(angles[left] - angles[right])
                if diff > self.ASYMMETRY_THRESHOLD:
                    affected.append(left.replace("left_", ""))
                    total_asymmetry += min(1.0, diff / (self.ASYMMETRY_THRESHOLD * 2))
                pairs_checked += 1
        
        avg_asymmetry = total_asymmetry / pairs_checked if pairs_checked > 0 else 0.0
        return avg_asymmetry, affected
    
    def _detect_hesitation(self, timestamp: float) -> float:
        """Detect pause/hesitation before movement."""
        if not self.velocity_history or len(self.velocity_history) < 5:
            return 0.0
        
        # Look for periods of near-zero velocity
        recent = list(self.velocity_history)[-10:]
        
        low_velocity_duration = 0.0
        prev_time = None
        
        for t, v in recent:
            if abs(v) < self.MIN_MOVEMENT_VELOCITY:
                if prev_time:
                    low_velocity_duration += t - prev_time
                prev_time = t
            else:
                prev_time = None
        
        # Score based on duration of pause
        if low_velocity_duration > self.HESITATION_PAUSE_THRESHOLD:
            return min(1.0, low_velocity_duration / (self.HESITATION_PAUSE_THRESHOLD * 2))
        
        return 0.0
    
    def _detect_rom_reduction(self) -> float:
        """Detect reduction in range of motion from baseline."""
        if not self.baseline_established or not self.rom_baseline or not self.rom_current:
            return 0.0
        
        reductions = []
        
        for key, baseline in self.rom_baseline.items():
            if key in self.rom_current and baseline != 0:
                current = self.rom_current[key]
                reduction = 1.0 - (current / baseline)
                
                if reduction > self.ROM_REDUCTION_THRESHOLD:
                    reductions.append(reduction)
        
        if not reductions:
            return 0.0
        
        avg_reduction = sum(reductions) / len(reductions)
        return min(1.0, avg_reduction / (self.ROM_REDUCTION_THRESHOLD * 2))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ADAPTIVE FEEDBACK
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _generate_adaptive_feedback(
        self,
        result: MovementAnalysisResult,
        angles: Dict[str, float]
    ) -> Tuple[List[str], str, str]:
        """
        Generate prioritized, adaptive feedback.
        
        Returns: (priority_list, current_focus, adaptation_suggestion)
        """
        feedback_items = []
        
        # Priority 1: Pain indicators (most important)
        if result.pain_indicators.overall_confidence > 0.3:
            for detail in result.pain_indicators.details[:2]:
                feedback_items.append((100, detail))
        
        # Priority 2: Smoothness issues
        if result.velocity_metrics.smoothness_score < 60:
            feedback_items.append((80, "Try to keep movements smooth and controlled"))
        
        if result.velocity_metrics.jerk > self.JERK_THRESHOLD_GOOD:
            feedback_items.append((75, "Reduce jerky movements"))
        
        # Priority 3: Tempo consistency
        if result.velocity_metrics.tempo_score < 60 and len(self.rep_durations) >= 3:
            feedback_items.append((70, f"Keep a steady tempo (avg {result.velocity_metrics.avg_rep_duration:.1f}s per rep)"))
        
        # Priority 4: ROM feedback
        if result.pain_indicators.rom_reduction_detected:
            feedback_items.append((65, "Try to maintain full range of motion"))
        
        # Sort by priority (highest first)
        feedback_items.sort(key=lambda x: x[0], reverse=True)
        feedback_list = [item[1] for item in feedback_items]
        
        # Current focus: single most important thing
        current_focus = feedback_list[0] if feedback_list else "Keep up the good work!"
        
        # Adaptation suggestion based on overall state
        adaptation = ""
        if result.pain_indicators.recommendation == "stop_exercise":
            adaptation = "Consider stopping and resting - potential injury risk"
        elif result.pain_indicators.recommendation == "take_break":
            adaptation = "Take a short break before continuing"
        elif result.pain_indicators.recommendation == "reduce_intensity":
            adaptation = "Try reducing the range of motion or number of reps"
        elif result.velocity_metrics.smoothness_score < 50:
            adaptation = "Focus on slow, controlled movements"
        elif result.velocity_metrics.tempo_score < 50:
            adaptation = "Try counting to maintain consistent tempo (e.g., 2 up, 2 down)"
        else:
            adaptation = ""
        
        return feedback_list[:5], current_focus, adaptation


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_movement_analyzers: Dict[str, MovementAnalyzer] = {}


def get_movement_analyzer(exercise_type: str) -> MovementAnalyzer:
    """Get or create a MovementAnalyzer for an exercise type."""
    if exercise_type not in _movement_analyzers:
        _movement_analyzers[exercise_type] = MovementAnalyzer(exercise_type)
    return _movement_analyzers[exercise_type]


def reset_movement_analyzer(exercise_type: str):
    """Reset analyzer for a new session."""
    if exercise_type in _movement_analyzers:
        _movement_analyzers[exercise_type].reset()
