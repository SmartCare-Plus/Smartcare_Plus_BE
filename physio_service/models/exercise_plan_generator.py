"""
SMARTCARE+ Physio Service - Exercise Plan Generator

Owner: Neelaka
Rule-based personalized exercise plan generation using:
- Patient profile (BMI, age, arthritis severity)
- Affected joints and mobility level
- Pain history from face mesh detection
- Exercise performance history
- Lifestyle factors and goals
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
from enum import Enum
import random
import logging
import sys

from .patient_profile import (
    PatientProfile,
    ArthritisType,
    ArthritisSeverity,
    JointLocation,
    MobilityLevel,
    ActivityLevel,
    BMICategory,
    get_patient_profile_store
)
from .pain_data_store import get_pain_data_store


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

logger = _setup_logger("smartcare.physio.plan_generator")


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class ExerciseDifficulty(Enum):
    """Exercise difficulty levels."""
    VERY_EASY = "very_easy"
    EASY = "easy"
    MODERATE = "moderate"
    CHALLENGING = "challenging"


class ExerciseCategory(Enum):
    """Exercise categories for elderly physiotherapy."""
    WARM_UP = "warm_up"
    STRETCHING = "stretching"
    STRENGTH = "strength"
    BALANCE = "balance"
    FLEXIBILITY = "flexibility"
    COOL_DOWN = "cool_down"
    BREATHING = "breathing"


class TargetArea(Enum):
    """Body areas targeted by exercises."""
    UPPER_BODY = "upper_body"
    LOWER_BODY = "lower_body"
    CORE = "core"
    FULL_BODY = "full_body"
    NECK = "neck"
    SHOULDERS = "shoulders"
    HIPS = "hips"
    KNEES = "knees"
    ANKLES = "ankles"


class PlanDuration(Enum):
    """Plan duration options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE PHASE DEFINITIONS (for progressive reference poses)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExercisePhase:
    """A single phase in an exercise movement with reference angles."""
    name: str  # e.g., "start", "peak", "return"
    description: str  # e.g., "Stand upright with arms at sides"
    target_angles: Dict[str, float]  # e.g., {"shoulder_angle": 0, "elbow_angle": 180}
    tolerance: float = 15.0  # Degrees tolerance for matching
    hold_seconds: float = 0.0  # How long to hold this position
    visual_cue: str = ""  # Short instruction shown on screen
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "target_angles": self.target_angles,
            "tolerance": self.tolerance,
            "hold_seconds": self.hold_seconds,
            "visual_cue": self.visual_cue,
        }


@dataclass
class ExercisePhaseSequence:
    """Complete phase sequence for an exercise rep."""
    exercise_id: str
    phases: List[ExercisePhase]
    is_repeating: bool = True  # Does this sequence repeat for each rep?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exercise_id": self.exercise_id,
            "phases": [p.to_dict() for p in self.phases],
            "is_repeating": self.is_repeating,
        }


# ─── Reference Pose Sequences for Exercises ───────────────────────────────────
# Each exercise has a sequence of phases the user follows. The ghost skeleton
# shows the current target phase, then animates to the next when user matches.

EXERCISE_PHASES: Dict[str, ExercisePhaseSequence] = {
    "neck_rotations": ExercisePhaseSequence(
        exercise_id="neck_rotations",
        phases=[
            ExercisePhase(
                name="center",
                description="Head facing forward",
                target_angles={},  # No measurable angles for head rotation
                visual_cue="Face forward",
            ),
            ExercisePhase(
                name="right",
                description="Turn head to the right",
                target_angles={},
                hold_seconds=1.0,
                visual_cue="Turn right slowly",
            ),
            ExercisePhase(
                name="center2",
                description="Return to center",
                target_angles={},
                visual_cue="Return to center",
            ),
            ExercisePhase(
                name="left",
                description="Turn head to the left",
                target_angles={},
                hold_seconds=1.0,
                visual_cue="Turn left slowly",
            ),
        ],
    ),
    
    "shoulder_rolls": ExercisePhaseSequence(
        exercise_id="shoulder_rolls",
        phases=[
            ExercisePhase(
                name="ready",
                description="Shoulders relaxed",
                target_angles={"left_shoulder": 15, "right_shoulder": 15},
                visual_cue="Relax shoulders",
            ),
            ExercisePhase(
                name="up",
                description="Raise shoulders up",
                target_angles={"left_shoulder": 30, "right_shoulder": 30},
                visual_cue="Lift shoulders up",
            ),
            ExercisePhase(
                name="back",
                description="Roll shoulders back",
                target_angles={"left_shoulder": 25, "right_shoulder": 25},
                visual_cue="Roll back",
            ),
            ExercisePhase(
                name="down",
                description="Drop shoulders down",
                target_angles={"left_shoulder": 10, "right_shoulder": 10},
                visual_cue="Drop down",
            ),
        ],
    ),
    
    "arm_raise": ExercisePhaseSequence(
        exercise_id="arm_raise",
        phases=[
            ExercisePhase(
                name="start",
                description="Arms at sides",
                target_angles={"left_shoulder": 0, "right_shoulder": 0},
                visual_cue="Arms down",
            ),
            ExercisePhase(
                name="raise",
                description="Raise arms overhead",
                target_angles={"left_shoulder": 170, "right_shoulder": 170},
                hold_seconds=1.0,
                visual_cue="Raise arms up",
            ),
            ExercisePhase(
                name="lower",
                description="Lower arms slowly",
                target_angles={"left_shoulder": 0, "right_shoulder": 0},
                visual_cue="Lower slowly",
            ),
        ],
    ),
    
    "chair_stand": ExercisePhaseSequence(
        exercise_id="chair_stand",
        phases=[
            ExercisePhase(
                name="seated",
                description="Seated in chair",
                target_angles={"left_knee": 90, "right_knee": 90, "left_hip": 90, "right_hip": 90},
                visual_cue="Sit back",
            ),
            ExercisePhase(
                name="rising",
                description="Begin to stand",
                target_angles={"left_knee": 120, "right_knee": 120, "left_hip": 120, "right_hip": 120},
                visual_cue="Lean forward, push up",
            ),
            ExercisePhase(
                name="standing",
                description="Standing upright",
                target_angles={"left_knee": 170, "right_knee": 170, "left_hip": 170, "right_hip": 170},
                hold_seconds=1.0,
                visual_cue="Stand tall",
            ),
            ExercisePhase(
                name="lowering",
                description="Begin to sit",
                target_angles={"left_knee": 120, "right_knee": 120, "left_hip": 120, "right_hip": 120},
                visual_cue="Lower slowly",
            ),
        ],
    ),
    
    "squat": ExercisePhaseSequence(
        exercise_id="squat",
        phases=[
            ExercisePhase(
                name="standing",
                description="Standing upright",
                target_angles={"left_knee": 170, "right_knee": 170},
                visual_cue="Stand tall",
            ),
            ExercisePhase(
                name="descending",
                description="Lowering into squat",
                target_angles={"left_knee": 120, "right_knee": 120},
                visual_cue="Lower down",
            ),
            ExercisePhase(
                name="bottom",
                description="Bottom of squat",
                target_angles={"left_knee": 90, "right_knee": 90},
                hold_seconds=0.5,
                visual_cue="Hold briefly",
            ),
            ExercisePhase(
                name="ascending",
                description="Rising from squat",
                target_angles={"left_knee": 120, "right_knee": 120},
                visual_cue="Push up",
            ),
        ],
    ),
    
    "leg_raise": ExercisePhaseSequence(
        exercise_id="leg_raise",
        phases=[
            ExercisePhase(
                name="down",
                description="Leg down",
                target_angles={"left_hip": 180, "left_knee": 170},
                visual_cue="Leg relaxed",
            ),
            ExercisePhase(
                name="raise",
                description="Lift leg forward",
                target_angles={"left_hip": 90, "left_knee": 170},
                hold_seconds=1.0,
                visual_cue="Lift leg up",
            ),
            ExercisePhase(
                name="lower",
                description="Lower leg slowly",
                target_angles={"left_hip": 180, "left_knee": 170},
                visual_cue="Lower slowly",
            ),
        ],
    ),
    
    "wall_pushup": ExercisePhaseSequence(
        exercise_id="wall_pushup",
        phases=[
            ExercisePhase(
                name="start",
                description="Arms extended against wall",
                target_angles={"left_elbow": 170, "right_elbow": 170},
                visual_cue="Arms straight",
            ),
            ExercisePhase(
                name="lean",
                description="Lean towards wall",
                target_angles={"left_elbow": 90, "right_elbow": 90},
                hold_seconds=0.5,
                visual_cue="Bend elbows",
            ),
            ExercisePhase(
                name="push",
                description="Push back from wall",
                target_angles={"left_elbow": 170, "right_elbow": 170},
                visual_cue="Push away",
            ),
        ],
    ),
    
    "deep_breathing": ExercisePhaseSequence(
        exercise_id="deep_breathing",
        phases=[
            ExercisePhase(
                name="exhale",
                description="Exhale, shoulders relaxed",
                target_angles={"left_shoulder": 10, "right_shoulder": 10},
                hold_seconds=3.0,
                visual_cue="Breathe out",
            ),
            ExercisePhase(
                name="inhale",
                description="Inhale, chest expands",
                target_angles={"left_shoulder": 20, "right_shoulder": 20},
                hold_seconds=4.0,
                visual_cue="Breathe in deeply",
            ),
        ],
    ),
    
    "ankle_circles": ExercisePhaseSequence(
        exercise_id="ankle_circles",
        phases=[
            ExercisePhase(
                name="center",
                description="Foot neutral",
                target_angles={"left_ankle": 90, "right_ankle": 90},
                visual_cue="Center position",
            ),
            ExercisePhase(
                name="point",
                description="Point toes down",
                target_angles={"left_ankle": 120, "right_ankle": 120},
                visual_cue="Point toes",
            ),
            ExercisePhase(
                name="flex",
                description="Flex foot up",
                target_angles={"left_ankle": 70, "right_ankle": 70},
                visual_cue="Pull toes up",
            ),
        ],
    ),
    
    # ─── Additional Exercises ───
    
    "heel_toe_walk": ExercisePhaseSequence(
        exercise_id="heel_toe_walk",
        phases=[
            ExercisePhase(
                name="heel_step",
                description="Step forward landing on heel",
                target_angles={"left_knee": 170, "right_knee": 170},
                visual_cue="Step heel first",
            ),
            ExercisePhase(
                name="toe_push",
                description="Roll through foot and push off toes",
                target_angles={"left_knee": 160, "right_knee": 160},
                visual_cue="Push off toes",
            ),
        ],
    ),
    
    "single_leg_stand": ExercisePhaseSequence(
        exercise_id="single_leg_stand",
        phases=[
            ExercisePhase(
                name="both_feet",
                description="Standing on both feet",
                target_angles={"left_hip": 180, "right_hip": 180},
                visual_cue="Stand tall",
            ),
            ExercisePhase(
                name="lift",
                description="Lift one leg off ground",
                target_angles={"left_hip": 150},
                hold_seconds=5.0,
                visual_cue="Lift knee, hold balance",
            ),
            ExercisePhase(
                name="lower",
                description="Lower leg back down",
                target_angles={"left_hip": 180},
                visual_cue="Lower slowly",
            ),
        ],
    ),
    
    "tandem_stand": ExercisePhaseSequence(
        exercise_id="tandem_stand",
        phases=[
            ExercisePhase(
                name="position",
                description="Heel-to-toe position",
                target_angles={"left_hip": 180, "right_hip": 180},
                hold_seconds=10.0,
                visual_cue="Heel touching toe, hold balance",
            ),
        ],
    ),
    
    "marching": ExercisePhaseSequence(
        exercise_id="marching",
        phases=[
            ExercisePhase(
                name="stand",
                description="Standing position",
                target_angles={"left_hip": 180, "right_hip": 180},
                visual_cue="Stand tall",
            ),
            ExercisePhase(
                name="lift_left",
                description="Lift left knee",
                target_angles={"left_hip": 90},
                visual_cue="Lift left knee high",
            ),
            ExercisePhase(
                name="down_left",
                description="Lower left leg",
                target_angles={"left_hip": 180},
                visual_cue="Lower left",
            ),
            ExercisePhase(
                name="lift_right",
                description="Lift right knee",
                target_angles={"right_hip": 90},
                visual_cue="Lift right knee high",
            ),
        ],
    ),
    
    "marching_in_place": ExercisePhaseSequence(
        exercise_id="marching_in_place",
        phases=[
            ExercisePhase(
                name="stand",
                description="Standing position",
                target_angles={"left_hip": 180, "right_hip": 180},
                visual_cue="Stand tall",
            ),
            ExercisePhase(
                name="lift_left",
                description="Lift left knee",
                target_angles={"left_hip": 90},
                visual_cue="Lift left knee",
            ),
            ExercisePhase(
                name="down_left",
                description="Lower left leg",
                target_angles={"left_hip": 180},
                visual_cue="Lower left",
            ),
            ExercisePhase(
                name="lift_right",
                description="Lift right knee",
                target_angles={"right_hip": 90},
                visual_cue="Lift right knee",
            ),
        ],
    ),
    
    "seated_hamstring_stretch": ExercisePhaseSequence(
        exercise_id="seated_hamstring_stretch",
        phases=[
            ExercisePhase(
                name="seated",
                description="Seated with legs extended",
                target_angles={"left_knee": 170, "right_knee": 170},
                visual_cue="Sit with legs straight",
            ),
            ExercisePhase(
                name="reach",
                description="Reach forward towards toes",
                target_angles={"left_hip": 70, "right_hip": 70},
                hold_seconds=15.0,
                visual_cue="Reach forward, hold stretch",
            ),
            ExercisePhase(
                name="release",
                description="Release and sit upright",
                target_angles={"left_hip": 90, "right_hip": 90},
                visual_cue="Sit back up",
            ),
        ],
    ),
    
    "seated_hip_stretch": ExercisePhaseSequence(
        exercise_id="seated_hip_stretch",
        phases=[
            ExercisePhase(
                name="seated",
                description="Seated in chair",
                target_angles={"left_hip": 90, "right_hip": 90},
                visual_cue="Sit upright",
            ),
            ExercisePhase(
                name="cross",
                description="Cross ankle over knee",
                target_angles={"left_hip": 80},
                hold_seconds=15.0,
                visual_cue="Cross ankle, lean forward gently",
            ),
            ExercisePhase(
                name="release",
                description="Release and switch sides",
                target_angles={"left_hip": 90, "right_hip": 90},
                visual_cue="Switch to other side",
            ),
        ],
    ),
    
    "calf_stretch": ExercisePhaseSequence(
        exercise_id="calf_stretch",
        phases=[
            ExercisePhase(
                name="stand",
                description="Standing near wall",
                target_angles={"left_knee": 170, "right_knee": 170},
                visual_cue="Stand tall, hands on wall",
            ),
            ExercisePhase(
                name="stretch",
                description="Step back and press heel down",
                target_angles={"left_knee": 170, "right_knee": 150},
                hold_seconds=15.0,
                visual_cue="Press back heel down, hold",
            ),
            ExercisePhase(
                name="switch",
                description="Switch legs",
                target_angles={"left_knee": 150, "right_knee": 170},
                hold_seconds=15.0,
                visual_cue="Switch sides, hold stretch",
            ),
        ],
    ),
    
    "seated_leg_raise": ExercisePhaseSequence(
        exercise_id="seated_leg_raise",
        phases=[
            ExercisePhase(
                name="seated",
                description="Seated with feet flat",
                target_angles={"left_knee": 90, "right_knee": 90},
                visual_cue="Sit tall",
            ),
            ExercisePhase(
                name="extend",
                description="Extend one leg straight",
                target_angles={"left_knee": 170},
                hold_seconds=2.0,
                visual_cue="Straighten leg, hold",
            ),
            ExercisePhase(
                name="lower",
                description="Lower leg back down",
                target_angles={"left_knee": 90},
                visual_cue="Lower slowly",
            ),
        ],
    ),
    
    "seated_arm_raises": ExercisePhaseSequence(
        exercise_id="seated_arm_raises",
        phases=[
            ExercisePhase(
                name="rest",
                description="Arms at sides",
                target_angles={"left_shoulder": 0, "right_shoulder": 0},
                visual_cue="Arms relaxed",
            ),
            ExercisePhase(
                name="raise",
                description="Raise arms overhead",
                target_angles={"left_shoulder": 170, "right_shoulder": 170},
                hold_seconds=1.0,
                visual_cue="Raise arms up",
            ),
            ExercisePhase(
                name="lower",
                description="Lower arms back down",
                target_angles={"left_shoulder": 0, "right_shoulder": 0},
                visual_cue="Lower slowly",
            ),
        ],
    ),
    
    "gentle_spinal_twist": ExercisePhaseSequence(
        exercise_id="gentle_spinal_twist",
        phases=[
            ExercisePhase(
                name="center",
                description="Seated facing forward",
                target_angles={"trunk_rotation": 0},
                visual_cue="Face forward",
            ),
            ExercisePhase(
                name="twist_right",
                description="Rotate torso to right",
                target_angles={"trunk_rotation": 30},
                hold_seconds=10.0,
                visual_cue="Turn right, hold gently",
            ),
            ExercisePhase(
                name="center2",
                description="Return to center",
                target_angles={"trunk_rotation": 0},
                visual_cue="Return to center",
            ),
            ExercisePhase(
                name="twist_left",
                description="Rotate torso to left",
                target_angles={"trunk_rotation": -30},
                hold_seconds=10.0,
                visual_cue="Turn left, hold gently",
            ),
        ],
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExerciseDefinition:
    """Definition of an exercise in the library."""
    exercise_id: str
    name: str
    description: str
    category: ExerciseCategory
    difficulty: ExerciseDifficulty
    target_areas: List[TargetArea]
    
    # Default parameters
    default_reps: int = 10
    default_sets: int = 2
    default_hold_seconds: int = 0  # For stretches
    rest_between_sets_seconds: int = 30
    
    # Safety
    requires_chair: bool = False
    requires_wall: bool = False
    requires_standing: bool = False
    supervision_recommended: bool = False
    
    # Contraindications - joints where this exercise should be avoided
    contraindicated_joints: List[JointLocation] = field(default_factory=list)
    
    # Benefits
    benefits: List[str] = field(default_factory=list)
    
    # Instructions
    instructions: List[str] = field(default_factory=list)
    
    # Video ID for demo (matches exercise video files)
    video_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exercise_id": self.exercise_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "target_areas": [t.value for t in self.target_areas],
            "default_reps": self.default_reps,
            "default_sets": self.default_sets,
            "default_hold_seconds": self.default_hold_seconds,
            "rest_between_sets_seconds": self.rest_between_sets_seconds,
            "requires_chair": self.requires_chair,
            "requires_wall": self.requires_wall,
            "requires_standing": self.requires_standing,
            "supervision_recommended": self.supervision_recommended,
            "contraindicated_joints": [j.value for j in self.contraindicated_joints],
            "benefits": self.benefits,
            "instructions": self.instructions,
            "video_id": self.video_id,
        }


# ── Exercise Library ─────────────────────────────────────────────────────────

EXERCISE_LIBRARY: Dict[str, ExerciseDefinition] = {
    # ─── WARM UP ───
    "neck_rotations": ExerciseDefinition(
        exercise_id="neck_rotations",
        name="Gentle Neck Rotations",
        description="Slowly rotate your neck in a circular motion to loosen up",
        category=ExerciseCategory.WARM_UP,
        difficulty=ExerciseDifficulty.VERY_EASY,
        target_areas=[TargetArea.NECK],
        default_reps=5,
        default_sets=2,
        requires_chair=True,
        benefits=["Improves neck mobility", "Reduces stiffness"],
        instructions=[
            "Sit comfortably in a chair",
            "Slowly tilt head to the right",
            "Roll head forward and to the left",
            "Complete the circle back to center",
            "Repeat in opposite direction"
        ],
        video_id="neck_rotations",
        contraindicated_joints=[JointLocation.NECK],
    ),
    
    "shoulder_rolls": ExerciseDefinition(
        exercise_id="shoulder_rolls",
        name="Shoulder Rolls",
        description="Roll shoulders forward and backward to warm up",
        category=ExerciseCategory.WARM_UP,
        difficulty=ExerciseDifficulty.VERY_EASY,
        target_areas=[TargetArea.SHOULDERS],
        default_reps=10,
        default_sets=2,
        requires_chair=True,
        benefits=["Loosens shoulder joints", "Reduces tension"],
        instructions=[
            "Sit or stand comfortably",
            "Roll shoulders forward in circles",
            "Make 10 circles forward",
            "Reverse direction for 10 circles backward"
        ],
        video_id="shoulder_rolls",
        contraindicated_joints=[JointLocation.LEFT_SHOULDER, JointLocation.RIGHT_SHOULDER],
    ),
    
    "ankle_circles": ExerciseDefinition(
        exercise_id="ankle_circles",
        name="Ankle Circles",
        description="Rotate ankles to improve circulation and mobility",
        category=ExerciseCategory.WARM_UP,
        difficulty=ExerciseDifficulty.VERY_EASY,
        target_areas=[TargetArea.ANKLES],
        default_reps=10,
        default_sets=2,
        requires_chair=True,
        benefits=["Improves ankle mobility", "Increases blood flow to feet"],
        instructions=[
            "Sit with one leg extended",
            "Rotate ankle clockwise 10 times",
            "Rotate ankle counter-clockwise 10 times",
            "Switch to other leg"
        ],
        video_id="ankle_circles",
        contraindicated_joints=[JointLocation.LEFT_ANKLE, JointLocation.RIGHT_ANKLE],
    ),
    
    # ─── STRETCHING ───
    "seated_hamstring_stretch": ExerciseDefinition(
        exercise_id="seated_hamstring_stretch",
        name="Seated Hamstring Stretch",
        description="Stretch the back of your thighs while seated",
        category=ExerciseCategory.STRETCHING,
        difficulty=ExerciseDifficulty.EASY,
        target_areas=[TargetArea.LOWER_BODY],
        default_reps=3,
        default_sets=1,
        default_hold_seconds=20,
        requires_chair=True,
        benefits=["Improves leg flexibility", "Reduces lower back tension"],
        instructions=[
            "Sit on edge of chair",
            "Extend one leg straight with heel on floor",
            "Keep back straight, lean forward from hips",
            "Hold for 20 seconds",
            "Switch legs"
        ],
        video_id="seated_hamstring_stretch",
    ),
    
    "seated_hip_stretch": ExerciseDefinition(
        exercise_id="seated_hip_stretch",
        name="Seated Hip Stretch",
        description="Gently stretch hip muscles while seated",
        category=ExerciseCategory.STRETCHING,
        difficulty=ExerciseDifficulty.EASY,
        target_areas=[TargetArea.HIPS],
        default_reps=3,
        default_sets=1,
        default_hold_seconds=20,
        requires_chair=True,
        benefits=["Improves hip mobility", "Reduces hip stiffness"],
        instructions=[
            "Sit in chair with feet flat",
            "Cross one ankle over opposite knee",
            "Gently press down on raised knee",
            "Hold for 20 seconds",
            "Switch sides"
        ],
        video_id="seated_hip_stretch",
        contraindicated_joints=[JointLocation.LEFT_HIP, JointLocation.RIGHT_HIP],
    ),
    
    "calf_stretch": ExerciseDefinition(
        exercise_id="calf_stretch",
        name="Seated Calf Stretch",
        description="Stretch calf muscles using a towel",
        category=ExerciseCategory.STRETCHING,
        difficulty=ExerciseDifficulty.EASY,
        target_areas=[TargetArea.LOWER_BODY],
        default_reps=3,
        default_sets=1,
        default_hold_seconds=15,
        requires_chair=True,
        benefits=["Improves calf flexibility", "Helps with walking"],
        instructions=[
            "Sit with one leg extended",
            "Loop a towel around the ball of your foot",
            "Gently pull towel toward you",
            "Hold for 15 seconds",
            "Switch legs"
        ],
        video_id="calf_stretch",
    ),
    
    # ─── STRENGTH ───
    "chair_stand": ExerciseDefinition(
        exercise_id="chair_stand",
        name="Chair Stand",
        description="Stand up and sit down to strengthen leg muscles",
        category=ExerciseCategory.STRENGTH,
        difficulty=ExerciseDifficulty.MODERATE,
        target_areas=[TargetArea.LOWER_BODY, TargetArea.CORE],
        default_reps=10,
        default_sets=2,
        rest_between_sets_seconds=60,
        requires_chair=True,
        requires_standing=True,
        benefits=["Strengthens quadriceps", "Improves functional mobility", "Core engagement"],
        instructions=[
            "Sit on edge of sturdy chair",
            "Cross arms over chest",
            "Lean slightly forward",
            "Stand up using leg muscles",
            "Slowly sit back down",
            "Control the movement"
        ],
        video_id="chair_stand",
        contraindicated_joints=[JointLocation.LEFT_KNEE, JointLocation.RIGHT_KNEE],
    ),
    
    "wall_pushup": ExerciseDefinition(
        exercise_id="wall_pushup",
        name="Wall Push-Up",
        description="Modified push-up against a wall for upper body strength",
        category=ExerciseCategory.STRENGTH,
        difficulty=ExerciseDifficulty.MODERATE,
        target_areas=[TargetArea.UPPER_BODY, TargetArea.CORE],
        default_reps=10,
        default_sets=2,
        rest_between_sets_seconds=45,
        requires_wall=True,
        requires_standing=True,
        benefits=["Strengthens chest and arms", "Core stability", "Low joint stress"],
        instructions=[
            "Stand facing wall at arm's length",
            "Place palms flat on wall at shoulder height",
            "Bend elbows and lean toward wall",
            "Push back to starting position",
            "Keep body straight throughout"
        ],
        video_id="wall_pushup",
        contraindicated_joints=[JointLocation.LEFT_WRIST, JointLocation.RIGHT_WRIST, 
                               JointLocation.LEFT_SHOULDER, JointLocation.RIGHT_SHOULDER],
    ),
    
    "seated_leg_raise": ExerciseDefinition(
        exercise_id="seated_leg_raise",
        name="Seated Leg Raise",
        description="Lift one leg at a time while seated to strengthen thighs",
        category=ExerciseCategory.STRENGTH,
        difficulty=ExerciseDifficulty.EASY,
        target_areas=[TargetArea.LOWER_BODY],
        default_reps=10,
        default_sets=2,
        requires_chair=True,
        benefits=["Strengthens quadriceps", "Improves leg control"],
        instructions=[
            "Sit with back against chair",
            "Hold sides of chair for support",
            "Slowly lift one leg straight out",
            "Hold for 2 seconds at top",
            "Lower slowly",
            "Repeat with other leg"
        ],
        video_id="seated_leg_raise",
    ),
    
    "arm_raise": ExerciseDefinition(
        exercise_id="arm_raise",
        name="Seated Arm Raises",
        description="Raise arms overhead to strengthen shoulders",
        category=ExerciseCategory.STRENGTH,
        difficulty=ExerciseDifficulty.EASY,
        target_areas=[TargetArea.UPPER_BODY, TargetArea.SHOULDERS],
        default_reps=10,
        default_sets=2,
        requires_chair=True,
        benefits=["Strengthens shoulders", "Improves overhead reach"],
        instructions=[
            "Sit with feet flat on floor",
            "Start with arms at sides",
            "Slowly raise both arms overhead",
            "Lower slowly back down",
            "Keep movements controlled"
        ],
        video_id="arm_raise",
        contraindicated_joints=[JointLocation.LEFT_SHOULDER, JointLocation.RIGHT_SHOULDER],
    ),
    
    # ─── BALANCE ───
    "tandem_stand": ExerciseDefinition(
        exercise_id="tandem_stand",
        name="Tandem Stand",
        description="Stand with one foot in front of the other",
        category=ExerciseCategory.BALANCE,
        difficulty=ExerciseDifficulty.MODERATE,
        target_areas=[TargetArea.LOWER_BODY, TargetArea.CORE],
        default_reps=1,
        default_sets=4,
        default_hold_seconds=30,
        requires_standing=True,
        supervision_recommended=True,
        benefits=["Improves balance", "Strengthens stabilizing muscles"],
        instructions=[
            "Stand near a wall or chair for support",
            "Place one foot directly in front of the other",
            "Heel of front foot touches toes of back foot",
            "Hold position for 30 seconds",
            "Switch feet and repeat"
        ],
        video_id="tandem_stand",
    ),
    
    "single_leg_stand": ExerciseDefinition(
        exercise_id="single_leg_stand",
        name="Single Leg Stand",
        description="Balance on one leg to improve stability",
        category=ExerciseCategory.BALANCE,
        difficulty=ExerciseDifficulty.CHALLENGING,
        target_areas=[TargetArea.LOWER_BODY, TargetArea.CORE],
        default_reps=1,
        default_sets=4,
        default_hold_seconds=20,
        requires_standing=True,
        requires_chair=True,  # For support if needed
        supervision_recommended=True,
        benefits=["Improves single-leg balance", "Strengthens ankles"],
        instructions=[
            "Stand next to a chair for support",
            "Lift one foot slightly off ground",
            "Hold position for up to 20 seconds",
            "Use chair only if needed for balance",
            "Switch legs and repeat"
        ],
        video_id="single_leg_stand",
        contraindicated_joints=[JointLocation.LEFT_ANKLE, JointLocation.RIGHT_ANKLE,
                               JointLocation.LEFT_KNEE, JointLocation.RIGHT_KNEE],
    ),
    
    "heel_toe_walk": ExerciseDefinition(
        exercise_id="heel_toe_walk",
        name="Heel-to-Toe Walk",
        description="Walk in a line placing heel directly in front of toes",
        category=ExerciseCategory.BALANCE,
        difficulty=ExerciseDifficulty.MODERATE,
        target_areas=[TargetArea.LOWER_BODY, TargetArea.CORE],
        default_reps=20,  # 20 steps
        default_sets=2,
        requires_standing=True,
        supervision_recommended=True,
        benefits=["Improves gait balance", "Strengthens leg coordination"],
        instructions=[
            "Stand near a wall for support if needed",
            "Step forward placing heel directly in front of toes",
            "Take 20 steps heel-to-toe",
            "Turn around and repeat"
        ],
        video_id="heel_toe_walk",
    ),
    
    "marching": ExerciseDefinition(
        exercise_id="marching",
        name="Marching in Place",
        description="March in place lifting knees",
        category=ExerciseCategory.BALANCE,
        difficulty=ExerciseDifficulty.EASY,
        target_areas=[TargetArea.LOWER_BODY, TargetArea.CORE],
        default_reps=30,
        default_sets=2,
        requires_standing=True,
        requires_chair=True,  # For support
        benefits=["Improves hip flexor strength", "Gentle cardio", "Balance training"],
        instructions=[
            "Stand behind a chair, holding backrest",
            "March in place lifting knees high",
            "Keep back straight",
            "March for 30 steps"
        ],
        video_id="marching",
    ),
    
    # ─── BREATHING / COOL DOWN ───
    "deep_breathing": ExerciseDefinition(
        exercise_id="deep_breathing",
        name="Deep Breathing Exercise",
        description="Controlled breathing to relax and recover",
        category=ExerciseCategory.BREATHING,
        difficulty=ExerciseDifficulty.VERY_EASY,
        target_areas=[TargetArea.CORE],
        default_reps=5,
        default_sets=1,
        requires_chair=True,
        benefits=["Reduces stress", "Improves oxygen flow", "Promotes relaxation"],
        instructions=[
            "Sit comfortably with hands on belly",
            "Breathe in slowly through nose for 4 counts",
            "Hold breath for 2 counts",
            "Exhale slowly through mouth for 6 counts",
            "Feel belly rise and fall with breath"
        ],
        video_id="deep_breathing",
    ),
    
    "gentle_spinal_twist": ExerciseDefinition(
        exercise_id="gentle_spinal_twist",
        name="Seated Spinal Twist",
        description="Gentle twist to release lower back tension",
        category=ExerciseCategory.COOL_DOWN,
        difficulty=ExerciseDifficulty.EASY,
        target_areas=[TargetArea.CORE, TargetArea.LOWER_BODY],
        default_reps=3,
        default_sets=1,
        default_hold_seconds=15,
        requires_chair=True,
        benefits=["Releases back tension", "Improves spinal mobility"],
        instructions=[
            "Sit upright in chair",
            "Cross right leg over left",
            "Twist torso to the right",
            "Place left hand on right knee",
            "Hold for 15 seconds",
            "Repeat on other side"
        ],
        video_id="gentle_spinal_twist",
        contraindicated_joints=[JointLocation.LOWER_BACK],
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE PLAN DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlannedExercise:
    """A single exercise in a plan with personalized parameters."""
    exercise: ExerciseDefinition
    prescribed_reps: int
    prescribed_sets: int
    hold_seconds: int = 0
    rest_seconds: int = 30
    notes: str = ""
    order: int = 0
    
    # Adaptations made
    adapted_from_default: bool = False
    adaptation_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exercise_id": self.exercise.exercise_id,
            "name": self.exercise.name,
            "description": self.exercise.description,
            "category": self.exercise.category.value,
            "difficulty": self.exercise.difficulty.value,
            "prescribed_reps": self.prescribed_reps,
            "prescribed_sets": self.prescribed_sets,
            "hold_seconds": self.hold_seconds,
            "rest_seconds": self.rest_seconds,
            "requires_chair": self.exercise.requires_chair,
            "requires_wall": self.exercise.requires_wall,
            "requires_standing": self.exercise.requires_standing,
            "instructions": self.exercise.instructions,
            "benefits": self.exercise.benefits,
            "video_id": self.exercise.video_id,
            "notes": self.notes,
            "order": self.order,
            "adapted": self.adapted_from_default,
            "adaptation_reason": self.adaptation_reason,
        }


@dataclass 
class DailyExercisePlan:
    """A complete daily exercise plan."""
    plan_id: str
    user_id: str
    date: date
    created_at: datetime = field(default_factory=datetime.now)
    
    # Plan content
    exercises: List[PlannedExercise] = field(default_factory=list)
    
    # Plan metadata
    total_duration_minutes: int = 0
    difficulty_level: str = "easy"
    focus_areas: List[str] = field(default_factory=list)
    
    # Generation context
    based_on_profile: bool = True
    adapted_for_pain: bool = False
    pain_adaptation_notes: str = ""
    
    # User notes
    additional_notes: List[str] = field(default_factory=list)
    
    # Completion tracking
    completed: bool = False
    completed_at: Optional[datetime] = None
    completion_feedback: Optional[str] = None
    completed_exercises: List[str] = field(default_factory=list)  # List of completed exercise_ids
    
    @property
    def exercise_count(self) -> int:
        return len(self.exercises)
    
    @property
    def completed_exercise_count(self) -> int:
        return len(self.completed_exercises)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "user_id": self.user_id,
            "date": self.date.isoformat(),
            "created_at": self.created_at.isoformat(),
            "exercises": [e.to_dict() for e in self.exercises],
            "exercise_count": self.exercise_count,
            "completed_exercise_count": self.completed_exercise_count,
            "total_duration_minutes": self.total_duration_minutes,
            "difficulty_level": self.difficulty_level,
            "focus_areas": self.focus_areas,
            "based_on_profile": self.based_on_profile,
            "adapted_for_pain": self.adapted_for_pain,
            "pain_adaptation_notes": self.pain_adaptation_notes,
            "additional_notes": self.additional_notes,
            "completed": self.completed,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "completion_feedback": self.completion_feedback,
            "completed_exercises": self.completed_exercises,
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE PLAN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class ExercisePlanGenerator:
    """
    Rule-based personalized exercise plan generator.
    
    Considers:
    - Patient profile (BMI, age, arthritis)
    - Affected joints (avoids contraindicated exercises)
    - Pain history (adapts intensity)
    - Mobility level
    - Available equipment (chair, wall)
    
    Plans are persisted to Firestore for cross-session persistence.
    """
    
    COLLECTION_NAME = 'physio_exercise_plans'
    
    # Plan structure by difficulty
    PLAN_STRUCTURE = {
        "very_easy": {
            "warm_up": 2,
            "stretching": 2,
            "strength": 1,
            "balance": 0,
            "cool_down": 1,
        },
        "easy": {
            "warm_up": 2,
            "stretching": 2,
            "strength": 2,
            "balance": 1,
            "cool_down": 1,
        },
        "moderate": {
            "warm_up": 2,
            "stretching": 2,
            "strength": 3,
            "balance": 2,
            "cool_down": 1,
        },
        "challenging": {
            "warm_up": 2,
            "stretching": 2,
            "strength": 4,
            "balance": 3,
            "cool_down": 1,
        },
    }
    
    def __init__(self):
        self._plans: Dict[str, DailyExercisePlan] = {}
        self._db = None
        self._init_firestore()
    
    def _init_firestore(self):
        """Initialize Firestore connection (lazy — retries on each access)."""
        if self._db:
            return True
        try:
            from core.database import get_db
            self._db = get_db()
            if self._db:
                logger.info("📊 ExercisePlanGenerator connected to Firestore")
                return True
            else:
                logger.debug("⚠️ Firestore not yet available")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Firestore init failed: {e}")
            return False

    def _ensure_db(self):
        """Ensure Firestore connection is available (lazy init)."""
        if not self._db:
            self._init_firestore()
        return self._db is not None
    
    def _plan_to_dict(self, plan: DailyExercisePlan) -> Dict[str, Any]:
        """Convert DailyExercisePlan to Firestore-compatible dict."""
        return {
            "plan_id": plan.plan_id,
            "user_id": plan.user_id,
            "date": plan.date.isoformat(),
            "created_at": plan.created_at.isoformat(),
            "exercises": [e.to_dict() for e in plan.exercises],
            "total_duration_minutes": plan.total_duration_minutes,
            "difficulty_level": plan.difficulty_level,
            "focus_areas": plan.focus_areas,
            "based_on_profile": plan.based_on_profile,
            "adapted_for_pain": plan.adapted_for_pain,
            "pain_adaptation_notes": plan.pain_adaptation_notes,
            "additional_notes": plan.additional_notes,
            "completed": plan.completed,
            "completed_at": plan.completed_at.isoformat() if plan.completed_at else None,
            "completion_feedback": plan.completion_feedback,
            "completed_exercises": plan.completed_exercises,
        }
    
    def _dict_to_plan(self, data: Dict[str, Any]) -> DailyExercisePlan:
        """Convert Firestore dict to DailyExercisePlan."""
        plan = DailyExercisePlan(
            plan_id=data["plan_id"],
            user_id=data["user_id"],
            date=date.fromisoformat(data["date"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            total_duration_minutes=data.get("total_duration_minutes", 0),
            difficulty_level=data.get("difficulty_level", "easy"),
            focus_areas=data.get("focus_areas", []),
            based_on_profile=data.get("based_on_profile", True),
            adapted_for_pain=data.get("adapted_for_pain", False),
            pain_adaptation_notes=data.get("pain_adaptation_notes", ""),
            additional_notes=data.get("additional_notes", []),
            completed=data.get("completed", False),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            completion_feedback=data.get("completion_feedback"),
            completed_exercises=data.get("completed_exercises", []),
        )
        
        # Reconstruct exercises from stored data
        exercises_data = data.get("exercises", [])
        for ex_data in exercises_data:
            # Find the exercise definition from EXERCISE_LIBRARY
            exercise_id = ex_data.get("exercise_id", "")
            exercise_def = EXERCISE_LIBRARY.get(exercise_id)
            if exercise_def:
                planned = PlannedExercise(
                    exercise=exercise_def,
                    prescribed_reps=ex_data.get("prescribed_reps", 10),
                    prescribed_sets=ex_data.get("prescribed_sets", 1),
                    hold_seconds=ex_data.get("hold_seconds", 0),
                    rest_seconds=ex_data.get("rest_seconds", 30),
                    notes=ex_data.get("notes", ""),
                    order=ex_data.get("order", 0),
                    adapted_from_default=ex_data.get("adapted", False),
                    adaptation_reason=ex_data.get("adaptation_reason", ""),
                )
                plan.exercises.append(planned)
        
        return plan
    
    def _save_to_firestore(self, plan: DailyExercisePlan):
        """Save plan to Firestore."""
        if not self._ensure_db():
            logger.warning(f"⚠️ Firestore unavailable — cannot save plan {plan.plan_id}")
            return
        try:
            data = self._plan_to_dict(plan)
            self._db.collection(self.COLLECTION_NAME).document(plan.plan_id).set(data)
            logger.debug(f"💾 Saved plan to Firestore: {plan.plan_id}")
        except Exception as e:
            logger.error(f"❌ Failed to save plan to Firestore: {e}")
    
    def _load_plan_from_firestore(self, plan_id: str) -> Optional[DailyExercisePlan]:
        """Load a single plan from Firestore."""
        if not self._ensure_db():
            return None
        try:
            doc = self._db.collection(self.COLLECTION_NAME).document(plan_id).get()
            if doc.exists:
                plan = self._dict_to_plan(doc.to_dict())
                logger.debug(f"📥 Loaded plan from Firestore: {plan_id}")
                return plan
        except Exception as e:
            logger.error(f"❌ Failed to load plan from Firestore: {e}")
        return None
    
    def _load_user_plans_from_firestore(self, user_id: str, days: int = 7) -> List[DailyExercisePlan]:
        """Load user's plans from Firestore."""
        if not self._ensure_db():
            logger.warning(f"⚠️ Firestore unavailable — cannot load plans for user {user_id}")
            return []
        try:
            cutoff = (date.today() - timedelta(days=days)).isoformat()
            # Use only user_id filter to avoid composite index requirement,
            # then filter by date in Python
            docs = (self._db.collection(self.COLLECTION_NAME)
                    .where("user_id", "==", user_id)
                    .stream())
            plans = []
            for doc in docs:
                try:
                    doc_data = doc.to_dict()
                    # Filter by date in Python (avoids composite index)
                    doc_date = doc_data.get("date", "")
                    if doc_date >= cutoff:
                        plan = self._dict_to_plan(doc_data)
                        plans.append(plan)
                        # Also cache it
                        self._plans[plan.plan_id] = plan
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse plan {doc.id}: {e}")
            logger.debug(f"📥 Loaded {len(plans)} plans from Firestore for user: {user_id}")
            return plans
        except Exception as e:
            logger.error(f"❌ Failed to load user plans from Firestore: {e}")
        return []
    
    def generate_daily_plan(
        self,
        user_id: str,
        plan_date: Optional[date] = None,
        override_difficulty: Optional[str] = None
    ) -> DailyExercisePlan:
        """
        Generate a personalized daily exercise plan.
        
        Args:
            user_id: Patient user ID
            plan_date: Date for the plan (default: today)
            override_difficulty: Force a specific difficulty level
        
        Returns:
            DailyExercisePlan with personalized exercises
        """
        plan_date = plan_date or date.today()
        
        # Get patient profile
        profile_store = get_patient_profile_store()
        profile = profile_store.get_profile(user_id)
        
        # Get pain history
        pain_store = get_pain_data_store()
        pain_history = pain_store.get_user_pain_history(user_id, days=7)
        
        # Determine difficulty level
        if override_difficulty:
            difficulty = override_difficulty
        else:
            difficulty = self._determine_difficulty(profile, pain_history)
        
        # Generate plan ID using UUID to avoid collisions after server restart
        import uuid
        plan_id = f"plan_{user_id}_{plan_date.isoformat()}_{uuid.uuid4().hex[:8]}"
        
        # Create plan
        plan = DailyExercisePlan(
            plan_id=plan_id,
            user_id=user_id,
            date=plan_date,
            difficulty_level=difficulty,
            based_on_profile=profile is not None,
        )
        
        # Get contraindicated joints
        contraindicated = self._get_contraindicated_joints(profile)
        
        # Check available equipment
        has_chair = True if not profile else profile.lifestyle.has_chair_for_support
        has_wall = True if not profile else profile.lifestyle.has_wall_for_support
        can_stand = True
        if profile and profile.lifestyle.mobility_level in [MobilityLevel.DEPENDENT, MobilityLevel.WHEELCHAIR]:
            can_stand = False
        
        # Get plan structure
        structure = self.PLAN_STRUCTURE.get(difficulty, self.PLAN_STRUCTURE["easy"])
        
        # Select exercises for each category
        order = 0
        focus_areas = set()
        
        # 1. Warm up
        warm_up_exercises = self._select_exercises(
            category=ExerciseCategory.WARM_UP,
            count=structure["warm_up"],
            contraindicated=contraindicated,
            difficulty_max=difficulty,
            has_chair=has_chair,
            has_wall=has_wall,
            can_stand=can_stand
        )
        for ex in warm_up_exercises:
            order += 1
            plan.exercises.append(self._create_planned_exercise(
                ex, profile, pain_history, order
            ))
            focus_areas.update(t.value for t in ex.target_areas)
        
        # 2. Stretching
        stretch_exercises = self._select_exercises(
            category=ExerciseCategory.STRETCHING,
            count=structure["stretching"],
            contraindicated=contraindicated,
            difficulty_max=difficulty,
            has_chair=has_chair,
            has_wall=has_wall,
            can_stand=can_stand
        )
        for ex in stretch_exercises:
            order += 1
            plan.exercises.append(self._create_planned_exercise(
                ex, profile, pain_history, order
            ))
            focus_areas.update(t.value for t in ex.target_areas)
        
        # 3. Strength
        strength_exercises = self._select_exercises(
            category=ExerciseCategory.STRENGTH,
            count=structure["strength"],
            contraindicated=contraindicated,
            difficulty_max=difficulty,
            has_chair=has_chair,
            has_wall=has_wall,
            can_stand=can_stand
        )
        for ex in strength_exercises:
            order += 1
            plan.exercises.append(self._create_planned_exercise(
                ex, profile, pain_history, order
            ))
            focus_areas.update(t.value for t in ex.target_areas)
        
        # 4. Balance (if can stand)
        if can_stand and structure["balance"] > 0:
            balance_exercises = self._select_exercises(
                category=ExerciseCategory.BALANCE,
                count=structure["balance"],
                contraindicated=contraindicated,
                difficulty_max=difficulty,
                has_chair=has_chair,
                has_wall=has_wall,
                can_stand=can_stand
            )
            for ex in balance_exercises:
                order += 1
                plan.exercises.append(self._create_planned_exercise(
                    ex, profile, pain_history, order
                ))
                focus_areas.update(t.value for t in ex.target_areas)
        
        # 5. Cool down / Breathing
        cooldown_exercises = self._select_exercises(
            category=ExerciseCategory.COOL_DOWN,
            count=structure["cool_down"],
            contraindicated=contraindicated,
            difficulty_max=difficulty,
            has_chair=has_chair,
            has_wall=has_wall,
            can_stand=can_stand
        )
        # Add breathing exercise
        if "deep_breathing" in EXERCISE_LIBRARY:
            cooldown_exercises.append(EXERCISE_LIBRARY["deep_breathing"])
        
        for ex in cooldown_exercises:
            order += 1
            plan.exercises.append(self._create_planned_exercise(
                ex, profile, pain_history, order
            ))
            focus_areas.update(t.value for t in ex.target_areas)
        
        # Calculate total duration (rough estimate)
        plan.total_duration_minutes = self._estimate_duration(plan.exercises)
        plan.focus_areas = list(focus_areas)
        
        # Add pain adaptation notes
        if pain_history and pain_history.sessions_with_pain > 0:
            plan.adapted_for_pain = True
            plan.pain_adaptation_notes = (
                f"Intensity reduced based on {pain_history.sessions_with_pain} "
                f"sessions with pain in the last 7 days. "
                f"Pain trend: {pain_history.pain_trend}."
            )
        
        # Add profile-based notes
        if profile:
            notes = []
            if profile.age >= 80:
                notes.append("Take extra rest between exercises as needed")
            if profile.medical_history.has_balance_issues:
                notes.append("Keep support (chair/wall) within reach at all times")
            if profile.medical_history.fear_of_falling:
                notes.append("Focus on building confidence with seated exercises first")
            bmi = profile.bmi
            if bmi >= 30:
                notes.append("Prioritize low-impact movements to protect joints")
            plan.additional_notes = notes
        
        # Store plan (cache and Firestore)
        self._plans[plan_id] = plan
        self._save_to_firestore(plan)
        
        logger.info(
            f"📅 Generated daily plan: {plan_id} | "
            f"{len(plan.exercises)} exercises | "
            f"~{plan.total_duration_minutes} min | "
            f"difficulty={difficulty}"
        )
        
        return plan
    
    def _determine_difficulty(
        self,
        profile: Optional[PatientProfile],
        pain_history: Any
    ) -> str:
        """Determine appropriate difficulty level based on patient data."""
        if not profile:
            return "easy"
        
        risk_level = profile.get_exercise_risk_level()
        
        # Start with risk-based difficulty
        if risk_level == "very_high":
            difficulty = "very_easy"
        elif risk_level == "high":
            difficulty = "easy"
        elif risk_level == "moderate":
            difficulty = "moderate"
        else:
            difficulty = "moderate"
        
        # Downgrade if recent pain
        if pain_history and pain_history.pain_trend == "worsening":
            difficulties = ["very_easy", "easy", "moderate", "challenging"]
            current_idx = difficulties.index(difficulty)
            if current_idx > 0:
                difficulty = difficulties[current_idx - 1]
        
        # Consider activity level
        if profile.lifestyle.activity_level == ActivityLevel.SEDENTARY:
            difficulty = "very_easy"
        elif profile.lifestyle.activity_level == ActivityLevel.LIGHTLY_ACTIVE:
            if difficulty == "challenging":
                difficulty = "moderate"
        
        return difficulty
    
    def _get_contraindicated_joints(
        self,
        profile: Optional[PatientProfile]
    ) -> List[JointLocation]:
        """Get list of joints to avoid based on patient profile."""
        if not profile:
            return []
        
        contraindicated = []
        for joint in profile.medical_history.affected_joints:
            # Contraindicate severely affected joints
            if joint.severity in [ArthritisSeverity.SEVERE, ArthritisSeverity.MODERATE]:
                contraindicated.append(joint.location)
            # Also contraindicate if high pain
            elif joint.pain_level >= 7:
                contraindicated.append(joint.location)
        
        return contraindicated
    
    def _select_exercises(
        self,
        category: ExerciseCategory,
        count: int,
        contraindicated: List[JointLocation],
        difficulty_max: str,
        has_chair: bool,
        has_wall: bool,
        can_stand: bool
    ) -> List[ExerciseDefinition]:
        """Select appropriate exercises from library."""
        difficulty_order = ["very_easy", "easy", "moderate", "challenging"]
        max_difficulty_idx = difficulty_order.index(difficulty_max)
        
        candidates = []
        
        for ex in EXERCISE_LIBRARY.values():
            # Check category
            if ex.category != category:
                continue
            
            # Check difficulty
            ex_difficulty_idx = difficulty_order.index(ex.difficulty.value)
            if ex_difficulty_idx > max_difficulty_idx:
                continue
            
            # Check equipment requirements
            if ex.requires_chair and not has_chair:
                continue
            if ex.requires_wall and not has_wall:
                continue
            if ex.requires_standing and not can_stand:
                continue
            
            # Check contraindications
            has_contraindication = any(
                j in contraindicated for j in ex.contraindicated_joints
            )
            if has_contraindication:
                continue
            
            candidates.append(ex)
        
        # Select requested number, shuffled for variety
        random.shuffle(candidates)
        return candidates[:count]
    
    def _create_planned_exercise(
        self,
        exercise: ExerciseDefinition,
        profile: Optional[PatientProfile],
        pain_history: Any,
        order: int
    ) -> PlannedExercise:
        """Create a PlannedExercise with personalized parameters."""
        reps = exercise.default_reps
        sets = exercise.default_sets
        hold = exercise.default_hold_seconds
        rest = exercise.rest_between_sets_seconds
        
        adapted = False
        adaptation_reason = ""
        
        # Adapt based on profile
        if profile:
            # Reduce for older patients
            if profile.age >= 80:
                reps = max(3, reps - 3)
                sets = max(1, sets - 1)
                adapted = True
                adaptation_reason = "Reduced for age 80+"
            
            # Reduce for high BMI
            if profile.bmi >= 35:
                reps = max(3, reps - 2)
                rest = min(90, rest + 15)
                adapted = True
                adaptation_reason += " Adjusted for BMI"
            
            # Reduce for low mobility
            if profile.lifestyle.mobility_level in [MobilityLevel.MODERATE_ASSIST, MobilityLevel.DEPENDENT]:
                reps = max(2, reps - 3)
                sets = 1
                adapted = True
                adaptation_reason += " Reduced for mobility level"
        
        # Adapt based on pain history
        if pain_history and pain_history.avg_pain_intensity > 0.3:
            reps = max(3, int(reps * 0.7))
            adapted = True
            adaptation_reason += " Reduced due to pain history"
        
        return PlannedExercise(
            exercise=exercise,
            prescribed_reps=reps,
            prescribed_sets=sets,
            hold_seconds=hold,
            rest_seconds=rest,
            order=order,
            adapted_from_default=adapted,
            adaptation_reason=adaptation_reason.strip(),
        )
    
    def _estimate_duration(self, exercises: List[PlannedExercise]) -> int:
        """Estimate total plan duration in minutes."""
        total_seconds = 0
        
        for ex in exercises:
            # Time per set
            if ex.hold_seconds > 0:
                # Stretch: hold time * reps
                set_time = ex.hold_seconds * ex.prescribed_reps
            else:
                # Reps: ~3 seconds per rep
                set_time = ex.prescribed_reps * 3
            
            # Total for all sets + rest between sets
            total_seconds += (set_time * ex.prescribed_sets) + (ex.rest_seconds * (ex.prescribed_sets - 1))
            
            # Transition time between exercises
            total_seconds += 15
        
        return max(10, total_seconds // 60)
    
    def get_plan(self, plan_id: str) -> Optional[DailyExercisePlan]:
        """Get a plan by ID (checks cache first, then Firestore)."""
        # Check cache first
        if plan_id in self._plans:
            return self._plans[plan_id]
        
        # Try loading from Firestore
        plan = self._load_plan_from_firestore(plan_id)
        if plan:
            self._plans[plan_id] = plan
            return plan
        
        return None
    
    def get_user_plans(self, user_id: str, days: int = 7) -> List[DailyExercisePlan]:
        """Get recent plans for a user (from cache and Firestore)."""
        # First, load from Firestore to ensure cache is up-to-date
        self._load_user_plans_from_firestore(user_id, days)
        
        # Now return from cache
        cutoff = date.today() - timedelta(days=days)
        plans = [
            p for p in self._plans.values()
            if p.user_id == user_id and p.date >= cutoff
        ]
        return sorted(plans, key=lambda p: p.date, reverse=True)
    
    def mark_plan_completed(
        self,
        plan_id: str,
        feedback: Optional[str] = None
    ) -> Optional[DailyExercisePlan]:
        """Mark a plan as completed."""
        plan = self.get_plan(plan_id)
        if plan:
            plan.completed = True
            plan.completed_at = datetime.now()
            plan.completion_feedback = feedback
            # Update in Firestore too
            self._save_to_firestore(plan)
            logger.info(f"✅ Plan completed: {plan_id}")
        return plan
    
    def mark_exercise_completed(
        self,
        plan_id: str,
        exercise_id: str
    ) -> Optional[DailyExercisePlan]:
        """Mark a single exercise as completed within a plan."""
        plan = self.get_plan(plan_id)
        if plan:
            if exercise_id not in plan.completed_exercises:
                plan.completed_exercises.append(exercise_id)
                logger.info(f"✅ Exercise '{exercise_id}' completed in plan: {plan_id}")
                
                # Auto-complete plan if all exercises are done
                if len(plan.completed_exercises) >= plan.exercise_count:
                    plan.completed = True
                    plan.completed_at = datetime.now()
                    logger.info(f"🎉 All exercises done — plan auto-completed: {plan_id}")
                
                # Persist to Firestore
                self._save_to_firestore(plan)
        return plan


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

_plan_generator_instance: Optional[ExercisePlanGenerator] = None


def get_exercise_plan_generator() -> ExercisePlanGenerator:
    """Get or create global exercise plan generator instance."""
    global _plan_generator_instance
    if _plan_generator_instance is None:
        _plan_generator_instance = ExercisePlanGenerator()
    return _plan_generator_instance
