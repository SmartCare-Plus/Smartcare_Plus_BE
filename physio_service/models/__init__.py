"""
SMARTCARE+ Physio Service Models

MediaPipe-based pose analysis with rule-based exercise monitoring.
Includes facial pain detection, patient profiles, and exercise plan generation.
"""

from .pose_analyzer import (
    PoseAnalyzer,
    PoseResult,
    Landmark,
    JointType,
    JointAngle,
    ExerciseType,
    FormQuality,
    FormAssessment,
    RepCounter,
    get_pose_analyzer
)

from .exercise_session import (
    ExerciseSession,
    ExerciseSessionHandler,
    SessionState,
    RepRecord,
    SetRecord,
    get_session_handler
)

from .face_pain_analyzer import (
    FacePainAnalyzer,
    FacialPainResult,
    PainLevel as FacePainLevel,
    get_face_pain_analyzer
)

from .pain_data_store import (
    PainDataStore,
    PainEvent,
    SessionPainSummary,
    UserPainHistory,
    PainLevel,
    get_pain_data_store
)

from .patient_profile import (
    PatientProfile,
    PatientProfileStore,
    BMIResult,
    MedicalHistory,
    LifestyleFactors,
    AffectedJoint,
    Gender,
    ArthritisType,
    ArthritisSeverity,
    JointLocation,
    MobilityLevel,
    ActivityLevel,
    PainToleranceLevel,
    BMICategory,
    calculate_bmi,
    get_patient_profile_store
)

from .exercise_plan_generator import (
    ExercisePlanGenerator,
    DailyExercisePlan,
    PlannedExercise,
    ExerciseDefinition,
    ExerciseDifficulty,
    ExerciseCategory,
    TargetArea,
    EXERCISE_LIBRARY,
    get_exercise_plan_generator
)

from .movement_analyzer import (
    MovementAnalyzer,
    MovementAnalysisResult,
    MovementPhase,
    VelocityMetrics,
    PainIndicators,
    RepDetectionResult,
    get_movement_analyzer,
    reset_movement_analyzer
)

__all__ = [
    # Pose Analyzer
    "PoseAnalyzer",
    "PoseResult",
    "Landmark",
    "JointType", 
    "JointAngle",
    "ExerciseType",
    "FormQuality",
    "FormAssessment",
    "RepCounter",
    "get_pose_analyzer",
    # Exercise Session
    "ExerciseSession",
    "ExerciseSessionHandler",
    "SessionState",
    "RepRecord",
    "SetRecord",
    "get_session_handler",
    # Face Pain Analysis
    "FacePainAnalyzer",
    "FacialPainResult",
    "FacePainLevel",
    "get_face_pain_analyzer",
    # Pain Data Storage
    "PainDataStore",
    "PainEvent",
    "SessionPainSummary",
    "UserPainHistory",
    "PainLevel",
    "get_pain_data_store",
    # Patient Profile
    "PatientProfile",
    "PatientProfileStore",
    "BMIResult",
    "MedicalHistory",
    "LifestyleFactors",
    "AffectedJoint",
    "Gender",
    "ArthritisType",
    "ArthritisSeverity",
    "JointLocation",
    "MobilityLevel",
    "ActivityLevel",
    "PainToleranceLevel",
    "BMICategory",
    "calculate_bmi",
    "get_patient_profile_store",
    # Exercise Plan Generator
    "ExercisePlanGenerator",
    "DailyExercisePlan",
    "PlannedExercise",
    "ExerciseDefinition",
    "ExerciseDifficulty",
    "ExerciseCategory",
    "TargetArea",
    "EXERCISE_LIBRARY",
    "get_exercise_plan_generator",
    # Movement Analyzer (3-Layer)
    "MovementAnalyzer",
    "MovementAnalysisResult",
    "MovementPhase",
    "VelocityMetrics",
    "PainIndicators",
    "RepDetectionResult",
    "get_movement_analyzer",
    "reset_movement_analyzer",
]
