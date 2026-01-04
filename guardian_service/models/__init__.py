"""
SMARTCARE+ Guardian Service Models

Fall detection, gait analysis, TUG test, and activity monitoring.
"""

from .gait_analyzer import (
    GaitAnalyzer,
    GaitMetrics,
    GaitAnalysisResult,
    GaitPattern,
    FallRiskLevel,
    FootPosition,
    get_gait_analyzer
)

from .tug_handler import (
    TUGTestHandler,
    TUGTestResult,
    TUGPhase,
    TUGPhaseResult,
    MobilityLevel,
    ActivityClassifier,
    ActivityType,
    get_tug_handler
)

from .video_classifier import (
    VideoClassifier,
    VideoClassificationResult,
    VideoActivityType,
    get_video_classifier
)

__all__ = [
    # Gait Analyzer
    "GaitAnalyzer",
    "GaitMetrics",
    "GaitAnalysisResult",
    "GaitPattern",
    "FallRiskLevel",
    "FootPosition",
    "get_gait_analyzer",
    # TUG Handler
    "TUGTestHandler",
    "TUGTestResult",
    "TUGPhase",
    "TUGPhaseResult",
    "MobilityLevel",
    "ActivityClassifier",
    "ActivityType",
    "get_tug_handler",
    # Video Classifier (Deep Learning)
    "VideoClassifier",
    "VideoClassificationResult",
    "VideoActivityType",
    "get_video_classifier",
]
