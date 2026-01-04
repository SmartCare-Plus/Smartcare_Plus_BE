"""
SMARTCARE+ Guardian Service - Gait Analyzer

Owner: Madhushani
Gait analysis from video/pose data for fall risk assessment.
Uses pose landmarks to calculate gait metrics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import time


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class GaitPattern(Enum):
    """Identified gait patterns."""
    NORMAL = "normal"
    SHUFFLING = "shuffling"
    ANTALGIC = "antalgic"  # Pain-avoidance gait
    FESTINATING = "festinating"  # Short, accelerating steps (Parkinson's)
    ATAXIC = "ataxic"  # Uncoordinated, wide-based
    HEMIPLEGIC = "hemiplegic"  # One-sided weakness
    SPASTIC = "spastic"  # Stiff, circumduction
    ARTHRITIC = "arthritic"  # Slow, guarded movements
    UNKNOWN = "unknown"


class FallRiskLevel(Enum):
    """Fall risk assessment levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class GaitMetrics:
    """Comprehensive gait metrics from analysis."""
    # Spatial parameters
    stride_length: float = 0.0  # meters
    step_length_left: float = 0.0
    step_length_right: float = 0.0
    step_width: float = 0.0  # lateral distance between feet
    
    # Temporal parameters
    cadence: float = 0.0  # steps per minute
    gait_speed: float = 0.0  # meters per second
    stride_time: float = 0.0  # seconds
    swing_time_left: float = 0.0
    swing_time_right: float = 0.0
    stance_time_left: float = 0.0
    stance_time_right: float = 0.0
    double_support_time: float = 0.0  # both feet on ground
    
    # Symmetry and variability
    step_length_symmetry: float = 0.0  # ratio (1.0 = perfect symmetry)
    stride_time_variability: float = 0.0  # coefficient of variation
    step_width_variability: float = 0.0
    
    # Quality indicators
    trunk_sway: float = 0.0  # degrees
    arm_swing_symmetry: float = 0.0
    knee_flexion_symmetry: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spatial": {
                "stride_length_m": round(self.stride_length, 3),
                "step_length_left_m": round(self.step_length_left, 3),
                "step_length_right_m": round(self.step_length_right, 3),
                "step_width_m": round(self.step_width, 3),
            },
            "temporal": {
                "cadence_steps_min": round(self.cadence, 1),
                "gait_speed_m_s": round(self.gait_speed, 2),
                "stride_time_s": round(self.stride_time, 2),
                "double_support_pct": round(self.double_support_time * 100, 1),
            },
            "symmetry": {
                "step_length_symmetry": round(self.step_length_symmetry, 2),
                "stride_variability_cv": round(self.stride_time_variability, 3),
            },
            "quality": {
                "trunk_sway_deg": round(self.trunk_sway, 1),
                "arm_swing_symmetry": round(self.arm_swing_symmetry, 2),
            }
        }


@dataclass
class GaitAnalysisResult:
    """Complete gait analysis result."""
    metrics: GaitMetrics
    gait_pattern: GaitPattern
    fall_risk: FallRiskLevel
    fall_risk_score: float  # 0-100
    confidence: float
    timestamp: float
    
    # Interpretations
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "gait_pattern": self.gait_pattern.value,
            "fall_risk": {
                "level": self.fall_risk.value,
                "score": round(self.fall_risk_score, 1),
            },
            "confidence": round(self.confidence, 2),
            "timestamp": self.timestamp,
            "findings": self.findings,
            "recommendations": self.recommendations,
        }


@dataclass
class FootPosition:
    """Foot position at a point in time."""
    x: float
    y: float
    z: float
    timestamp: float
    foot: str  # "left" or "right"
    is_stance: bool = True  # True if foot is on ground


# ═══════════════════════════════════════════════════════════════════════════════
# GAIT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class GaitAnalyzer:
    """
    Analyzes gait patterns from pose landmark sequences.
    
    Uses temporal analysis of hip, knee, and ankle positions
    to calculate gait metrics and assess fall risk.
    """
    
    # Normal gait reference values for elderly (65+)
    NORMAL_RANGES = {
        "stride_length": (0.9, 1.4),  # meters
        "cadence": (85, 120),  # steps/min
        "gait_speed": (0.8, 1.4),  # m/s
        "step_width": (0.05, 0.15),  # meters
        "double_support": (0.15, 0.30),  # percentage of gait cycle
        "trunk_sway": (0, 8),  # degrees
        "symmetry": (0.90, 1.10),  # ratio
    }
    
    # Fall risk thresholds
    FALL_RISK_THRESHOLDS = {
        "gait_speed_low": 0.6,  # m/s - below this = high risk
        "stride_variability_high": 0.08,  # CV - above this = higher risk
        "step_width_wide": 0.20,  # m - above this = compensating
        "asymmetry_high": 0.15,  # deviation from 1.0
    }
    
    def __init__(self, pixel_to_meter_ratio: float = 0.002):
        """
        Initialize gait analyzer.
        
        Args:
            pixel_to_meter_ratio: Conversion from normalized coords to meters
                                  (depends on camera setup/calibration)
        """
        self.pixel_to_meter = pixel_to_meter_ratio
        
        # History buffers for temporal analysis
        self.left_foot_history: deque = deque(maxlen=150)  # ~5 seconds at 30fps
        self.right_foot_history: deque = deque(maxlen=150)
        self.hip_history: deque = deque(maxlen=150)
        self.shoulder_history: deque = deque(maxlen=150)
        
        # Stride detection
        self.strides: List[Dict] = []
        self.last_heel_strike_left: Optional[float] = None
        self.last_heel_strike_right: Optional[float] = None
    
    def add_pose_frame(
        self,
        landmarks: Dict[int, Any],
        timestamp: float
    ):
        """
        Add a pose frame to the analysis buffer.
        
        Args:
            landmarks: Dict of landmark ID -> {x, y, z, visibility}
            timestamp: Frame timestamp in seconds
        """
        # Extract relevant landmarks (using MediaPipe landmark IDs)
        # 27/28 = ankles, 23/24 = hips, 11/12 = shoulders
        
        left_ankle = landmarks.get(27, {})
        right_ankle = landmarks.get(28, {})
        left_hip = landmarks.get(23, {})
        right_hip = landmarks.get(24, {})
        left_shoulder = landmarks.get(11, {})
        right_shoulder = landmarks.get(12, {})
        
        # Store foot positions
        if left_ankle:
            self.left_foot_history.append(FootPosition(
                x=left_ankle.get("x", 0),
                y=left_ankle.get("y", 0),
                z=left_ankle.get("z", 0),
                timestamp=timestamp,
                foot="left"
            ))
        
        if right_ankle:
            self.right_foot_history.append(FootPosition(
                x=right_ankle.get("x", 0),
                y=right_ankle.get("y", 0),
                z=right_ankle.get("z", 0),
                timestamp=timestamp,
                foot="right"
            ))
        
        # Store hip center for velocity estimation
        if left_hip and right_hip:
            hip_center = {
                "x": (left_hip.get("x", 0) + right_hip.get("x", 0)) / 2,
                "y": (left_hip.get("y", 0) + right_hip.get("y", 0)) / 2,
                "timestamp": timestamp
            }
            self.hip_history.append(hip_center)
        
        # Store shoulder center for trunk sway
        if left_shoulder and right_shoulder:
            shoulder_center = {
                "x": (left_shoulder.get("x", 0) + right_shoulder.get("x", 0)) / 2,
                "y": (left_shoulder.get("y", 0) + right_shoulder.get("y", 0)) / 2,
                "timestamp": timestamp
            }
            self.shoulder_history.append(shoulder_center)
        
        # Detect heel strikes for stride analysis
        self._detect_heel_strikes(timestamp)
    
    def _detect_heel_strikes(self, current_time: float):
        """Detect heel strike events for stride timing."""
        if len(self.left_foot_history) < 5 or len(self.right_foot_history) < 5:
            return
        
        # Simple heel strike detection: local minimum in foot Y position
        # (foot at lowest point in vertical direction)
        
        for foot_history, last_strike_attr, foot_name in [
            (self.left_foot_history, "last_heel_strike_left", "left"),
            (self.right_foot_history, "last_heel_strike_right", "right"),
        ]:
            recent = list(foot_history)[-5:]
            y_positions = [f.y for f in recent]
            
            # Check if middle frame is local maximum (lowest foot position)
            if len(y_positions) >= 5:
                mid = len(y_positions) // 2
                if y_positions[mid] >= max(y_positions[mid-2:mid]) and \
                   y_positions[mid] >= max(y_positions[mid+1:mid+3]):
                    
                    last_strike = getattr(self, last_strike_attr)
                    if last_strike is None or current_time - last_strike > 0.3:
                        setattr(self, last_strike_attr, current_time)
                        
                        self.strides.append({
                            "foot": foot_name,
                            "timestamp": current_time,
                            "position": recent[mid]
                        })
    
    def analyze(self) -> Optional[GaitAnalysisResult]:
        """
        Perform full gait analysis on collected data.
        
        Returns:
            GaitAnalysisResult or None if insufficient data
        """
        if len(self.left_foot_history) < 30 or len(self.right_foot_history) < 30:
            return None
        
        metrics = self._calculate_metrics()
        pattern = self._identify_pattern(metrics)
        fall_risk, risk_score = self._assess_fall_risk(metrics)
        findings = self._generate_findings(metrics, pattern)
        recommendations = self._generate_recommendations(metrics, fall_risk)
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence()
        
        return GaitAnalysisResult(
            metrics=metrics,
            gait_pattern=pattern,
            fall_risk=fall_risk,
            fall_risk_score=risk_score,
            confidence=confidence,
            timestamp=time.time(),
            findings=findings,
            recommendations=recommendations
        )
    
    def _calculate_metrics(self) -> GaitMetrics:
        """Calculate all gait metrics from history."""
        metrics = GaitMetrics()
        
        # Calculate step lengths
        left_positions = [f for f in self.left_foot_history]
        right_positions = [f for f in self.right_foot_history]
        
        if len(left_positions) >= 2:
            # Step length from consecutive positions
            step_lengths_left = []
            for i in range(1, len(left_positions)):
                dx = (left_positions[i].x - left_positions[i-1].x) * self.pixel_to_meter
                step_lengths_left.append(abs(dx))
            
            if step_lengths_left:
                metrics.step_length_left = np.mean(step_lengths_left) * 100  # Scale up
        
        if len(right_positions) >= 2:
            step_lengths_right = []
            for i in range(1, len(right_positions)):
                dx = (right_positions[i].x - right_positions[i-1].x) * self.pixel_to_meter
                step_lengths_right.append(abs(dx))
            
            if step_lengths_right:
                metrics.step_length_right = np.mean(step_lengths_right) * 100
        
        # Stride length (left + right step)
        metrics.stride_length = metrics.step_length_left + metrics.step_length_right
        
        # Step width (lateral distance between feet)
        if left_positions and right_positions:
            min_len = min(len(left_positions), len(right_positions))
            widths = []
            for i in range(min_len):
                width = abs(left_positions[i].x - right_positions[i].x) * self.pixel_to_meter
                widths.append(width)
            
            if widths:
                metrics.step_width = np.mean(widths) * 100
        
        # Cadence and timing from strides
        if len(self.strides) >= 4:
            stride_times = []
            for i in range(1, len(self.strides)):
                dt = self.strides[i]["timestamp"] - self.strides[i-1]["timestamp"]
                if 0.3 < dt < 2.0:  # Valid stride time range
                    stride_times.append(dt)
            
            if stride_times:
                metrics.stride_time = np.mean(stride_times)
                metrics.cadence = 60.0 / (metrics.stride_time / 2) if metrics.stride_time > 0 else 0
                metrics.stride_time_variability = np.std(stride_times) / np.mean(stride_times) if stride_times else 0
        
        # Gait speed from hip movement
        if len(self.hip_history) >= 10:
            hip_positions = list(self.hip_history)
            total_distance = 0
            for i in range(1, len(hip_positions)):
                dx = (hip_positions[i]["x"] - hip_positions[i-1]["x"]) * self.pixel_to_meter
                total_distance += abs(dx)
            
            total_time = hip_positions[-1]["timestamp"] - hip_positions[0]["timestamp"]
            if total_time > 0:
                metrics.gait_speed = (total_distance * 100) / total_time  # Scale up
        
        # Symmetry calculations
        if metrics.step_length_left > 0 and metrics.step_length_right > 0:
            metrics.step_length_symmetry = min(
                metrics.step_length_left / metrics.step_length_right,
                metrics.step_length_right / metrics.step_length_left
            )
        
        # Trunk sway from shoulder movement
        if len(self.shoulder_history) >= 10:
            shoulder_positions = list(self.shoulder_history)
            x_positions = [s["x"] for s in shoulder_positions]
            
            # Lateral sway as standard deviation
            if x_positions:
                sway_range = max(x_positions) - min(x_positions)
                # Convert to approximate degrees (rough estimation)
                metrics.trunk_sway = sway_range * 50  # Scale factor
        
        return metrics
    
    def _identify_pattern(self, metrics: GaitMetrics) -> GaitPattern:
        """Identify gait pattern from metrics."""
        # Check for specific patterns based on metric combinations
        
        # Shuffling: short stride, low clearance
        if metrics.stride_length < 0.7 and metrics.cadence > 100:
            return GaitPattern.SHUFFLING
        
        # Festinating: accelerating short steps (common in Parkinson's)
        if metrics.stride_length < 0.6 and metrics.stride_time_variability > 0.15:
            return GaitPattern.FESTINATING
        
        # Ataxic: wide base, high variability
        if metrics.step_width > 0.18 and metrics.trunk_sway > 10:
            return GaitPattern.ATAXIC
        
        # Antalgic: asymmetry (pain avoidance)
        if metrics.step_length_symmetry < 0.85:
            return GaitPattern.ANTALGIC
        
        # Arthritic: slow, guarded
        if metrics.gait_speed < 0.6 and metrics.cadence < 80:
            return GaitPattern.ARTHRITIC
        
        # Normal gait
        if (0.8 <= metrics.gait_speed <= 1.4 and 
            85 <= metrics.cadence <= 120 and
            metrics.step_length_symmetry >= 0.90):
            return GaitPattern.NORMAL
        
        return GaitPattern.UNKNOWN
    
    def _assess_fall_risk(self, metrics: GaitMetrics) -> Tuple[FallRiskLevel, float]:
        """Assess fall risk based on gait metrics."""
        risk_score = 0.0
        
        # Gait speed (major predictor)
        if metrics.gait_speed < 0.6:
            risk_score += 30
        elif metrics.gait_speed < 0.8:
            risk_score += 15
        
        # Stride variability
        if metrics.stride_time_variability > 0.10:
            risk_score += 25
        elif metrics.stride_time_variability > 0.06:
            risk_score += 10
        
        # Step width (wide base = compensating)
        if metrics.step_width > 0.20:
            risk_score += 15
        elif metrics.step_width > 0.15:
            risk_score += 5
        
        # Asymmetry
        if metrics.step_length_symmetry < 0.85:
            risk_score += 20
        elif metrics.step_length_symmetry < 0.90:
            risk_score += 10
        
        # Trunk sway
        if metrics.trunk_sway > 12:
            risk_score += 15
        elif metrics.trunk_sway > 8:
            risk_score += 5
        
        # Slow cadence
        if metrics.cadence < 80:
            risk_score += 10
        
        # Cap at 100
        risk_score = min(100, risk_score)
        
        # Determine level
        if risk_score < 25:
            level = FallRiskLevel.LOW
        elif risk_score < 50:
            level = FallRiskLevel.MODERATE
        elif risk_score < 75:
            level = FallRiskLevel.HIGH
        else:
            level = FallRiskLevel.VERY_HIGH
        
        return level, risk_score
    
    def _generate_findings(self, metrics: GaitMetrics, pattern: GaitPattern) -> List[str]:
        """Generate clinical findings from analysis."""
        findings = []
        
        if metrics.gait_speed < 0.8:
            findings.append(f"Reduced gait speed ({metrics.gait_speed:.2f} m/s)")
        
        if metrics.step_length_symmetry < 0.90:
            findings.append("Asymmetric step length detected")
        
        if metrics.stride_time_variability > 0.06:
            findings.append("Increased stride time variability")
        
        if metrics.step_width > 0.15:
            findings.append("Wide base of support")
        
        if metrics.trunk_sway > 8:
            findings.append("Increased lateral trunk sway")
        
        if pattern != GaitPattern.NORMAL and pattern != GaitPattern.UNKNOWN:
            findings.append(f"Gait pattern: {pattern.value}")
        
        if not findings:
            findings.append("Gait parameters within normal limits")
        
        return findings
    
    def _generate_recommendations(self, metrics: GaitMetrics, risk: FallRiskLevel) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if risk in [FallRiskLevel.HIGH, FallRiskLevel.VERY_HIGH]:
            recommendations.append("Consider assistive device evaluation")
            recommendations.append("Home safety assessment recommended")
            recommendations.append("Physical therapy referral for balance training")
        
        if risk == FallRiskLevel.MODERATE:
            recommendations.append("Balance exercises recommended")
            recommendations.append("Remove home hazards (rugs, cables)")
        
        if metrics.gait_speed < 0.8:
            recommendations.append("Gait speed training exercises")
        
        if metrics.step_length_symmetry < 0.90:
            recommendations.append("Investigate cause of asymmetry")
        
        if metrics.trunk_sway > 8:
            recommendations.append("Core strengthening exercises")
        
        if risk == FallRiskLevel.LOW:
            recommendations.append("Continue regular physical activity")
            recommendations.append("Annual gait reassessment")
        
        return recommendations
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in analysis based on data quality."""
        confidence = 1.0
        
        # Reduce confidence if insufficient data
        if len(self.left_foot_history) < 60:
            confidence *= 0.7
        
        if len(self.strides) < 4:
            confidence *= 0.8
        
        return confidence
    
    def reset(self):
        """Clear all history buffers."""
        self.left_foot_history.clear()
        self.right_foot_history.clear()
        self.hip_history.clear()
        self.shoulder_history.clear()
        self.strides.clear()
        self.last_heel_strike_left = None
        self.last_heel_strike_right = None


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_analyzer: Optional[GaitAnalyzer] = None

def get_gait_analyzer() -> GaitAnalyzer:
    """Get or create gait analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = GaitAnalyzer()
    return _analyzer
