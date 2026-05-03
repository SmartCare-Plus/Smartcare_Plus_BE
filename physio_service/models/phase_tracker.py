"""
SMARTCARE+ Physio Service - Exercise Phase Tracker

Owner: Neelaka
Tracks user progress through exercise phases and provides:
- Progressive reference pose guidance (ghost skeleton)
- Phase matching detection
- Dynamic rep counting based on phase completion
- Rep/intensity adaptation based on pain detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import logging
import sys
import numpy as np

from .exercise_plan_generator import EXERCISE_PHASES, ExercisePhase, ExercisePhaseSequence


def _to_python(val: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, (np.bool_, np.bool8)):
        return bool(val)
    elif isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    elif isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, dict):
        return {k: _to_python(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [_to_python(v) for v in val]
    return val


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

logger = _setup_logger("smartcare.physio.phase_tracker")


class PhaseMatchQuality(Enum):
    """How well the user matches the target phase."""
    NOT_MATCHING = "not_matching"
    APPROACHING = "approaching"  # Within 2x tolerance
    MATCHING = "matching"  # Within tolerance
    HOLDING = "holding"  # Matching and holding position


@dataclass
class PhaseProgress:
    """Progress through a single phase."""
    phase: ExercisePhase
    phase_index: int
    match_quality: PhaseMatchQuality = PhaseMatchQuality.NOT_MATCHING
    match_score: float = 0.0  # 0-100
    hold_start_time: Optional[datetime] = None
    hold_duration: float = 0.0
    completed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return _to_python({
            "phase_name": self.phase.name,
            "phase_index": int(self.phase_index),
            "description": self.phase.description,
            "visual_cue": self.phase.visual_cue,
            "target_angles": self.phase.target_angles,
            "tolerance": float(self.phase.tolerance),
            "match_quality": self.match_quality.value,
            "match_score": float(round(self.match_score, 1)),
            "hold_required": float(self.phase.hold_seconds),
            "hold_progress": float(round(self.hold_duration, 1)),
            "completed": bool(self.completed),
        })


@dataclass
class RepAdaptation:
    """Dynamic adaptation of reps based on session data."""
    original_reps: int
    adapted_reps: int
    adaptation_reason: str = ""
    pain_events: int = 0
    fatigue_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_reps": self.original_reps,
            "adapted_reps": self.adapted_reps,
            "adaptation_reason": self.adaptation_reason,
            "pain_events": self.pain_events,
            "fatigue_detected": self.fatigue_detected,
        }


class ExercisePhaseTracker:
    """
    Tracks user progress through exercise phases for progressive guidance.
    
    Features:
    - Detects when user matches current target phase
    - Tracks hold duration for phases requiring holds
    - Advances to next phase when current is completed
    - Counts reps when full phase sequence is completed
    - Adapts rep count based on pain/fatigue detection
    """
    
    # Pain level thresholds for rep adaptation
    PAIN_REDUCTION_THRESHOLDS = {
        "mild": 0.2,      # Reduce reps by 20% on mild pain
        "moderate": 0.4,  # Reduce reps by 40% on moderate pain
        "severe": 0.6,    # Reduce reps by 60% on severe pain
    }
    
    def __init__(self, exercise_id: str, target_reps: int = 10, target_sets: int = 3):
        self.exercise_id = exercise_id
        self.target_reps = target_reps
        self.target_sets = target_sets
        self.adapted_reps = target_reps
        
        # Get phase sequence for this exercise
        self.phase_sequence: Optional[ExercisePhaseSequence] = EXERCISE_PHASES.get(exercise_id)
        
        # Current state
        self.current_phase_index = 0
        self.current_rep = 0
        self.current_set = 1
        self.phases_completed_in_rep = 0
        
        # Phase progress tracking
        self._current_phase_progress: Optional[PhaseProgress] = None
        self._hold_start: Optional[datetime] = None
        
        # Phase transition stabilization
        self._matching_frame_count = 0          # Consecutive matching frames
        self._min_matching_frames = 3           # Need 3 consecutive matches (~0.6s at 5fps)
        self._phase_transition_cooldown = 0     # Frames to skip after transition
        self._transition_cooldown_frames = 2    # Skip 2 frames after advancing (~0.4s)
        
        # Pain/adaptation tracking
        self.pain_events: List[Dict[str, Any]] = []
        self.total_pain_score = 0.0
        self.fatigue_indicators = 0
        self.trembling_detected = False
        
        # Initialize first phase
        if self.phase_sequence and self.phase_sequence.phases:
            self._current_phase_progress = PhaseProgress(
                phase=self.phase_sequence.phases[0],
                phase_index=0,
            )
        
        logger.info(
            f"🎯 PhaseTracker initialized: {exercise_id} | "
            f"{len(self.phase_sequence.phases) if self.phase_sequence else 0} phases | "
            f"target={target_sets}x{target_reps}"
        )
    
    @property
    def has_phases(self) -> bool:
        """Check if this exercise has defined phases."""
        return self.phase_sequence is not None and len(self.phase_sequence.phases) > 0
    
    @property
    def current_phase(self) -> Optional[ExercisePhase]:
        """Get the current target phase."""
        if not self.has_phases:
            return None
        return self.phase_sequence.phases[self.current_phase_index]
    
    @property
    def next_phase(self) -> Optional[ExercisePhase]:
        """Preview the next phase (for animation transition)."""
        if not self.has_phases:
            return None
        next_idx = (self.current_phase_index + 1) % len(self.phase_sequence.phases)
        return self.phase_sequence.phases[next_idx]
    
    def update(
        self,
        current_angles: Dict[str, float],
        pain_level: Optional[str] = None,
        pain_confidence: float = 0.0,
        trembling: bool = False,
    ) -> Dict[str, Any]:
        """
        Update phase tracking with current pose angles.
        
        Args:
            current_angles: Current joint angles from pose analysis
            pain_level: Detected pain level (none, mild, moderate, severe)
            pain_confidence: Pain detection confidence
            trembling: Whether trembling/shaking was detected
        
        Returns:
            Dict with phase progress, rep counts, and guidance
        """
        # Track pain events
        if pain_level and pain_level != "none" and pain_confidence > 0.5:
            self._record_pain_event(pain_level, pain_confidence)
        
        # Track trembling as fatigue indicator
        if trembling:
            self.trembling_detected = True
            self.fatigue_indicators += 1
        
        # If no phases defined, return basic progress
        if not self.has_phases:
            return self._get_basic_response()
        
        # Calculate match quality for current phase
        match_result = self._calculate_phase_match(current_angles)
        
        # Update phase progress
        self._current_phase_progress.match_quality = match_result["quality"]
        self._current_phase_progress.match_score = match_result["score"]
        
        # Apply transition cooldown — skip advancement for a few frames after a phase change
        if self._phase_transition_cooldown > 0:
            self._phase_transition_cooldown -= 1
            return self._build_response(False)
        
        # Handle hold phases
        phase_completed = False
        if self.current_phase.hold_seconds > 0:
            phase_completed = self._update_hold_progress(match_result["quality"])
        else:
            # Non-hold phases: require multiple consecutive matching frames
            if match_result["quality"] == PhaseMatchQuality.MATCHING:
                self._matching_frame_count += 1
                if self._matching_frame_count >= self._min_matching_frames:
                    phase_completed = True
            else:
                self._matching_frame_count = 0
        
        # Advance to next phase if completed
        rep_completed = False
        if phase_completed:
            self._current_phase_progress.completed = True
            self._matching_frame_count = 0
            self._phase_transition_cooldown = self._transition_cooldown_frames
            rep_completed = self._advance_phase()
        
        # Build response
        return self._build_response(rep_completed)
    
    def _calculate_phase_match(self, current_angles: Dict[str, float]) -> Dict[str, Any]:
        """Calculate how well current pose matches target phase."""
        if not self.current_phase:
            return {"quality": PhaseMatchQuality.NOT_MATCHING, "score": 0.0, "details": {}}
        
        target_angles = self.current_phase.target_angles
        tolerance = self.current_phase.tolerance
        
        if not target_angles:
            # Phase has no specific angle requirements
            return {"quality": PhaseMatchQuality.MATCHING, "score": 100.0, "details": {}}
        
        total_error = 0.0
        details = {}
        matched_count = 0
        
        for angle_name, target_value in target_angles.items():
            current_value = current_angles.get(angle_name)
            if current_value is None:
                # Try alternate naming (e.g., "left_shoulder" -> "left_shoulder_angle")
                current_value = current_angles.get(f"{angle_name}_angle")
            if current_value is None:
                # Angle not available - skip but don't penalize
                continue
            
            error = abs(current_value - target_value)
            total_error += error
            
            is_match = bool(error <= tolerance)
            is_approaching = bool(error <= tolerance * 2)
            
            details[angle_name] = {
                "target": float(target_value),
                "current": float(current_value),
                "error": float(round(error, 1)),
                "matched": is_match,
            }
            
            if is_match:
                matched_count += 1
        
        # Calculate overall score and quality
        num_targets = len(target_angles)
        if num_targets == 0:
            return {"quality": PhaseMatchQuality.MATCHING, "score": 100.0, "details": details}
        
        avg_error = total_error / num_targets
        score = max(0, 100 - (avg_error * 2))  # 50-degree error = 0%
        
        # Determine quality
        if matched_count == num_targets:
            quality = PhaseMatchQuality.MATCHING
        elif matched_count >= num_targets * 0.5 or avg_error <= tolerance * 2:
            quality = PhaseMatchQuality.APPROACHING
        else:
            quality = PhaseMatchQuality.NOT_MATCHING
        
        return {
            "quality": quality,
            "score": score,
            "details": details,
            "matched": matched_count,
            "total": num_targets,
        }
    
    def _update_hold_progress(self, match_quality: PhaseMatchQuality) -> bool:
        """Update hold progress for phases requiring a hold. Returns True if hold completed."""
        if match_quality == PhaseMatchQuality.MATCHING:
            if self._hold_start is None:
                self._hold_start = datetime.now()
                self._current_phase_progress.match_quality = PhaseMatchQuality.HOLDING
            
            # Calculate hold duration
            hold_duration = (datetime.now() - self._hold_start).total_seconds()
            self._current_phase_progress.hold_duration = hold_duration
            
            # Check if hold is complete
            if hold_duration >= self.current_phase.hold_seconds:
                return True
        else:
            # User moved out of position - reset hold
            if self._hold_start is not None:
                logger.debug(f"  ⚠️ Hold interrupted - position lost")
            self._hold_start = None
            self._current_phase_progress.hold_duration = 0.0
        
        return False
    
    def _advance_phase(self) -> bool:
        """Advance to next phase. Returns True if a rep was completed."""
        self.phases_completed_in_rep += 1
        self.current_phase_index += 1
        
        # Check if we completed a full rep
        rep_completed = False
        if self.current_phase_index >= len(self.phase_sequence.phases):
            self.current_phase_index = 0
            self.current_rep += 1
            self.phases_completed_in_rep = 0
            rep_completed = True
            
            logger.info(f"  ✅ Rep {self.current_rep}/{self.adapted_reps} completed")
            
            # Check if set completed
            if self.current_rep >= self.adapted_reps:
                self.current_set += 1
                self.current_rep = 0
                logger.info(f"  🎉 Set {self.current_set - 1}/{self.target_sets} completed!")
        
        # Reset hold tracking
        self._hold_start = None
        
        # Create progress for new phase
        if self.has_phases:
            self._current_phase_progress = PhaseProgress(
                phase=self.phase_sequence.phases[self.current_phase_index],
                phase_index=self.current_phase_index,
            )
        
        return rep_completed
    
    def _record_pain_event(self, pain_level: str, confidence: float):
        """Record a pain detection event and adapt reps if needed."""
        self.pain_events.append({
            "level": pain_level,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "rep": self.current_rep,
            "set": self.current_set,
        })
        
        # Calculate cumulative pain score
        pain_weights = {"mild": 1, "moderate": 2, "severe": 4}
        self.total_pain_score += pain_weights.get(pain_level, 0) * confidence
        
        # Adapt reps based on pain
        self._adapt_for_pain()
    
    def _adapt_for_pain(self):
        """Dynamically adapt rep count based on accumulated pain."""
        if self.total_pain_score <= 0:
            return
        
        # Determine reduction level
        if self.total_pain_score >= 8:
            reduction = self.PAIN_REDUCTION_THRESHOLDS["severe"]
            reason = "Significant discomfort detected - reducing intensity"
        elif self.total_pain_score >= 4:
            reduction = self.PAIN_REDUCTION_THRESHOLDS["moderate"]
            reason = "Moderate discomfort detected - easing exercise load"
        elif self.total_pain_score >= 2:
            reduction = self.PAIN_REDUCTION_THRESHOLDS["mild"]
            reason = "Mild discomfort noted - slightly reducing reps"
        else:
            return
        
        # Apply reduction (but don't reduce below 3 reps)
        new_reps = max(3, int(self.target_reps * (1 - reduction)))
        
        if new_reps < self.adapted_reps:
            old_reps = self.adapted_reps
            self.adapted_reps = new_reps
            logger.warning(
                f"  ⚠️ Pain adaptation: {old_reps} → {new_reps} reps | "
                f"reason: {reason} | pain_score={self.total_pain_score:.1f}"
            )
    
    def _get_basic_response(self) -> Dict[str, Any]:
        """Get response for exercises without phase definitions."""
        return _to_python({
            "has_phases": False,
            "current_rep": int(self.current_rep),
            "current_set": int(self.current_set),
            "target_reps": int(self.adapted_reps),
            "target_sets": int(self.target_sets),
            "adaptation": self._get_adaptation_info(),
        })
    
    def _build_response(self, rep_completed: bool) -> Dict[str, Any]:
        """Build full response with phase progress."""
        response = {
            "has_phases": True,
            "current_phase": self._current_phase_progress.to_dict() if self._current_phase_progress else None,
            "next_phase": self.next_phase.to_dict() if self.next_phase else None,
            "phase_index": int(self.current_phase_index),
            "total_phases": int(len(self.phase_sequence.phases)),
            "phases_in_rep": int(self.phases_completed_in_rep),
            "rep_completed": bool(rep_completed),
            "current_rep": int(self.current_rep),
            "current_set": int(self.current_set),
            "target_reps": int(self.adapted_reps),
            "target_sets": int(self.target_sets),
            "set_completed": bool(self.current_set > self.target_sets),
            "exercise_completed": bool(self.current_set > self.target_sets),
            "adaptation": self._get_adaptation_info(),
        }
        
        # Ensure all values are JSON-serializable
        return _to_python(response)
    
    def _get_adaptation_info(self) -> Dict[str, Any]:
        """Get information about rep adaptation."""
        if self.adapted_reps == self.target_reps:
            return {
                "adapted": False,
                "original_reps": int(self.target_reps),
                "adapted_reps": int(self.adapted_reps),
            }
        
        return {
            "adapted": True,
            "original_reps": int(self.target_reps),
            "adapted_reps": int(self.adapted_reps),
            "reduction_percent": float(round((1 - self.adapted_reps / self.target_reps) * 100, 1)),
            "pain_events": int(len(self.pain_events)),
            "total_pain_score": float(round(self.total_pain_score, 1)),
            "fatigue_indicators": int(self.fatigue_indicators),
        }
    
    def get_reference_pose_data(self) -> Optional[Dict[str, Any]]:
        """
        Get reference pose data for skeleton overlay.
        
        Returns normalized landmark positions for the current target phase
        that can be drawn as a ghost skeleton.
        """
        if not self.has_phases or not self.current_phase:
            return None
        
        # Return phase info that frontend can use to generate reference skeleton
        return _to_python({
            "phase_name": self.current_phase.name,
            "visual_cue": self.current_phase.visual_cue,
            "target_angles": self.current_phase.target_angles,
            "tolerance": float(self.current_phase.tolerance),
            "hold_seconds": float(self.current_phase.hold_seconds),
            "match_quality": self._current_phase_progress.match_quality.value if self._current_phase_progress else "not_matching",
            "match_score": float(self._current_phase_progress.match_score) if self._current_phase_progress else 0.0,
        })


# ══════════════════════════════════════════════════════════════════════════════
# SESSION PHASE TRACKERS
# ══════════════════════════════════════════════════════════════════════════════

_session_phase_trackers: Dict[str, ExercisePhaseTracker] = {}


def get_or_create_phase_tracker(
    session_id: str,
    exercise_id: str,
    target_reps: int = 10,
    target_sets: int = 3
) -> ExercisePhaseTracker:
    """Get or create phase tracker for a session."""
    if session_id not in _session_phase_trackers:
        _session_phase_trackers[session_id] = ExercisePhaseTracker(
            exercise_id=exercise_id,
            target_reps=target_reps,
            target_sets=target_sets,
        )
    return _session_phase_trackers[session_id]


def get_phase_tracker(session_id: str) -> Optional[ExercisePhaseTracker]:
    """Get existing phase tracker for a session."""
    return _session_phase_trackers.get(session_id)


def remove_phase_tracker(session_id: str):
    """Remove phase tracker when session ends."""
    if session_id in _session_phase_trackers:
        del _session_phase_trackers[session_id]
        logger.debug(f"🗑️ Removed phase tracker for session: {session_id}")
