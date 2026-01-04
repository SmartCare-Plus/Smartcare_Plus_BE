"""
SMARTCARE+ Physio Service - Exercise Session Handler

Owner: Neelaka
Manages exercise sessions with real-time feedback, rep counting, and scoring.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import time
import uuid

from .pose_analyzer import (
    PoseAnalyzer, 
    PoseResult, 
    ExerciseType, 
    FormQuality,
    FormAssessment,
    get_pose_analyzer
)


class SessionState(Enum):
    """Exercise session states."""
    IDLE = "idle"
    WARMUP = "warmup"
    ACTIVE = "active"
    REST = "rest"
    COOLDOWN = "cooldown"
    COMPLETED = "completed"
    PAUSED = "paused"


@dataclass
class RepRecord:
    """Record of a single repetition."""
    rep_number: int
    timestamp: float
    form_score: float
    form_quality: FormQuality
    duration_seconds: float
    feedback: List[str] = field(default_factory=list)


@dataclass
class SetRecord:
    """Record of an exercise set."""
    set_number: int
    exercise_type: ExerciseType
    target_reps: int
    completed_reps: int
    reps: List[RepRecord] = field(default_factory=list)
    avg_form_score: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0.0
    
    def calculate_avg_score(self):
        if self.reps:
            self.avg_form_score = sum(r.form_score for r in self.reps) / len(self.reps)


@dataclass  
class ExerciseSession:
    """Complete exercise session data."""
    session_id: str
    user_id: str
    exercise_type: ExerciseType
    state: SessionState = SessionState.IDLE
    
    # Configuration
    target_sets: int = 3
    target_reps_per_set: int = 10
    rest_duration_seconds: int = 30
    
    # Progress tracking
    current_set: int = 1
    current_rep: int = 0
    sets: List[SetRecord] = field(default_factory=list)
    
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    last_rep_time: float = 0.0
    
    # Metrics
    total_reps: int = 0
    avg_form_score: float = 0.0
    pain_detected_count: int = 0
    
    # Real-time feedback
    current_feedback: List[str] = field(default_factory=list)
    intensity_recommendation: str = "Continue at current intensity"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "exercise_type": self.exercise_type.value,
            "state": self.state.value,
            "target_sets": self.target_sets,
            "target_reps_per_set": self.target_reps_per_set,
            "current_set": self.current_set,
            "current_rep": self.current_rep,
            "total_reps": self.total_reps,
            "avg_form_score": round(self.avg_form_score, 1),
            "pain_detected_count": self.pain_detected_count,
            "intensity_recommendation": self.intensity_recommendation,
            "current_feedback": self.current_feedback,
            "duration_seconds": (self.end_time or time.time()) - (self.start_time or time.time()),
            "sets": [
                {
                    "set_number": s.set_number,
                    "completed_reps": s.completed_reps,
                    "target_reps": s.target_reps,
                    "avg_form_score": round(s.avg_form_score, 1),
                    "duration_seconds": round(s.duration_seconds, 1)
                }
                for s in self.sets
            ]
        }


class ExerciseSessionHandler:
    """
    Manages exercise sessions with real-time pose analysis.
    
    Features:
    - Multi-set exercise tracking
    - Real-time form feedback
    - Rep counting with form scoring
    - Pain/discomfort detection
    - Session summary generation
    """
    
    def __init__(self, pose_analyzer: Optional[PoseAnalyzer] = None):
        """
        Initialize session handler.
        
        Args:
            pose_analyzer: PoseAnalyzer instance (uses global if None)
        """
        self.pose_analyzer = pose_analyzer or get_pose_analyzer()
        self.active_sessions: Dict[str, ExerciseSession] = {}
    
    def create_session(
        self,
        user_id: str,
        exercise_type: ExerciseType,
        target_sets: int = 3,
        target_reps: int = 10,
        rest_duration: int = 30
    ) -> ExerciseSession:
        """
        Create a new exercise session.
        
        Args:
            user_id: User ID
            exercise_type: Type of exercise
            target_sets: Number of sets
            target_reps: Reps per set
            rest_duration: Rest time between sets (seconds)
        
        Returns:
            New ExerciseSession
        """
        session_id = str(uuid.uuid4())[:8]
        
        session = ExerciseSession(
            session_id=session_id,
            user_id=user_id,
            exercise_type=exercise_type,
            target_sets=target_sets,
            target_reps_per_set=target_reps,
            rest_duration_seconds=rest_duration
        )
        
        self.active_sessions[session_id] = session
        
        # Initialize rep counter for this exercise
        self.pose_analyzer.init_rep_counter(exercise_type)
        
        return session
    
    def start_session(self, session_id: str) -> Dict[str, Any]:
        """
        Start an exercise session.
        
        Returns status dict.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found", "session_id": session_id}
        
        session.state = SessionState.ACTIVE
        session.start_time = time.time()
        
        # Create first set
        first_set = SetRecord(
            set_number=1,
            exercise_type=session.exercise_type,
            target_reps=session.target_reps_per_set,
            completed_reps=0,
            start_time=time.time()
        )
        session.sets.append(first_set)
        
        return {
            "status": "started",
            "session_id": session_id,
            "exercise": session.exercise_type.value,
            "target_sets": session.target_sets,
            "target_reps": session.target_reps_per_set
        }
    
    def process_frame(self, session_id: str, pose: PoseResult) -> Dict[str, Any]:
        """
        Process a video frame during exercise.
        
        Args:
            session_id: Active session ID
            pose: Pose result from frame
        
        Returns:
            Real-time feedback dict
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        if session.state != SessionState.ACTIVE:
            return {"status": session.state.value, "message": "Session not active"}
        
        # Assess form
        form_assessment = self.pose_analyzer.assess_form(pose, session.exercise_type)
        
        # Count reps
        rep_count, rep_completed = self.pose_analyzer.count_rep(pose, session.exercise_type)
        
        # Check for pain indicators
        pain_indicators = self.pose_analyzer.detect_pain_indicators(pose)
        
        response = {
            "session_id": session_id,
            "state": session.state.value,
            "current_set": session.current_set,
            "current_rep": rep_count,
            "target_reps": session.target_reps_per_set,
            "form_score": form_assessment.score,
            "form_quality": form_assessment.quality.value,
            "feedback": form_assessment.feedback,
            "rep_completed": rep_completed,
            "pain_indicators": {
                "detected": pain_indicators.get("confidence", 0) > 0.3,
                "confidence": pain_indicators.get("confidence", 0),
                "details": pain_indicators.get("details", [])
            }
        }
        
        # Handle rep completion
        if rep_completed:
            self._record_rep(session, form_assessment)
            
            # Check if set is complete
            current_set = session.sets[-1] if session.sets else None
            if current_set and current_set.completed_reps >= session.target_reps_per_set:
                response["set_completed"] = True
                response["message"] = f"Set {session.current_set} complete!"
                
                # Check if all sets complete
                if session.current_set >= session.target_sets:
                    self.complete_session(session_id)
                    response["session_completed"] = True
                else:
                    # Start rest period
                    session.state = SessionState.REST
                    response["rest_duration"] = session.rest_duration_seconds
        
        # Update pain tracking
        if pain_indicators.get("confidence", 0) > 0.4:
            session.pain_detected_count += 1
            session.intensity_recommendation = self.pose_analyzer.get_intensity_recommendation(pain_indicators)
            response["intensity_recommendation"] = session.intensity_recommendation
        
        session.current_feedback = form_assessment.feedback
        
        return response
    
    def _record_rep(self, session: ExerciseSession, assessment: FormAssessment):
        """Record a completed repetition."""
        current_time = time.time()
        duration = current_time - session.last_rep_time if session.last_rep_time > 0 else 2.0
        
        rep = RepRecord(
            rep_number=session.current_rep + 1,
            timestamp=current_time,
            form_score=assessment.score,
            form_quality=assessment.quality,
            duration_seconds=duration,
            feedback=assessment.feedback
        )
        
        if session.sets:
            current_set = session.sets[-1]
            current_set.reps.append(rep)
            current_set.completed_reps += 1
            current_set.calculate_avg_score()
        
        session.current_rep += 1
        session.total_reps += 1
        session.last_rep_time = current_time
        
        # Update average form score
        all_scores = []
        for s in session.sets:
            all_scores.extend([r.form_score for r in s.reps])
        if all_scores:
            session.avg_form_score = sum(all_scores) / len(all_scores)
    
    def start_next_set(self, session_id: str) -> Dict[str, Any]:
        """
        Start the next set after rest period.
        
        Returns status dict.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        if session.current_set >= session.target_sets:
            return {"error": "All sets completed"}
        
        # End current set
        if session.sets:
            session.sets[-1].end_time = time.time()
        
        # Start new set
        session.current_set += 1
        session.current_rep = 0
        session.state = SessionState.ACTIVE
        
        # Reset rep counter
        self.pose_analyzer.reset_rep_counter(session.exercise_type)
        
        new_set = SetRecord(
            set_number=session.current_set,
            exercise_type=session.exercise_type,
            target_reps=session.target_reps_per_set,
            completed_reps=0,
            start_time=time.time()
        )
        session.sets.append(new_set)
        
        return {
            "status": "set_started",
            "session_id": session_id,
            "current_set": session.current_set,
            "total_sets": session.target_sets
        }
    
    def pause_session(self, session_id: str) -> Dict[str, Any]:
        """Pause an active session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        session.state = SessionState.PAUSED
        return {"status": "paused", "session_id": session_id}
    
    def resume_session(self, session_id: str) -> Dict[str, Any]:
        """Resume a paused session."""
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        if session.state == SessionState.PAUSED:
            session.state = SessionState.ACTIVE
            return {"status": "resumed", "session_id": session_id}
        
        return {"error": "Session not paused"}
    
    def complete_session(self, session_id: str) -> Dict[str, Any]:
        """
        Complete an exercise session and generate summary.
        
        Returns complete session summary.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        session.state = SessionState.COMPLETED
        session.end_time = time.time()
        
        # Finalize last set
        if session.sets:
            session.sets[-1].end_time = time.time()
        
        # Generate summary
        summary = self._generate_summary(session)
        
        # Clean up
        self.pose_analyzer.reset_rep_counter(session.exercise_type)
        
        return summary
    
    def _generate_summary(self, session: ExerciseSession) -> Dict[str, Any]:
        """Generate session summary."""
        duration = (session.end_time or time.time()) - (session.start_time or time.time())
        
        # Calculate achievements
        target_total = session.target_sets * session.target_reps_per_set
        completion_rate = (session.total_reps / target_total * 100) if target_total > 0 else 0
        
        # Determine performance rating
        if completion_rate >= 100 and session.avg_form_score >= 85:
            performance = "excellent"
            message = "Outstanding performance! 🌟"
        elif completion_rate >= 80 and session.avg_form_score >= 70:
            performance = "good"
            message = "Great job! Keep it up! 👍"
        elif completion_rate >= 60:
            performance = "fair"
            message = "Good effort! Room for improvement."
        else:
            performance = "needs_improvement"
            message = "Keep practicing! You'll get better."
        
        return {
            "status": "completed",
            "session_id": session.session_id,
            "user_id": session.user_id,
            "exercise": session.exercise_type.value,
            "summary": {
                "total_reps": session.total_reps,
                "target_reps": target_total,
                "completion_rate": round(completion_rate, 1),
                "sets_completed": len([s for s in session.sets if s.completed_reps > 0]),
                "target_sets": session.target_sets,
                "avg_form_score": round(session.avg_form_score, 1),
                "duration_seconds": round(duration, 1),
                "pain_incidents": session.pain_detected_count,
                "performance_rating": performance,
                "message": message
            },
            "sets": [
                {
                    "set_number": s.set_number,
                    "reps": s.completed_reps,
                    "avg_score": round(s.avg_form_score, 1),
                    "duration": round(s.duration_seconds, 1)
                }
                for s in session.sets
            ],
            "recommendations": self._get_recommendations(session),
            "completed_at": datetime.now().isoformat()
        }
    
    def _get_recommendations(self, session: ExerciseSession) -> List[str]:
        """Generate personalized recommendations based on session performance."""
        recommendations = []
        
        if session.avg_form_score < 70:
            recommendations.append("Focus on maintaining proper form over completing more reps")
        
        if session.pain_detected_count > 2:
            recommendations.append("Consider consulting with a physiotherapist about the discomfort")
            recommendations.append("Try lower intensity exercises in your next session")
        
        completion = session.total_reps / (session.target_sets * session.target_reps_per_set) * 100
        if completion < 80:
            recommendations.append("Try reducing the number of sets or reps in your next session")
        elif completion >= 100 and session.avg_form_score >= 85:
            recommendations.append("You're ready to increase difficulty! Try more reps or add resistance")
        
        if not recommendations:
            recommendations.append("Great progress! Maintain this consistency")
        
        return recommendations
    
    def get_session(self, session_id: str) -> Optional[ExerciseSession]:
        """Get session by ID."""
        return self.active_sessions.get(session_id)
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current session status."""
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return session.to_dict()
    
    def cleanup_session(self, session_id: str):
        """Remove session from active sessions."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_handler_instance: Optional[ExerciseSessionHandler] = None

def get_session_handler() -> ExerciseSessionHandler:
    """Get or create the global session handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = ExerciseSessionHandler()
    return _handler_instance
