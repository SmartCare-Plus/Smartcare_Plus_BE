"""
SMARTCARE+ Physio Service - Pain Data Storage

Owner: Neelaka
Stores pain detection events for:
- Real-time feedback during exercise sessions
- Exercise plan adaptation based on pain history
- Weekly/monthly reports for caregivers
- Trend analysis over time
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json
import os
import logging
import sys


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

logger = _setup_logger("smartcare.physio.pain_storage")


class PainLevel(Enum):
    """Pain severity levels."""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class PainEvent:
    """A single pain detection event."""
    event_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    pain_level: PainLevel
    confidence: float
    source: str  # "face" or "pose" or "combined"
    
    # Context
    exercise_type: str
    rep_number: int
    set_number: int
    
    # Detailed indicators
    action_units: Dict[str, float] = field(default_factory=dict)
    details: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "pain_level": self.pain_level.value,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "exercise_type": self.exercise_type,
            "rep_number": self.rep_number,
            "set_number": self.set_number,
            "action_units": self.action_units,
            "details": self.details,
        }


@dataclass
class SessionPainSummary:
    """Pain summary for a single exercise session."""
    session_id: str
    user_id: str
    exercise_type: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Aggregated metrics
    total_pain_events: int = 0
    mild_events: int = 0
    moderate_events: int = 0
    severe_events: int = 0
    avg_pain_confidence: float = 0.0
    max_pain_confidence: float = 0.0
    pain_events: List[PainEvent] = field(default_factory=list)
    
    # Exercise impact
    exercise_stopped_due_to_pain: bool = False
    intensity_reduced: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "exercise_type": self.exercise_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_pain_events": self.total_pain_events,
            "mild_events": self.mild_events,
            "moderate_events": self.moderate_events,
            "severe_events": self.severe_events,
            "avg_pain_confidence": round(self.avg_pain_confidence, 3),
            "max_pain_confidence": round(self.max_pain_confidence, 3),
            "exercise_stopped_due_to_pain": self.exercise_stopped_due_to_pain,
            "intensity_reduced": self.intensity_reduced,
        }


@dataclass
class UserPainHistory:
    """Aggregated pain history for a user."""
    user_id: str
    period_start: datetime
    period_end: datetime
    
    # Per-exercise pain data
    exercise_pain_rates: Dict[str, float] = field(default_factory=dict)  # exercise -> avg pain rate
    exercise_session_counts: Dict[str, int] = field(default_factory=dict)  # exercise -> session count
    
    # Overall metrics
    total_sessions: int = 0
    sessions_with_pain: int = 0
    total_pain_events: int = 0
    avg_pain_intensity: float = 0.0
    
    # Trends
    pain_trend: str = "stable"  # "improving", "stable", "worsening"
    most_painful_exercise: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "exercise_pain_rates": self.exercise_pain_rates,
            "exercise_session_counts": self.exercise_session_counts,
            "total_sessions": self.total_sessions,
            "sessions_with_pain": self.sessions_with_pain,
            "total_pain_events": self.total_pain_events,
            "avg_pain_intensity": round(self.avg_pain_intensity, 3),
            "pain_trend": self.pain_trend,
            "most_painful_exercise": self.most_painful_exercise,
        }


class PainDataStore:
    """
    In-memory pain data storage with persistence support.
    
    Stores:
    - Individual pain events
    - Session pain summaries
    - User pain history
    """
    
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize pain data store.
        
        Args:
            persist_dir: Directory to persist data (None = memory only)
        """
        self.persist_dir = persist_dir
        
        # In-memory storage
        self._pain_events: Dict[str, List[PainEvent]] = {}  # session_id -> events
        self._session_summaries: Dict[str, SessionPainSummary] = {}  # session_id -> summary
        self._user_histories: Dict[str, List[SessionPainSummary]] = {}  # user_id -> session summaries
        
        self._event_counter = 0
        
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load_persisted_data()
    
    def record_pain_event(
        self,
        session_id: str,
        user_id: str,
        pain_level: PainLevel,
        confidence: float,
        source: str,
        exercise_type: str,
        rep_number: int,
        set_number: int,
        action_units: Dict[str, float] = None,
        details: List[str] = None
    ) -> PainEvent:
        """
        Record a new pain detection event.
        
        Returns the created PainEvent.
        """
        self._event_counter += 1
        event_id = f"pain_{session_id}_{self._event_counter}"
        
        event = PainEvent(
            event_id=event_id,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(),
            pain_level=pain_level,
            confidence=confidence,
            source=source,
            exercise_type=exercise_type,
            rep_number=rep_number,
            set_number=set_number,
            action_units=action_units or {},
            details=details or []
        )
        
        # Store event
        if session_id not in self._pain_events:
            self._pain_events[session_id] = []
        self._pain_events[session_id].append(event)
        
        logger.info(
            f"📝 Pain event recorded: {pain_level.value} ({confidence:.0%}) "
            f"session={session_id} exercise={exercise_type} rep={rep_number}"
        )
        
        return event
    
    def get_session_events(self, session_id: str) -> List[PainEvent]:
        """Get all pain events for a session."""
        return self._pain_events.get(session_id, [])
    
    def create_session_summary(
        self,
        session_id: str,
        user_id: str,
        exercise_type: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        exercise_stopped: bool = False,
        intensity_reduced: bool = False
    ) -> SessionPainSummary:
        """
        Create a summary of pain events for a completed session.
        """
        events = self.get_session_events(session_id)
        
        summary = SessionPainSummary(
            session_id=session_id,
            user_id=user_id,
            exercise_type=exercise_type,
            start_time=start_time,
            end_time=end_time or datetime.now(),
            pain_events=events,
            exercise_stopped_due_to_pain=exercise_stopped,
            intensity_reduced=intensity_reduced
        )
        
        # Calculate aggregates
        if events:
            summary.total_pain_events = len(events)
            summary.mild_events = sum(1 for e in events if e.pain_level == PainLevel.MILD)
            summary.moderate_events = sum(1 for e in events if e.pain_level == PainLevel.MODERATE)
            summary.severe_events = sum(1 for e in events if e.pain_level == PainLevel.SEVERE)
            summary.avg_pain_confidence = sum(e.confidence for e in events) / len(events)
            summary.max_pain_confidence = max(e.confidence for e in events)
        
        self._session_summaries[session_id] = summary
        
        # Add to user history
        if user_id not in self._user_histories:
            self._user_histories[user_id] = []
        self._user_histories[user_id].append(summary)
        
        logger.info(
            f"📊 Session pain summary: {session_id} | "
            f"events={summary.total_pain_events} (mild={summary.mild_events} "
            f"mod={summary.moderate_events} severe={summary.severe_events})"
        )
        
        return summary
    
    def get_user_pain_history(
        self,
        user_id: str,
        days: int = 7
    ) -> UserPainHistory:
        """
        Get aggregated pain history for a user over specified period.
        
        Args:
            user_id: User ID
            days: Number of days to look back
        
        Returns:
            UserPainHistory with aggregated metrics
        """
        period_end = datetime.now()
        period_start = period_end - timedelta(days=days)
        
        # Get sessions in period
        user_sessions = self._user_histories.get(user_id, [])
        period_sessions = [
            s for s in user_sessions 
            if s.start_time >= period_start
        ]
        
        history = UserPainHistory(
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
            total_sessions=len(period_sessions)
        )
        
        if not period_sessions:
            return history
        
        # Calculate per-exercise stats
        exercise_pain_totals: Dict[str, List[float]] = {}
        total_pain_events = 0
        sessions_with_pain = 0
        pain_confidences = []
        
        for session in period_sessions:
            exercise = session.exercise_type
            
            if exercise not in exercise_pain_totals:
                exercise_pain_totals[exercise] = []
                history.exercise_session_counts[exercise] = 0
            
            history.exercise_session_counts[exercise] += 1
            
            if session.total_pain_events > 0:
                sessions_with_pain += 1
                exercise_pain_totals[exercise].append(session.avg_pain_confidence)
                total_pain_events += session.total_pain_events
                pain_confidences.append(session.max_pain_confidence)
        
        # Calculate averages
        for exercise, confidences in exercise_pain_totals.items():
            if confidences:
                history.exercise_pain_rates[exercise] = sum(confidences) / len(confidences)
        
        history.sessions_with_pain = sessions_with_pain
        history.total_pain_events = total_pain_events
        
        if pain_confidences:
            history.avg_pain_intensity = sum(pain_confidences) / len(pain_confidences)
        
        # Find most painful exercise
        if history.exercise_pain_rates:
            history.most_painful_exercise = max(
                history.exercise_pain_rates.keys(),
                key=lambda x: history.exercise_pain_rates[x]
            )
        
        # Determine trend (simple: compare first half vs second half)
        if len(period_sessions) >= 4:
            mid = len(period_sessions) // 2
            first_half = period_sessions[:mid]
            second_half = period_sessions[mid:]
            
            first_avg = sum(s.avg_pain_confidence for s in first_half if s.total_pain_events > 0) / max(1, len([s for s in first_half if s.total_pain_events > 0]))
            second_avg = sum(s.avg_pain_confidence for s in second_half if s.total_pain_events > 0) / max(1, len([s for s in second_half if s.total_pain_events > 0]))
            
            if second_avg < first_avg * 0.8:
                history.pain_trend = "improving"
            elif second_avg > first_avg * 1.2:
                history.pain_trend = "worsening"
            else:
                history.pain_trend = "stable"
        
        return history
    
    def get_caregiver_report(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Generate a caregiver-friendly pain report.
        
        Returns dict suitable for frontend display or PDF generation.
        """
        history = self.get_user_pain_history(user_id, days)
        sessions = self._user_histories.get(user_id, [])
        
        # Get recent sessions
        period_start = datetime.now() - timedelta(days=days)
        recent_sessions = [
            s for s in sessions 
            if s.start_time >= period_start
        ]
        
        return {
            "user_id": user_id,
            "report_generated": datetime.now().isoformat(),
            "period_days": days,
            "summary": {
                "total_sessions": history.total_sessions,
                "sessions_with_pain": history.sessions_with_pain,
                "pain_rate": round(history.sessions_with_pain / max(1, history.total_sessions) * 100, 1),
                "avg_pain_intensity": round(history.avg_pain_intensity * 100, 1),
                "pain_trend": history.pain_trend,
                "most_problematic_exercise": history.most_painful_exercise,
            },
            "exercise_breakdown": [
                {
                    "exercise": exercise,
                    "sessions": history.exercise_session_counts.get(exercise, 0),
                    "avg_pain_level": round(history.exercise_pain_rates.get(exercise, 0) * 100, 1),
                }
                for exercise in history.exercise_session_counts.keys()
            ],
            "recommendations": self._generate_recommendations(history),
            "session_details": [s.to_dict() for s in recent_sessions[-10:]]  # Last 10 sessions
        }
    
    def _generate_recommendations(self, history: UserPainHistory) -> List[str]:
        """Generate recommendations based on pain history."""
        recommendations = []
        
        if history.pain_trend == "worsening":
            recommendations.append("⚠️ Pain levels increasing - consider consulting healthcare provider")
        
        if history.most_painful_exercise:
            pain_rate = history.exercise_pain_rates.get(history.most_painful_exercise, 0)
            if pain_rate > 0.4:
                recommendations.append(
                    f"Consider reducing intensity or modifying {history.most_painful_exercise.replace('_', ' ')}"
                )
        
        if history.sessions_with_pain > history.total_sessions * 0.5:
            recommendations.append("More than half of sessions have pain - review exercise difficulty levels")
        
        if not recommendations:
            recommendations.append("✓ Pain levels within normal range - continue current exercise plan")
        
        return recommendations
    
    def _load_persisted_data(self):
        """Load data from disk if persistence enabled."""
        if not self.persist_dir:
            return
        
        # For now, just log that we would load
        logger.debug(f"Pain data persistence directory: {self.persist_dir}")
    
    def _persist_data(self):
        """Save data to disk if persistence enabled."""
        if not self.persist_dir:
            return
        
        # Would save to JSON files here
        pass


# ── Global instance ──────────────────────────────────────────────────────────
_pain_store_instance: Optional[PainDataStore] = None


def get_pain_data_store() -> PainDataStore:
    """Get or create global pain data store instance."""
    global _pain_store_instance
    if _pain_store_instance is None:
        _pain_store_instance = PainDataStore()
    return _pain_store_instance
