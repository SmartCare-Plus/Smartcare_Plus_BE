"""
SMARTCARE+ Report Service

Generates patient reports by aggregating data from Firestore:
  - meal_logs       → nutrition stats
  - hydration_logs  → hydration stats
  - alerts          → safety / fall risk stats
  - exercise_plans  → physio progress stats
  - tasks           → caregiver task completion stats

Also builds an activity timeline from real events.
"""

import logging
from datetime import datetime, date as date_cls, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException

from core.database import get_db, is_mock_mode

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Helpers ───

def _date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _query_exercise_plans(patient_id: str, start_str: str) -> List[Dict[str, Any]]:
    """
    Query physio_exercise_plans from Firestore using .stream() (proven reliable).
    Falls back to ExercisePlanGenerator if direct query returns nothing.
    """
    if is_mock_mode():
        return []

    plans = []

    # Primary: direct Firestore query using .stream() (same as physio service)
    try:
        db = get_db()
        if db:
            docs = db.collection("physio_exercise_plans") \
                .where("user_id", "==", patient_id) \
                .stream()
            for doc in docs:
                data = doc.to_dict()
                plan_date = str(data.get("date", ""))
                if plan_date >= start_str:
                    plans.append(data)
            if plans:
                logger.info(f"[Reports] Found {len(plans)} exercise plans for {patient_id} (direct query)")
                return plans
    except Exception as e:
        logger.warning(f"[Reports] Direct exercise query failed: {e}")

    # Fallback: use ExercisePlanGenerator which has proven working code path
    try:
        from physio_service.models.exercise_plan_generator import get_exercise_plan_generator
        gen = get_exercise_plan_generator()
        # Calculate days from start_str to today
        today = date_cls.today()
        start_date = date_cls.fromisoformat(start_str)
        days_diff = (today - start_date).days + 1
        loaded_plans = gen.get_user_plans(patient_id, days=days_diff)
        plans = []
        for p in loaded_plans:
            d = {
                "date": p.date.isoformat(),
                "exercises": [e.to_dict() for e in p.exercises],
                "completed_exercises": p.completed_exercises,
                "completed": p.completed,
                "completed_at": p.completed_at.isoformat() if p.completed_at else None,
            }
            plans.append(d)
        if plans:
            logger.info(f"[Reports] Found {len(plans)} exercise plans for {patient_id} (via ExercisePlanGenerator)")
    except Exception as e:
        logger.warning(f"[Reports] ExercisePlanGenerator fallback failed: {e}")

    return plans


def _parse_date(s: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ─── Patient Report ───

@router.get("/patient/{patient_id}")
async def get_patient_report(patient_id: str, days: int = 7):
    """
    Generate a comprehensive patient report aggregating data from
    nutrition, hydration, exercise, and alert collections.
    """
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    start_str = _date_str(start)
    period_label = f"Last {days} days"

    nutrition = _get_nutrition_stats(patient_id, start_str, days)
    hydration = _get_hydration_stats(patient_id, start_str, days)
    physio = _get_physio_stats(patient_id, start_str, days)
    safety = _get_safety_stats(patient_id, start, days)

    return {
        "patient_id": patient_id,
        "period": period_label,
        "days": days,
        "generated_at": now.isoformat(),
        "nutrition": nutrition,
        "hydration": hydration,
        "physio": physio,
        "safety": safety,
    }


# ─── Activity Log ───

@router.get("/activity-log/{patient_id}")
async def get_activity_log(patient_id: str, days: int = 1):
    """
    Build a chronological activity timeline from real Firestore events
    (meals, hydration, exercises, alerts) for the given patient.
    """
    now = datetime.utcnow()
    start = now - timedelta(days=days)
    start_str = _date_str(start)
    activities = []

    if is_mock_mode():
        # Return demo activities
        today_str = _date_str(now)
        activities = _mock_activities(today_str)
    else:
        db = get_db()

        # Meal events (query by user_id only, filter date in Python to avoid composite index)
        try:
            meals = db.collection("meal_logs") \
                .where("user_id", "==", patient_id) \
                .get()
            for doc in meals:
                m = doc.to_dict()
                doc_date = m.get("date", "")
                if doc_date < start_str:
                    continue
                meal_type = (m.get("meal_type") or "Meal").capitalize()
                foods = m.get("foods") or []
                food_names = ", ".join(f.get("name", "") for f in foods[:3]) if foods else ""
                cals = m.get("totals", {}).get("calories", 0)
                logged_at = m.get("logged_at") or m.get("date", "")
                activities.append({
                    "time": _extract_time(logged_at),
                    "event": f"{meal_type} logged",
                    "detail": f"{food_names} • {int(cals)} cal" if food_names else f"{int(cals)} cal",
                    "category": "nutrition",
                    "icon": "restaurant",
                    "date": m.get("date", ""),
                    "timestamp": logged_at,
                })
        except Exception as e:
            logger.warning(f"Error fetching meal logs for activity: {e}")

        # Hydration events (query by user_id only, filter date in Python)
        try:
            hydrations = db.collection("hydration_logs") \
                .where("user_id", "==", patient_id) \
                .get()
            for doc in hydrations:
                h = doc.to_dict()
                h_date = h.get("date", "")
                if h_date < start_str:
                    continue
                amount = h.get("amount_ml", 0)
                bev = (h.get("beverage_type") or "Water").capitalize()
                logged_at = h.get("logged_at") or h.get("date", "")
                activities.append({
                    "time": _extract_time(logged_at),
                    "event": f"Drank {bev}",
                    "detail": f"{int(amount)} ml",
                    "category": "hydration",
                    "icon": "water_drop",
                    "date": h.get("date", ""),
                    "timestamp": logged_at,
                })
        except Exception as e:
            logger.warning(f"Error fetching hydration logs for activity: {e}")

        # Exercise events — use shared reliable query
        try:
            exercise_plans = _query_exercise_plans(patient_id, start_str)
            for p in exercise_plans:
                plan_date = str(p.get("date", ""))
                completed = p.get("completed", False)
                n_exercises = len(p.get("exercises", []))
                n_completed = len(p.get("completed_exercises", []))
                status = "Completed" if completed else f"{n_completed}/{n_exercises} done"
                activities.append({
                    "time": _extract_time(p.get("completed_at") or plan_date),
                    "event": "Exercise plan" if not completed else "Exercise plan completed",
                    "detail": f"{n_exercises} exercises • {status}",
                    "category": "exercise",
                    "icon": "fitness_center",
                    "date": plan_date,
                    "timestamp": p.get("completed_at") or plan_date,
                })
        except Exception as e:
            logger.warning(f"Error fetching exercise plans for activity: {e}")

        # Alert events (query by elderly_id only, filter date in Python)
        try:
            alerts_ref = db.collection("alerts") \
                .where("elderly_id", "==", patient_id) \
                .limit(200)
            alerts = alerts_ref.get()
            for doc in alerts:
                a = doc.to_dict()
                created = a.get("created_at", "")
                # Filter by date
                parsed = _parse_date(str(created)[:19]) if created else None
                if parsed and parsed < start:
                    continue
                alert_type = (a.get("type") or "alert").replace("_", " ").capitalize()
                severity = a.get("severity", "")
                title = a.get("title") or alert_type
                activities.append({
                    "time": _extract_time(created),
                    "event": title,
                    "detail": f"{alert_type} • {severity}" if severity else alert_type,
                    "category": "alert",
                    "icon": "warning",
                    "date": str(created)[:10] if created else "",
                    "timestamp": str(created) if created else "",
                })
        except Exception as e:
            logger.warning(f"Error fetching alerts for activity: {e}")

    # Sort by timestamp descending (most recent first)
    activities.sort(key=lambda a: a.get("timestamp", ""), reverse=True)

    # Compute summary
    meal_count = sum(1 for a in activities if a["category"] == "nutrition")
    alert_count = sum(1 for a in activities if a["category"] == "alert")
    hydration_count = sum(1 for a in activities if a["category"] == "hydration")

    # Count actual individual exercises from plan data already fetched
    exercise_completed = 0
    exercise_total = 0
    if not is_mock_mode():
        try:
            exercise_plans = _query_exercise_plans(patient_id, start_str)
            for p in exercise_plans:
                exercise_completed += len(p.get("completed_exercises", []))
                exercise_total += len(p.get("exercises", []))
        except Exception as e:
            logger.warning(f"Error counting exercises for summary: {e}")

    return {
        "patient_id": patient_id,
        "days": days,
        "total_events": len(activities),
        "summary": {
            "meals": meal_count,
            "exercises": exercise_completed,
            "exercises_total": exercise_total,
            "alerts": alert_count,
            "hydration_entries": hydration_count,
        },
        "activities": activities,
    }


# ─── Internal Stats Builders ───

def _get_nutrition_stats(patient_id: str, start_str: str, days: int) -> dict:
    """Aggregate nutrition data from meal_logs collection."""
    if is_mock_mode():
        return {
            "meals_logged": 0,
            "avg_calories": 0,
            "avg_protein": 0,
            "avg_carbs": 0,
            "avg_fat": 0,
            "note": "Connect Firebase for real nutrition data",
        }

    db = get_db()
    try:
        docs = db.collection("meal_logs") \
            .where("user_id", "==", patient_id) \
            .get()

        meals = [d.to_dict() for d in docs if d.to_dict().get("date", "") >= start_str]
        count = len(meals)
        if count == 0:
            return {
                "meals_logged": 0,
                "avg_calories": 0,
                "avg_protein": 0,
                "avg_carbs": 0,
                "avg_fat": 0,
                "note": "No meals logged in this period",
            }

        total_cal = sum(_safe_float(m.get("totals", {}).get("calories")) for m in meals)
        total_protein = sum(_safe_float(m.get("totals", {}).get("protein")) for m in meals)
        total_carbs = sum(_safe_float(m.get("totals", {}).get("carbs")) for m in meals)
        total_fat = sum(_safe_float(m.get("totals", {}).get("fat")) for m in meals)

        # Group by date to get per-day averages
        dates = set(m.get("date", "") for m in meals)
        num_days = max(len(dates), 1)

        return {
            "meals_logged": count,
            "avg_calories": round(total_cal / num_days),
            "avg_protein": round(total_protein / num_days, 1),
            "avg_carbs": round(total_carbs / num_days, 1),
            "avg_fat": round(total_fat / num_days, 1),
            "total_calories": round(total_cal),
        }
    except Exception as e:
        logger.error(f"Nutrition stats error: {e}")
        return {"meals_logged": 0, "avg_calories": 0, "error": str(e)}


def _get_hydration_stats(patient_id: str, start_str: str, days: int) -> dict:
    """Aggregate hydration data from hydration_logs collection."""
    if is_mock_mode():
        return {"total_ml": 0, "daily_avg_ml": 0, "entries": 0, "note": "Connect Firebase for real data"}

    db = get_db()
    try:
        docs = db.collection("hydration_logs") \
            .where("user_id", "==", patient_id) \
            .get()

        logs = [d.to_dict() for d in docs if d.to_dict().get("date", "") >= start_str]
        count = len(logs)
        if count == 0:
            return {"total_ml": 0, "daily_avg_ml": 0, "entries": 0, "note": "No hydration logged"}

        total_ml = sum(_safe_float(l.get("amount_ml")) for l in logs)
        dates = set(l.get("date", "") for l in logs)
        num_days = max(len(dates), 1)

        return {
            "total_ml": round(total_ml),
            "daily_avg_ml": round(total_ml / num_days),
            "entries": count,
        }
    except Exception as e:
        logger.error(f"Hydration stats error: {e}")
        return {"total_ml": 0, "daily_avg_ml": 0, "entries": 0, "error": str(e)}


def _get_physio_stats(patient_id: str, start_str: str, days: int) -> dict:
    """Aggregate physio data from physio_exercise_plans collection."""
    if is_mock_mode():
        return {
            "total_plans": 0,
            "completed_plans": 0,
            "completion_rate": "0%",
            "total_exercises": 0,
            "completed_exercises": 0,
            "note": "Connect Firebase for real data",
        }

    try:
        plans = _query_exercise_plans(patient_id, start_str)

        total = len(plans)
        logger.info(f"[Physio Stats] {total} plans in period (since {start_str})")
        if total == 0:
            return {
                "total_plans": 0,
                "completed_plans": 0,
                "completion_rate": "0%",
                "total_exercises": 0,
                "completed_exercises": 0,
                "note": "No exercise plans in this period",
            }

        completed = sum(1 for p in plans if p.get("completed"))
        total_exercises = sum(len(p.get("exercises", [])) for p in plans)
        completed_exercises = sum(len(p.get("completed_exercises", [])) for p in plans)
        rate = round((completed / total) * 100) if total > 0 else 0

        return {
            "total_plans": total,
            "completed_plans": completed,
            "completion_rate": f"{rate}%",
            "total_exercises": total_exercises,
            "completed_exercises": completed_exercises,
        }
    except Exception as e:
        logger.error(f"Physio stats error: {e}")
        return {"total_plans": 0, "completed_plans": 0, "completion_rate": "0%", "error": str(e)}


def _get_safety_stats(patient_id: str, start: datetime, days: int) -> dict:
    """Aggregate safety data from alerts collection."""
    if is_mock_mode():
        return {
            "total_alerts": 0, "fall_alerts": 0, "gait_alerts": 0,
            "inactivity_alerts": 0, "sos_alerts": 0, "meal_skip_alerts": 0,
            "unresolved": 0, "note": "Connect Firebase for real data",
        }

    db = get_db()
    try:
        docs = db.collection("alerts") \
            .where("elderly_id", "==", patient_id) \
            .limit(500) \
            .get()

        alerts = []
        for doc in docs:
            a = doc.to_dict()
            created = a.get("created_at", "")
            parsed = _parse_date(str(created)[:19]) if created else None
            if parsed and parsed >= start:
                alerts.append(a)

        total = len(alerts)
        fall = sum(1 for a in alerts if a.get("type") == "fall")
        gait = sum(1 for a in alerts if a.get("type") == "gait")
        inactivity = sum(1 for a in alerts if a.get("type") == "inactivity")
        sos = sum(1 for a in alerts if a.get("type") == "sos")
        meal_skip = sum(1 for a in alerts if a.get("type") == "meal_skipped")
        unresolved = sum(1 for a in alerts if not a.get("acknowledged", False))

        return {
            "total_alerts": total,
            "fall_alerts": fall,
            "gait_alerts": gait,
            "inactivity_alerts": inactivity,
            "sos_alerts": sos,
            "meal_skip_alerts": meal_skip,
            "unresolved": unresolved,
        }
    except Exception as e:
        logger.error(f"Safety stats error: {e}")
        return {"total_alerts": 0, "error": str(e)}


# ─── Helpers ───

def _extract_time(timestamp) -> str:
    """Extract HH:MM AM/PM from a timestamp string or datetime."""
    if not timestamp:
        return ""
    ts = str(timestamp)
    parsed = _parse_date(ts[:19])
    if parsed:
        return parsed.strftime("%I:%M %p").lstrip("0")
    # Try to find time pattern in string
    if "T" in ts and len(ts) > 11:
        time_part = ts[11:16]
        try:
            h, m = map(int, time_part.split(":"))
            ampm = "AM" if h < 12 else "PM"
            h12 = h % 12 or 12
            return f"{h12}:{m:02d} {ampm}"
        except (ValueError, IndexError):
            pass
    return ""


def _mock_activities(date_str: str) -> list:
    """Return mock activities for demo/testing."""
    return [
        {"time": "8:30 AM", "event": "Breakfast logged", "detail": "Toast, eggs • 420 cal", "category": "nutrition", "icon": "restaurant", "date": date_str, "timestamp": f"{date_str}T08:30:00"},
        {"time": "9:00 AM", "event": "Drank Water", "detail": "250 ml", "category": "hydration", "icon": "water_drop", "date": date_str, "timestamp": f"{date_str}T09:00:00"},
        {"time": "10:00 AM", "event": "Exercise plan started", "detail": "3 exercises • 1/3 done", "category": "exercise", "icon": "fitness_center", "date": date_str, "timestamp": f"{date_str}T10:00:00"},
        {"time": "11:30 AM", "event": "Drank Tea", "detail": "200 ml", "category": "hydration", "icon": "water_drop", "date": date_str, "timestamp": f"{date_str}T11:30:00"},
        {"time": "12:30 PM", "event": "Lunch logged", "detail": "Rice, chicken curry • 650 cal", "category": "nutrition", "icon": "restaurant", "date": date_str, "timestamp": f"{date_str}T12:30:00"},
        {"time": "2:00 PM", "event": "Drank Water", "detail": "300 ml", "category": "hydration", "icon": "water_drop", "date": date_str, "timestamp": f"{date_str}T14:00:00"},
        {"time": "4:00 PM", "event": "Snack logged", "detail": "Biscuits, tea • 180 cal", "category": "nutrition", "icon": "restaurant", "date": date_str, "timestamp": f"{date_str}T16:00:00"},
        {"time": "6:30 PM", "event": "Exercise plan completed", "detail": "3 exercises • Completed", "category": "exercise", "icon": "fitness_center", "date": date_str, "timestamp": f"{date_str}T18:30:00"},
        {"time": "7:30 PM", "event": "Dinner logged", "detail": "Soup, bread • 480 cal", "category": "nutrition", "icon": "restaurant", "date": date_str, "timestamp": f"{date_str}T19:30:00"},
    ]