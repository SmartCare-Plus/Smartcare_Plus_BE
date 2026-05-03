"""Diagnostic script to find why exercises show as 0 in reports."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.database import init_firebase, get_db, is_mock_mode

init_firebase()
db = get_db()

patient_id = "FotKtoYUQ1gZVeddXE0hU1ZMotk1"

print(f"=== DIAGNOSTIC: Exercise data for {patient_id} ===")
print(f"Mock mode: {is_mock_mode()}")
print(f"DB type: {type(db)}")
print()

# 1. Direct Firestore query - same as reports.py
print("--- physio_exercise_plans collection ---")
try:
    docs = db.collection("physio_exercise_plans").where("user_id", "==", patient_id).get()
    print(f"Total docs found: {len(docs)}")
    for d in docs:
        data = d.to_dict()
        date_val = data.get("date")
        exercises = data.get("exercises", [])
        completed = data.get("completed_exercises", [])
        print(f"  Doc ID: {d.id}")
        print(f"    date value: {repr(date_val)}")
        print(f"    date type: {type(date_val).__name__}")
        print(f"    exercises count: {len(exercises)}")
        print(f"    completed_exercises count: {len(completed)}")
        print(f"    completed_exercises: {completed}")
        print(f"    completed flag: {data.get('completed')}")
        print()
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# 2. Check what the date comparison looks like
from datetime import datetime, timedelta
now = datetime.utcnow()
start_7 = now - timedelta(days=7)
start_str_7 = start_7.strftime("%Y-%m-%d")
start_1 = now - timedelta(days=1)
start_str_1 = start_1.strftime("%Y-%m-%d")

print(f"--- Date comparison ---")
print(f"Now (UTC): {now}")
print(f"start_str (7 days): {start_str_7}")
print(f"start_str (1 day): {start_str_1}")

try:
    docs = db.collection("physio_exercise_plans").where("user_id", "==", patient_id).get()
    for d in docs:
        data = d.to_dict()
        date_val = data.get("date", "")
        date_str = str(date_val)
        print(f"  Doc {d.id}: date={repr(date_val)}, type={type(date_val).__name__}")
        print(f"    str(date)={date_str}")
        print(f"    date >= start_str_7 ? {date_str} >= {start_str_7} = {date_str >= start_str_7}")
        print(f"    date >= start_str_1 ? {date_str} >= {start_str_1} = {date_str >= start_str_1}")
        # Check if it's a datetime object from Firestore
        if hasattr(date_val, 'isoformat'):
            print(f"    isoformat: {date_val.isoformat()}")
        if hasattr(date_val, 'strftime'):
            print(f"    strftime: {date_val.strftime('%Y-%m-%d')}")
except Exception as e:
    print(f"ERROR: {e}")

# 3. Also check how physio service stores/retrieves plans
print()
print("--- Checking via physio service's get_today_plan ---")
try:
    from physio_service.models.exercise_plan_generator import ExercisePlanGenerator
    gen = ExercisePlanGenerator()
    plan = gen.get_today_plan(patient_id)
    if plan:
        print(f"  plan_id: {plan.plan_id}")
        print(f"  date: {plan.date}, type: {type(plan.date).__name__}")
        print(f"  exercise_count: {plan.exercise_count}")
        print(f"  completed_exercise_count: {plan.completed_exercise_count}")
        print(f"  completed_exercises: {plan.completed_exercises}")
    else:
        print("  No plan returned")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
