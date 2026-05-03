"""Quick script to check exercise plans in Firestore."""
import sys
sys.path.insert(0, ".")
from core.database import get_db, init_firebase

init_firebase()
db = get_db()

docs = db.collection("physio_exercise_plans").get()
print(f"Total exercise plans: {len(docs)}")
for d in docs:
    data = d.to_dict()
    user_id = data.get("user_id", "?")
    plan_date = data.get("date", "?")
    exercises = data.get("exercises", [])
    completed_exercises = data.get("completed_exercises", [])
    completed = data.get("completed", False)
    print(f"  user_id={user_id}, date={plan_date}, exercises={len(exercises)}, "
          f"completed_exercises={len(completed_exercises)}, completed={completed}")

# Now simulate the exact report function
from datetime import datetime, timedelta
now = datetime.utcnow()
start = now - timedelta(days=7)
start_str = start.strftime("%Y-%m-%d")

# Get unique user IDs
user_ids = set(d.to_dict().get("user_id") for d in docs)
print(f"\nUnique user IDs: {user_ids}")

for uid in user_ids:
    print(f"\n--- Simulating _get_physio_stats for user {uid} ---")
    udocs = db.collection("physio_exercise_plans").where("user_id", "==", uid).get()
    print(f"  Query returned {len(udocs)} docs")
    plans = []
    for d in udocs:
        data = d.to_dict()
        if data.get("date", "") >= start_str:
            plans.append(data)
    total = len(plans)
    completed_count = sum(1 for p in plans if p.get("completed"))
    total_ex = sum(len(p.get("exercises", [])) for p in plans)
    completed_ex = sum(len(p.get("completed_exercises", [])) for p in plans)
    rate = round((completed_count / total) * 100) if total > 0 else 0
    print(f"  total_plans={total}, completed={completed_count}, rate={rate}%")
    print(f"  total_exercises={total_ex}, completed_exercises={completed_ex}")
