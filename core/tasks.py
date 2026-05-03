"""
SMARTCARE+ Task Service

Provides CRUD operations for caregiver tasks stored in Firestore.
Tasks persist across server restarts and re-logins.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.database import get_db, is_mock_mode

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Pydantic Models ───

class TaskCreate(BaseModel):
    caregiver_id: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    title: str
    description: Optional[str] = ""
    task_type: str = "General"          # Exercise, Medication, Check-up, Nutrition, Assessment, General
    time: Optional[str] = ""            # e.g. "10:00 AM"
    duration: Optional[str] = ""        # e.g. "30 min"
    is_priority: bool = False
    due_date: Optional[str] = None      # YYYY-MM-DD, defaults to today


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    task_type: Optional[str] = None
    time: Optional[str] = None
    duration: Optional[str] = None
    is_priority: Optional[bool] = None
    completed: Optional[bool] = None
    due_date: Optional[str] = None


# ─── In-memory fallback (mock mode) ───

_mock_tasks: dict = {}  # caregiver_id -> list of tasks


def _get_mock_tasks(caregiver_id: str) -> list:
    return _mock_tasks.get(caregiver_id, [])


def _save_mock_task(task: dict):
    cid = task["caregiver_id"]
    if cid not in _mock_tasks:
        _mock_tasks[cid] = []
    # Replace if exists
    _mock_tasks[cid] = [t for t in _mock_tasks[cid] if t["task_id"] != task["task_id"]]
    _mock_tasks[cid].append(task)


def _delete_mock_task(caregiver_id: str, task_id: str):
    if caregiver_id in _mock_tasks:
        _mock_tasks[caregiver_id] = [t for t in _mock_tasks[caregiver_id] if t["task_id"] != task_id]


# ─── Endpoints ───

@router.post("/create")
async def create_task(request: TaskCreate):
    """Create a new task for a caregiver."""
    task_id = str(uuid4())[:8]
    now = datetime.utcnow().isoformat()
    due = request.due_date or date.today().strftime("%Y-%m-%d")

    task_doc = {
        "task_id": task_id,
        "caregiver_id": request.caregiver_id,
        "patient_id": request.patient_id or "",
        "patient_name": request.patient_name or "",
        "title": request.title,
        "description": request.description or "",
        "task_type": request.task_type,
        "time": request.time or "",
        "duration": request.duration or "",
        "is_priority": request.is_priority,
        "completed": False,
        "due_date": due,
        "created_at": now,
        "updated_at": now,
    }

    db = get_db()
    if db and not is_mock_mode():
        try:
            db.collection("tasks").document(task_id).set(task_doc)
            logger.info(f"✅ Task created: {task_id} for caregiver {request.caregiver_id}")
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        _save_mock_task(task_doc)

    return {"success": True, "task": task_doc}


@router.get("/list/{caregiver_id}")
async def get_tasks(
    caregiver_id: str,
    filter: str = "all",          # "today", "week", "all"
    include_completed: bool = True,
):
    """
    Get all tasks for a caregiver.
    
    Filters:
    - today: tasks due today
    - week: tasks due within the next 7 days
    - all: all tasks
    """
    db = get_db()
    tasks = []

    if db and not is_mock_mode():
        try:
            query = db.collection("tasks").where("caregiver_id", "==", caregiver_id)
            docs = query.stream()

            for doc in docs:
                task = doc.to_dict()
                tasks.append(task)
        except Exception as e:
            logger.error(f"Error fetching tasks: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        tasks = _get_mock_tasks(caregiver_id)

    # Apply date filter
    today_str = date.today().strftime("%Y-%m-%d")
    week_end = (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")

    if filter == "today":
        tasks = [t for t in tasks if t.get("due_date", "") == today_str]
    elif filter == "week":
        tasks = [t for t in tasks if today_str <= t.get("due_date", "") <= week_end]

    # Filter completed
    if not include_completed:
        tasks = [t for t in tasks if not t.get("completed", False)]

    # Sort: incomplete first, then by priority (priority first), then by time
    def sort_key(t):
        completed = 1 if t.get("completed", False) else 0
        priority = 0 if t.get("is_priority", False) else 1
        time_str = t.get("time", "99:99")
        return (completed, priority, time_str)

    tasks.sort(key=sort_key)

    # Count stats
    total = len(tasks)
    completed_count = sum(1 for t in tasks if t.get("completed", False))
    pending_count = total - completed_count

    return {
        "success": True,
        "tasks": tasks,
        "total": total,
        "completed": completed_count,
        "pending": pending_count,
    }


@router.put("/update/{task_id}")
async def update_task(task_id: str, request: TaskUpdate):
    """Update a task (mark complete, edit details, etc.)."""
    db = get_db()

    if db and not is_mock_mode():
        try:
            doc_ref = db.collection("tasks").document(task_id)
            doc = doc_ref.get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Task not found")

            update_data = {k: v for k, v in request.dict().items() if v is not None}
            update_data["updated_at"] = datetime.utcnow().isoformat()
            doc_ref.update(update_data)

            updated_doc = doc_ref.get().to_dict()
            return {"success": True, "task": updated_doc}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating task: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Mock mode
        for cid, task_list in _mock_tasks.items():
            for i, t in enumerate(task_list):
                if t["task_id"] == task_id:
                    update_data = {k: v for k, v in request.dict().items() if v is not None}
                    update_data["updated_at"] = datetime.utcnow().isoformat()
                    _mock_tasks[cid][i].update(update_data)
                    return {"success": True, "task": _mock_tasks[cid][i]}
        raise HTTPException(status_code=404, detail="Task not found")


@router.delete("/delete/{task_id}")
async def delete_task(task_id: str, caregiver_id: str = ""):
    """Delete a task."""
    db = get_db()

    if db and not is_mock_mode():
        try:
            doc_ref = db.collection("tasks").document(task_id)
            doc = doc_ref.get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Task not found")
            doc_ref.delete()
            return {"success": True, "message": "Task deleted"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting task: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        _delete_mock_task(caregiver_id, task_id)
        return {"success": True, "message": "Task deleted"}


@router.put("/toggle/{task_id}")
async def toggle_task(task_id: str):
    """Toggle task completion status."""
    db = get_db()

    if db and not is_mock_mode():
        try:
            doc_ref = db.collection("tasks").document(task_id)
            doc = doc_ref.get()
            if not doc.exists:
                raise HTTPException(status_code=404, detail="Task not found")

            current = doc.to_dict()
            new_status = not current.get("completed", False)
            doc_ref.update({
                "completed": new_status,
                "updated_at": datetime.utcnow().isoformat(),
                "completed_at": datetime.utcnow().isoformat() if new_status else None,
            })

            updated = doc_ref.get().to_dict()
            return {"success": True, "task": updated}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error toggling task: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        for cid, task_list in _mock_tasks.items():
            for i, t in enumerate(task_list):
                if t["task_id"] == task_id:
                    _mock_tasks[cid][i]["completed"] = not t.get("completed", False)
                    _mock_tasks[cid][i]["updated_at"] = datetime.utcnow().isoformat()
                    return {"success": True, "task": _mock_tasks[cid][i]}
        raise HTTPException(status_code=404, detail="Task not found")
