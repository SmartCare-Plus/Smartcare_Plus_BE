"""
SMARTCARE+ Physio Service Router

Owner: Neelaka
Endpoints for physiotherapy analysis, exercise monitoring, and progress tracking.
Uses MediaPipe pose estimation for real-time exercise form feedback.
"""

from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import random
import cv2
import numpy as np
import base64
import asyncio
import tempfile
import os

from .models import (
    PoseAnalyzer,
    PoseResult,
    ExerciseType,
    FormQuality,
    FormAssessment,
    RepCounter,
    ExerciseSession,
    ExerciseSessionHandler,
    SessionState,
    get_pose_analyzer,
    get_session_handler
)

router = APIRouter()


# Service instances (singleton pattern)
_pose_analyzer: Optional[PoseAnalyzer] = None
_session_handler: Optional[ExerciseSessionHandler] = None


def get_services():
    """Get or initialize service instances."""
    global _pose_analyzer, _session_handler
    if _pose_analyzer is None:
        _pose_analyzer = get_pose_analyzer()
    if _session_handler is None:
        _session_handler = get_session_handler()
    return _pose_analyzer, _session_handler


# ============= Pydantic Models =============

class Exercise(BaseModel):
    id: str
    name: str
    category: str
    duration: str
    difficulty: str
    reps: str
    description: str
    is_prescribed: bool = False


class StartSessionRequest(BaseModel):
    user_id: str
    exercise_type: str
    target_reps: int = 10
    target_sets: int = 3


class SessionFeedbackRequest(BaseModel):
    session_id: str
    pain_level: Optional[int] = None  # 0-10
    difficulty_rating: Optional[int] = None  # 1-5
    notes: Optional[str] = None


# ============= Mock Data =============

EXERCISES_DB = [
    {"id": "1", "name": "Chair Stand", "category": "Strength", "duration": "10 min", "difficulty": "Easy", "reps": "3 sets x 10 reps", "description": "Stand up from chair without using hands", "is_prescribed": True},
    {"id": "2", "name": "Heel-to-Toe Walk", "category": "Gait", "duration": "5 min", "difficulty": "Medium", "reps": "2 sets x 20 steps", "description": "Walk in a straight line, heel to toe", "is_prescribed": True},
    {"id": "3", "name": "Single Leg Stand", "category": "Balance", "duration": "8 min", "difficulty": "Medium", "reps": "3 sets x 30 sec", "description": "Stand on one leg while holding support", "is_prescribed": True},
    {"id": "4", "name": "Ankle Circles", "category": "Flexibility", "duration": "5 min", "difficulty": "Easy", "reps": "10 circles each direction", "description": "Rotate ankles to improve flexibility", "is_prescribed": False},
    {"id": "5", "name": "Wall Push-ups", "category": "Strength", "duration": "8 min", "difficulty": "Easy", "reps": "3 sets x 8 reps", "description": "Push-ups against the wall for upper body", "is_prescribed": False},
    {"id": "6", "name": "Tandem Stand", "category": "Balance", "duration": "5 min", "difficulty": "Hard", "reps": "3 sets x 20 sec", "description": "Stand with feet in tandem position", "is_prescribed": False},
    {"id": "7", "name": "Marching in Place", "category": "Gait", "duration": "10 min", "difficulty": "Easy", "reps": "100 steps", "description": "March in place lifting knees high", "is_prescribed": True},
    {"id": "8", "name": "Seated Leg Raises", "category": "Strength", "duration": "8 min", "difficulty": "Easy", "reps": "3 sets x 10 each leg", "description": "Lift legs while seated in chair", "is_prescribed": False},
]


# ============= REST Endpoints =============

@router.post("/analyze-exercise")
async def analyze_exercise(
    video: UploadFile = File(...),
    exercise_type: str = "chair_stand"
):
    """
    Analyze exercise form from uploaded video using MediaPipe pose estimation.
    
    Detects:
    - Joint angles and body posture
    - Repetition counting
    - Form quality assessment
    - Safety concerns
    """
    pose_analyzer, _ = get_services()
    
    # Validate exercise type
    try:
        ex_type = ExerciseType(exercise_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid exercise type. Valid types: {[e.value for e in ExerciseType]}"
        )
    
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process video frames
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        rep_counter = RepCounter(ex_type)
        form_assessments = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze pose in frame
            pose_result = pose_analyzer.process_frame(frame)
            
            if pose_result and pose_result.pose_detected:
                # Count reps
                rep_counter.update(pose_result)
                
                # Assess form periodically
                if frame_count % 30 == 0:  # Every 30 frames
                    assessment = pose_analyzer.assess_form(pose_result, ex_type)
                    if assessment:
                        form_assessments.append(assessment)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate overall form quality
        if form_assessments:
            avg_score = sum(a.score for a in form_assessments) / len(form_assessments)
            if avg_score >= 0.9:
                overall_quality = FormQuality.EXCELLENT
            elif avg_score >= 0.75:
                overall_quality = FormQuality.GOOD
            elif avg_score >= 0.5:
                overall_quality = FormQuality.FAIR
            else:
                overall_quality = FormQuality.POOR
            
            # Collect all feedback
            all_feedback = []
            for a in form_assessments:
                all_feedback.extend(a.feedback)
            unique_feedback = list(set(all_feedback))[:5]
        else:
            avg_score = 0.0
            overall_quality = FormQuality.FAIR
            unique_feedback = ["Unable to detect clear pose in video"]
        
    finally:
        os.unlink(tmp_path)
    
    return {
        "status": "completed",
        "exercise_type": exercise_type,
        "frames_processed": frame_count,
        "reps_counted": rep_counter.count,
        "form_assessment": {
            "score": round(avg_score * 100, 1),
            "quality": overall_quality.value,
            "feedback": unique_feedback
        },
        "recommendations": [
            "Focus on controlled movement throughout the exercise",
            "Maintain proper alignment during each repetition"
        ]
    }


@router.post("/session/start")
async def start_exercise_session(request: StartSessionRequest):
    """
    Start a new exercise session for real-time monitoring.
    
    Returns a session ID for use with the WebSocket stream.
    """
    _, session_handler = get_services()
    
    try:
        ex_type = ExerciseType(request.exercise_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid exercise type. Valid types: {[e.value for e in ExerciseType]}"
        )
    
    session = session_handler.create_session(
        user_id=request.user_id,
        exercise_type=ex_type,
        target_reps=request.target_reps,
        target_sets=request.target_sets
    )
    
    return {
        "status": "created",
        "session_id": session.session_id,
        "user_id": request.user_id,
        "exercise_type": request.exercise_type,
        "target": {
            "reps": request.target_reps,
            "sets": request.target_sets
        },
        "websocket_url": f"/physio/ws/session/{session.session_id}"
    }


@router.post("/session/{session_id}/complete")
async def complete_session(session_id: str):
    """Complete an exercise session and get final results."""
    _, session_handler = get_services()
    
    session = session_handler.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = session_handler.complete_session(session_id)
    
    return {
        "status": "completed",
        "session_id": session_id,
        "result": result
    }


@router.post("/session/{session_id}/feedback")
async def submit_session_feedback(session_id: str, request: SessionFeedbackRequest):
    """Submit user feedback for a completed session."""
    _, session_handler = get_services()
    
    session = session_handler.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Store feedback (would save to database in production)
    return {
        "status": "feedback_recorded",
        "session_id": session_id,
        "pain_level": request.pain_level,
        "difficulty_rating": request.difficulty_rating
    }


@router.get("/exercises")
async def get_exercises(category: Optional[str] = None, difficulty: Optional[str] = None):
    """Get exercise library."""
    exercises = EXERCISES_DB.copy()
    
    if category and category != "All":
        exercises = [e for e in exercises if e["category"] == category]
    
    if difficulty:
        exercises = [e for e in exercises if e["difficulty"] == difficulty]
    
    return {
        "exercises": exercises,
        "total": len(exercises),
        "categories": ["All", "Balance", "Strength", "Flexibility", "Gait"]
    }


@router.get("/exercises/{user_id}")
async def get_user_exercises(user_id: str):
    """Get prescribed exercises for a user."""
    prescribed = [e for e in EXERCISES_DB if e["is_prescribed"]]
    return {
        "user_id": user_id,
        "exercises": prescribed,
        "total_prescribed": len(prescribed),
        "completion_rate": round(random.uniform(60, 95), 1)
    }


@router.get("/fall-risk/{user_id}")
async def get_fall_risk(user_id: str):
    """Get current fall risk score for a user."""
    risk_score = round(random.uniform(15, 45), 1)
    
    if risk_score < 20:
        risk_level = "Low"
    elif risk_score < 35:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    return {
        "user_id": user_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "contributing_factors": [
            "Gait asymmetry detected",
            "Previous fall history",
            "Medication side effects"
        ][:random.randint(1, 3)],
        "last_assessment": datetime.now().isoformat(),
        "trend": "improving" if random.random() > 0.5 else "stable"
    }


@router.get("/progress/{user_id}")
async def get_progress(user_id: str, days: int = 7):
    """Get physiotherapy progress over time."""
    progress = []
    for i in range(days):
        progress.append({
            "date": f"2026-01-{max(1, 31-i):02d}",
            "exercises_completed": random.randint(2, 5),
            "total_duration_minutes": random.randint(15, 45),
            "fall_risk_score": round(random.uniform(20, 40), 1),
            "gait_score": round(random.uniform(70, 95), 1)
        })
    
    return {
        "user_id": user_id,
        "progress": progress,
        "summary": {
            "total_sessions": random.randint(10, 20),
            "avg_daily_minutes": round(random.uniform(20, 35), 1),
            "improvement_percentage": round(random.uniform(5, 15), 1)
        }
    }


@router.get("/sessions/{user_id}")
async def get_recent_sessions(user_id: str, limit: int = 5):
    """Get recent physio sessions for a user."""
    sessions = []
    for i in range(limit):
        sessions.append({
            "id": f"session_{i+1}",
            "date": f"2026-01-{max(1, 31-i):02d}",
            "type": random.choice(["Gait Analysis", "TUG Test", "Balance Exercise", "Strength Training"]),
            "duration_minutes": random.randint(10, 30),
            "score": round(random.uniform(70, 95), 1),
            "status": "completed"
        })
    
    return {"user_id": user_id, "sessions": sessions}


# ============= WebSocket Endpoints =============

@router.websocket("/ws/stream/{user_id}")
async def physio_stream(websocket: WebSocket, user_id: str):
    """
    Real-time video stream for exercise monitoring.
    
    Receives video frames, processes with MediaPipe, returns pose analysis.
    """
    await websocket.accept()
    pose_analyzer, _ = get_services()
    
    try:
        await websocket.send_json({
            "type": "CONNECTED",
            "user_id": user_id,
            "message": "Physio stream connected"
        })
        
        while True:
            # Receive video frame as base64 or bytes
            data = await websocket.receive_bytes()
            
            # Decode frame
            try:
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({
                        "type": "ERROR",
                        "message": "Invalid frame data"
                    })
                    continue
                
                # Process frame with pose analyzer
                pose_result = pose_analyzer.process_frame(frame)
                
                if pose_result and pose_result.pose_detected:
                    # Get key joint angles for feedback
                    angles = {}
                    for angle in pose_result.joint_angles:
                        angles[angle.name] = round(angle.angle, 1)
                    
                    await websocket.send_json({
                        "type": "POSE_RESULT",
                        "pose_detected": True,
                        "landmark_count": len(pose_result.landmarks) if pose_result.landmarks else 0,
                        "joint_angles": angles,
                        "timestamp": pose_result.timestamp
                    })
                else:
                    await websocket.send_json({
                        "type": "POSE_RESULT",
                        "pose_detected": False,
                        "message": "No pose detected in frame"
                    })
                    
            except Exception as e:
                await websocket.send_json({
                    "type": "ERROR",
                    "message": f"Processing error: {str(e)}"
                })
            
    except WebSocketDisconnect:
        print(f"Client {user_id} disconnected from physio stream")


@router.websocket("/ws/session/{session_id}")
async def exercise_session_stream(websocket: WebSocket, session_id: str):
    """
    Real-time exercise session monitoring with form feedback.
    
    Provides:
    - Live rep counting
    - Form quality assessment
    - Pain detection warnings
    - Session progress updates
    """
    await websocket.accept()
    pose_analyzer, session_handler = get_services()
    
    session = session_handler.get_session(session_id)
    if not session:
        await websocket.send_json({
            "type": "ERROR",
            "message": f"Session {session_id} not found"
        })
        await websocket.close()
        return
    
    try:
        await websocket.send_json({
            "type": "SESSION_STARTED",
            "session_id": session_id,
            "exercise_type": session.exercise_type.value,
            "target_reps": session.target_reps,
            "target_sets": session.target_sets
        })
        
        session.state = SessionState.ACTIVE
        
        while True:
            data = await websocket.receive_bytes()
            
            try:
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Process frame
                pose_result = pose_analyzer.process_frame(frame)
                
                if pose_result and pose_result.pose_detected:
                    # Update session with new frame
                    frame_result = session_handler.process_session_frame(
                        session_id, 
                        pose_result
                    )
                    
                    if frame_result:
                        await websocket.send_json({
                            "type": "FRAME_RESULT",
                            "reps_completed": frame_result.get("reps", 0),
                            "sets_completed": frame_result.get("sets", 0),
                            "current_form": frame_result.get("form_quality", "unknown"),
                            "form_score": frame_result.get("form_score", 0),
                            "feedback": frame_result.get("feedback", []),
                            "session_progress": frame_result.get("progress", 0)
                        })
                        
                        # Check if set completed
                        if frame_result.get("set_completed"):
                            await websocket.send_json({
                                "type": "SET_COMPLETED",
                                "set_number": frame_result.get("sets", 0),
                                "set_score": frame_result.get("set_score", 0)
                            })
                        
                        # Check if session completed
                        if frame_result.get("session_completed"):
                            await websocket.send_json({
                                "type": "SESSION_COMPLETED",
                                "total_score": frame_result.get("total_score", 0),
                                "summary": frame_result.get("summary", {})
                            })
                            break
                
            except Exception as e:
                await websocket.send_json({
                    "type": "ERROR",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected")
        # Mark session as paused
        if session:
            session.state = SessionState.PAUSED


# ============= Video Comparison Endpoints =============

class VideoCompareRequest(BaseModel):
    reference_video_path: str
    comparison_video_path: str
    exercise_type: str = "chair_stand"
    sample_rate: int = 10  # Analyze every Nth frame


@router.post("/compare-videos")
async def compare_videos(request: VideoCompareRequest):
    """
    Compare two exercise videos and calculate similarity score.
    
    Uses pose estimation to extract joint angles and compares
    the movement patterns between reference and comparison videos.
    """
    pose_analyzer, _ = get_services()
    
    # Validate exercise type
    try:
        ex_type = ExerciseType(request.exercise_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid exercise type. Valid types: {[e.value for e in ExerciseType]}"
        )
    
    # Check if files exist
    if not os.path.exists(request.reference_video_path):
        raise HTTPException(status_code=404, detail=f"Reference video not found: {request.reference_video_path}")
    if not os.path.exists(request.comparison_video_path):
        raise HTTPException(status_code=404, detail=f"Comparison video not found: {request.comparison_video_path}")
    
    # Extract poses from both videos
    ref_poses = _extract_poses_from_video(pose_analyzer, request.reference_video_path, request.sample_rate)
    comp_poses = _extract_poses_from_video(pose_analyzer, request.comparison_video_path, request.sample_rate)
    
    if not ref_poses or not comp_poses:
        return {
            "status": "error",
            "message": "Could not extract poses from one or both videos",
            "reference_frames": len(ref_poses),
            "comparison_frames": len(comp_poses)
        }
    
    # Compare poses and calculate similarity
    similarity_results = _compare_pose_sequences(pose_analyzer, ref_poses, comp_poses, ex_type)
    
    return {
        "status": "completed",
        "exercise_type": request.exercise_type,
        "reference_video": os.path.basename(request.reference_video_path),
        "comparison_video": os.path.basename(request.comparison_video_path),
        "reference_frames_analyzed": len(ref_poses),
        "comparison_frames_analyzed": len(comp_poses),
        "similarity": similarity_results
    }


@router.get("/analyze-local-video/{video_name}")
async def analyze_local_video(video_name: str, exercise_type: str = "chair_stand"):
    """
    Analyze a video from the local media/simulation_footage/physio/exercise folder.
    """
    pose_analyzer, _ = get_services()
    
    # Construct path to video
    base_path = os.path.dirname(os.path.dirname(__file__))
    video_path = os.path.join(base_path, "media", "simulation_footage", "physio", "exercise", video_name)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_name}")
    
    # Validate exercise type
    try:
        ex_type = ExerciseType(exercise_type)
    except ValueError:
        ex_type = ExerciseType.CHAIR_STAND
    
    # Extract and analyze poses
    poses = _extract_poses_from_video(pose_analyzer, video_path, sample_rate=5)
    
    if not poses:
        return {
            "status": "error",
            "video": video_name,
            "message": "Could not detect poses in video"
        }
    
    # Analyze form for each pose
    form_scores = []
    all_feedback = []
    joint_angle_history = []
    
    for pose in poses:
        assessment = pose_analyzer.assess_form(pose, ex_type)
        if assessment:
            form_scores.append(assessment.score)
            all_feedback.extend(assessment.feedback)
            
            # Get joint angles
            angles = pose_analyzer.get_joint_angles(pose)
            joint_angle_history.append(angles)
    
    avg_score = sum(form_scores) / len(form_scores) if form_scores else 0
    unique_feedback = list(set(all_feedback))[:5]
    
    # Calculate average joint angles
    avg_angles = {}
    if joint_angle_history:
        for key in joint_angle_history[0].keys():
            values = [angles.get(key, 0) for angles in joint_angle_history]
            avg_angles[key] = round(sum(values) / len(values), 1)
    
    return {
        "status": "completed",
        "video": video_name,
        "exercise_type": exercise_type,
        "frames_analyzed": len(poses),
        "form_assessment": {
            "average_score": round(avg_score, 1),
            "quality": _score_to_quality(avg_score),
            "feedback": unique_feedback
        },
        "average_joint_angles": avg_angles,
        "analysis_timestamp": datetime.now().isoformat()
    }


@router.get("/compare-local-videos")
async def compare_local_videos(
    video1: str,
    video2: str,
    exercise_type: str = "chair_stand"
):
    """
    Compare two local videos from the physio exercise folder.
    
    Example: /api/physio/compare-local-videos?video1=ex1.mp4&video2=ex2.mp4
    """
    pose_analyzer, _ = get_services()
    
    base_path = os.path.dirname(os.path.dirname(__file__))
    video_dir = os.path.join(base_path, "media", "simulation_footage", "physio", "exercise")
    
    video1_path = os.path.join(video_dir, video1)
    video2_path = os.path.join(video_dir, video2)
    
    if not os.path.exists(video1_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video1}")
    if not os.path.exists(video2_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video2}")
    
    try:
        ex_type = ExerciseType(exercise_type)
    except ValueError:
        ex_type = ExerciseType.CHAIR_STAND
    
    # Extract poses
    poses1 = _extract_poses_from_video(pose_analyzer, video1_path, sample_rate=10)
    poses2 = _extract_poses_from_video(pose_analyzer, video2_path, sample_rate=10)
    
    if not poses1 or not poses2:
        return {
            "status": "error",
            "message": "Could not extract poses from one or both videos",
            "video1_frames": len(poses1) if poses1 else 0,
            "video2_frames": len(poses2) if poses2 else 0
        }
    
    # Compare
    similarity = _compare_pose_sequences(pose_analyzer, poses1, poses2, ex_type)
    
    # Also get individual form assessments
    form1 = _assess_video_form(pose_analyzer, poses1, ex_type)
    form2 = _assess_video_form(pose_analyzer, poses2, ex_type)
    
    return {
        "status": "completed",
        "video1": {
            "name": video1,
            "frames_analyzed": len(poses1),
            "form_score": form1["score"],
            "form_quality": form1["quality"]
        },
        "video2": {
            "name": video2,
            "frames_analyzed": len(poses2),
            "form_score": form2["score"],
            "form_quality": form2["quality"]
        },
        "comparison": similarity,
        "analysis_timestamp": datetime.now().isoformat()
    }


@router.get("/list-exercise-videos")
async def list_exercise_videos():
    """List all available exercise videos in the media folder."""
    base_path = os.path.dirname(os.path.dirname(__file__))
    video_dir = os.path.join(base_path, "media", "simulation_footage", "physio", "exercise")
    
    if not os.path.exists(video_dir):
        return {"videos": [], "path": video_dir, "error": "Directory not found"}
    
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    return {
        "videos": videos,
        "count": len(videos),
        "path": video_dir
    }


# ============= Helper Functions =============

def _extract_poses_from_video(pose_analyzer: PoseAnalyzer, video_path: str, sample_rate: int = 10) -> List[PoseResult]:
    """Extract pose results from video at specified sample rate."""
    poses = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return poses
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = (frame_count / fps) * 1000
            
            pose = pose_analyzer.detect_pose(rgb_frame, timestamp_ms)
            if pose:
                poses.append(pose)
        
        frame_count += 1
    
    cap.release()
    return poses


def _compare_pose_sequences(
    pose_analyzer: PoseAnalyzer,
    poses1: List[PoseResult],
    poses2: List[PoseResult],
    exercise_type: ExerciseType
) -> Dict[str, Any]:
    """Compare two sequences of poses and calculate similarity metrics."""
    
    # Get joint angles for all poses
    angles1 = [pose_analyzer.get_joint_angles(p) for p in poses1]
    angles2 = [pose_analyzer.get_joint_angles(p) for p in poses2]
    
    if not angles1 or not angles2:
        return {"overall_similarity": 0, "error": "No angles extracted"}
    
    # Key angles for comparison based on exercise type
    key_angles = ["left_knee", "right_knee", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]
    
    # Normalize sequence lengths using dynamic time warping concept (simplified)
    min_len = min(len(angles1), len(angles2))
    
    # Sample to same length
    step1 = len(angles1) / min_len
    step2 = len(angles2) / min_len
    
    sampled1 = [angles1[int(i * step1)] for i in range(min_len)]
    sampled2 = [angles2[int(i * step2)] for i in range(min_len)]
    
    # Calculate per-angle similarity
    angle_similarities = {}
    overall_diffs = []
    
    for angle_name in key_angles:
        diffs = []
        for a1, a2 in zip(sampled1, sampled2):
            v1 = a1.get(angle_name, 0)
            v2 = a2.get(angle_name, 0)
            diff = abs(v1 - v2)
            diffs.append(diff)
        
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        # Convert to similarity (0-100), where 0 diff = 100% similar
        # 45 degree diff = 0% similar
        similarity = max(0, 100 - (avg_diff / 45 * 100))
        angle_similarities[angle_name] = round(similarity, 1)
        overall_diffs.append(avg_diff)
    
    # Overall similarity
    overall_similarity = sum(angle_similarities.values()) / len(angle_similarities) if angle_similarities else 0
    
    # Movement pattern similarity (variance comparison)
    pattern_score = _compare_movement_patterns(sampled1, sampled2, key_angles)
    
    return {
        "overall_similarity": round(overall_similarity, 1),
        "pattern_similarity": round(pattern_score, 1),
        "combined_score": round((overall_similarity + pattern_score) / 2, 1),
        "angle_similarities": angle_similarities,
        "frames_compared": min_len
    }


def _compare_movement_patterns(angles1: List[Dict], angles2: List[Dict], key_angles: List[str]) -> float:
    """Compare the movement patterns (variance and range) between two sequences."""
    
    pattern_scores = []
    
    for angle_name in key_angles:
        values1 = [a.get(angle_name, 0) for a in angles1]
        values2 = [a.get(angle_name, 0) for a in angles2]
        
        if not values1 or not values2:
            continue
        
        # Compare range of motion
        range1 = max(values1) - min(values1)
        range2 = max(values2) - min(values2)
        range_diff = abs(range1 - range2)
        range_similarity = max(0, 100 - (range_diff / 90 * 100))
        
        # Compare variance (movement smoothness)
        var1 = np.var(values1) if len(values1) > 1 else 0
        var2 = np.var(values2) if len(values2) > 1 else 0
        var_ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else 1
        var_similarity = var_ratio * 100
        
        pattern_scores.append((range_similarity + var_similarity) / 2)
    
    return sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0


def _assess_video_form(pose_analyzer: PoseAnalyzer, poses: List[PoseResult], exercise_type: ExerciseType) -> Dict[str, Any]:
    """Assess overall form quality for a video."""
    scores = []
    
    for pose in poses:
        assessment = pose_analyzer.assess_form(pose, exercise_type)
        if assessment:
            scores.append(assessment.score)
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    return {
        "score": round(avg_score, 1),
        "quality": _score_to_quality(avg_score)
    }


def _score_to_quality(score: float) -> str:
    """Convert numeric score to quality label."""
    if score >= 90:
        return "excellent"
    elif score >= 75:
        return "good"
    elif score >= 50:
        return "fair"
    else:
        return "poor"
