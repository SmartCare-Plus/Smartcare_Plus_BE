"""
Test script for 3-Layer Movement Analyzer - Version 2

Includes exercise posture validation to prevent false positive detections
when video content doesn't match the expected exercise type.

Key improvements:
1. Detects base posture (standing, sitting, quadruped, lying, etc.)
2. Validates if posture matches expected exercise before counting reps
3. Reports posture mismatch when video doesn't match exercise type

Run: python test_movement_analyzer_v2.py --video ex1.mp4
"""

import sys
import os
import io
import argparse

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import time
from pathlib import Path
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

from physio_service.models import (
    PoseAnalyzer,
    ExerciseType,
    get_pose_analyzer,
)
from physio_service.models.movement_analyzer import (
    MovementAnalyzer,
    MovementAnalysisResult,
    MovementPhase,
)


# ═══════════════════════════════════════════════════════════════════════════════
# POSTURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class BasePosture(Enum):
    """Detected base body posture."""
    STANDING = "standing"
    SITTING = "sitting"
    QUADRUPED = "quadruped"      # On all fours (cat-cow, table-top)
    LYING_SUPINE = "lying_supine"  # On back
    LYING_PRONE = "lying_prone"    # On stomach
    KNEELING = "kneeling"
    UNKNOWN = "unknown"


@dataclass
class PostureDetectionResult:
    """Result of posture detection."""
    posture: BasePosture
    confidence: float
    details: Dict[str, float]


def detect_base_posture(landmarks: Dict, pose_analyzer: PoseAnalyzer) -> PostureDetectionResult:
    """
    Detect the base posture of the person from landmarks.
    
    Uses relative landmark positions to determine posture:
    - Standing: nose above hips above ankles, vertical alignment
    - Sitting: nose above hips, hips at similar height to knees
    - Quadruped: hands and knees on ground, horizontal spine
    - Lying: horizontal body alignment
    """
    if not landmarks or len(landmarks) < 20:
        return PostureDetectionResult(BasePosture.UNKNOWN, 0.0, {})
    
    # Key landmark indices (MediaPipe)
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    try:
        # Get Y coordinates (normalized, 0=top, 1=bottom)
        nose_y = landmarks.get(NOSE).y if NOSE in landmarks else 0.5
        l_shoulder_y = landmarks.get(LEFT_SHOULDER).y if LEFT_SHOULDER in landmarks else 0.5
        r_shoulder_y = landmarks.get(RIGHT_SHOULDER).y if RIGHT_SHOULDER in landmarks else 0.5
        l_hip_y = landmarks.get(LEFT_HIP).y if LEFT_HIP in landmarks else 0.5
        r_hip_y = landmarks.get(RIGHT_HIP).y if RIGHT_HIP in landmarks else 0.5
        l_knee_y = landmarks.get(LEFT_KNEE).y if LEFT_KNEE in landmarks else 0.5
        r_knee_y = landmarks.get(RIGHT_KNEE).y if RIGHT_KNEE in landmarks else 0.5
        l_ankle_y = landmarks.get(LEFT_ANKLE).y if LEFT_ANKLE in landmarks else 0.5
        r_ankle_y = landmarks.get(RIGHT_ANKLE).y if RIGHT_ANKLE in landmarks else 0.5
        l_wrist_y = landmarks.get(LEFT_WRIST).y if LEFT_WRIST in landmarks else 0.5
        r_wrist_y = landmarks.get(RIGHT_WRIST).y if RIGHT_WRIST in landmarks else 0.5
        
        # Averages
        avg_shoulder_y = (l_shoulder_y + r_shoulder_y) / 2
        avg_hip_y = (l_hip_y + r_hip_y) / 2
        avg_knee_y = (l_knee_y + r_knee_y) / 2
        avg_ankle_y = (l_ankle_y + r_ankle_y) / 2
        avg_wrist_y = (l_wrist_y + r_wrist_y) / 2
        
        # Calculate vertical distances
        head_to_hip = avg_hip_y - nose_y
        hip_to_knee = avg_knee_y - avg_hip_y
        knee_to_ankle = avg_ankle_y - avg_knee_y
        
        details = {
            "head_to_hip": head_to_hip,
            "hip_to_knee": hip_to_knee,
            "knee_to_ankle": knee_to_ankle,
            "shoulder_y": avg_shoulder_y,
            "hip_y": avg_hip_y,
            "wrist_y": avg_wrist_y,
        }
        
        # Detection logic
        
        # QUADRUPED: Wrists and knees at similar Y level, body horizontal
        # In quadruped, wrists are typically near waist level, spine is ~horizontal
        if abs(avg_wrist_y - avg_shoulder_y) < 0.15 and abs(avg_hip_y - avg_shoulder_y) < 0.15:
            # Horizontal spine (shoulders and hips at similar height)
            # Wrists on ground (similar height to shoulders)
            return PostureDetectionResult(BasePosture.QUADRUPED, 0.8, details)
        
        # LYING: Very small vertical distance from head to feet
        if abs(head_to_hip) < 0.1 and abs(hip_to_knee) < 0.1:
            if avg_hip_y > 0.5:  # Lower in frame = lying
                return PostureDetectionResult(BasePosture.LYING_SUPINE, 0.7, details)
            return PostureDetectionResult(BasePosture.LYING_PRONE, 0.7, details)
        
        # STANDING: Clear vertical progression nose > shoulders > hips > knees > ankles
        if (head_to_hip > 0.15 and hip_to_knee > 0.1 and knee_to_ankle > 0.05 and
            nose_y < avg_shoulder_y < avg_hip_y < avg_knee_y < avg_ankle_y):
            return PostureDetectionResult(BasePosture.STANDING, 0.9, details)
        
        # SITTING: Hips and knees at similar Y level (both "bent")
        if (head_to_hip > 0.1 and abs(hip_to_knee) < 0.1):
            return PostureDetectionResult(BasePosture.SITTING, 0.7, details)
        
        # KNEELING: Knees at ankle level, torso vertical
        if (abs(avg_knee_y - avg_ankle_y) < 0.1 and head_to_hip > 0.15):
            return PostureDetectionResult(BasePosture.KNEELING, 0.6, details)
        
        return PostureDetectionResult(BasePosture.UNKNOWN, 0.3, details)
        
    except Exception as e:
        return PostureDetectionResult(BasePosture.UNKNOWN, 0.0, {"error": str(e)})


def get_expected_postures(exercise_type: ExerciseType) -> List[BasePosture]:
    """Get expected postures for an exercise type."""
    posture_map = {
        ExerciseType.CHAIR_STAND: [BasePosture.SITTING, BasePosture.STANDING],
        ExerciseType.SQUAT: [BasePosture.STANDING],
        ExerciseType.LEG_RAISE: [BasePosture.STANDING, BasePosture.LYING_SUPINE],
        ExerciseType.ARM_RAISE: [BasePosture.STANDING, BasePosture.SITTING],
        ExerciseType.WALL_PUSHUP: [BasePosture.STANDING],
        ExerciseType.MARCHING: [BasePosture.STANDING],
        ExerciseType.SINGLE_LEG_STAND: [BasePosture.STANDING],
        ExerciseType.SEATED_LEG_RAISE: [BasePosture.SITTING],
        ExerciseType.SEATED_ARM_RAISES: [BasePosture.SITTING],
        ExerciseType.SHOULDER_ROLLS: [BasePosture.STANDING, BasePosture.SITTING],
    }
    return posture_map.get(exercise_type, [BasePosture.UNKNOWN])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_video_with_validation(
    video_path: str,
    exercise_type: ExerciseType = ExerciseType.CHAIR_STAND,
    sample_rate: int = 5
) -> dict:
    """
    Analyze video with posture validation.
    
    First detects the actual posture in the video, then validates
    if it matches the expected exercise type.
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(video_path)}")
    print(f"Expected Exercise: {exercise_type.value}")
    print(f"{'='*60}")
    
    # Initialize
    pose_analyzer = get_pose_analyzer()
    movement_analyzer = MovementAnalyzer(exercise_type.value)
    
    # Get thresholds
    thresholds = pose_analyzer.EXERCISE_THRESHOLDS.get(exercise_type, {})
    threshold_up = thresholds.get("rep_up", 150)
    threshold_down = thresholds.get("rep_down", 100)
    
    primary_angle_mapping = {
        ExerciseType.CHAIR_STAND: "avg_knee",
        ExerciseType.SQUAT: "avg_knee",
        ExerciseType.LEG_RAISE: "avg_hip",
        ExerciseType.ARM_RAISE: "avg_shoulder",
        ExerciseType.WALL_PUSHUP: "avg_elbow",
        ExerciseType.MARCHING: "avg_hip",
    }
    primary_angle_name = primary_angle_mapping.get(exercise_type, "avg_knee")
    expected_postures = get_expected_postures(exercise_type)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video Info: {total_frames} frames @ {fps:.1f}fps ({duration:.1f}s)")
    print(f"Expected postures for {exercise_type.value}: {[p.value for p in expected_postures]}")
    
    # Tracking
    frame_count = 0
    poses_detected = 0
    posture_counts: Dict[BasePosture, int] = {}
    posture_matches = 0
    reps_completed = 0
    
    # Sample first 20 frames to detect posture
    posture_samples = []
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % sample_rate != 0:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = (frame_count / fps) * 1000
        
        pose = pose_analyzer.detect_pose(rgb_frame, timestamp_ms)
        
        if pose and pose.confidence > 0.3 and len(pose.landmarks) > 20:
            poses_detected += 1
            
            # Detect posture
            posture_result = detect_base_posture(pose.landmarks, pose_analyzer)
            posture = posture_result.posture
            
            # Track posture frequency
            posture_counts[posture] = posture_counts.get(posture, 0) + 1
            
            # Check if posture matches expected
            if posture in expected_postures:
                posture_matches += 1
                
                # Only analyze movement if posture matches
                angles = pose_analyzer.get_joint_angles(pose)
                if "avg_knee" not in angles:
                    angles["avg_knee"] = (angles.get("left_knee", 180) + angles.get("right_knee", 180)) / 2
                if "avg_hip" not in angles:
                    angles["avg_hip"] = (angles.get("left_hip", 180) + angles.get("right_hip", 180)) / 2
                if "avg_shoulder" not in angles:
                    angles["avg_shoulder"] = (angles.get("left_shoulder", 0) + angles.get("right_shoulder", 0)) / 2
                if "avg_elbow" not in angles:
                    angles["avg_elbow"] = (angles.get("left_elbow", 180) + angles.get("right_elbow", 180)) / 2
                
                timestamp = pose.timestamp / 1000 if pose.timestamp > 1e10 else pose.timestamp
                result = movement_analyzer.analyze(
                    angles=angles,
                    primary_angle_name=primary_angle_name,
                    threshold_up=threshold_up,
                    threshold_down=threshold_down,
                    timestamp=timestamp,
                    landmarks=pose.landmarks  # Pass landmarks for posture validation
                )
                
                # Check posture validation result
                if not result.posture_validation.is_valid:
                    print(f"  [!] POSTURE MISMATCH at frame {frame_count}: {result.posture_validation.detected_posture.value}")
                
                if result.rep_result.rep_completed:
                    reps_completed += 1
                    print(f"  [REP] Rep {reps_completed} at frame {frame_count}")
            
            # Sample posture for first portion of video
            if len(posture_samples) < 20:
                posture_samples.append(posture)
    
    cap.release()
    processing_time = time.time() - start_time
    
    # Determine dominant posture
    if posture_counts:
        dominant_posture = max(posture_counts.keys(), key=lambda k: posture_counts[k])
        dominant_count = posture_counts[dominant_posture]
    else:
        dominant_posture = BasePosture.UNKNOWN
        dominant_count = 0
    
    # Calculate posture match rate
    match_rate = (posture_matches / poses_detected * 100) if poses_detected > 0 else 0
    
    # Validation result
    posture_valid = dominant_posture in expected_postures
    
    print(f"\n{'='*60}")
    print(f"POSTURE VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Dominant posture detected: {dominant_posture.value}")
    print(f"Expected postures: {[p.value for p in expected_postures]}")
    print(f"Posture match rate: {match_rate:.1f}%")
    print(f"Exercise type valid: {'YES' if posture_valid else 'NO - MISMATCH!'}")
    
    if not posture_valid:
        print(f"\n[!] WARNING: Video shows '{dominant_posture.value}' posture")
        print(f"    but '{exercise_type.value}' exercise expects {[p.value for p in expected_postures]}")
        print(f"    Rep count ({reps_completed}) may be INVALID due to posture mismatch!")
        
        # Suggest correct exercise type
        suggestions = suggest_exercise_for_posture(dominant_posture)
        if suggestions:
            print(f"    Suggested exercise types for this posture: {suggestions}")
    
    print(f"\n[RESULTS]")
    print(f"  Frames analyzed: {poses_detected}")
    print(f"  Posture distribution: {dict((k.value, v) for k,v in posture_counts.items())}")
    print(f"  Valid reps (with matched posture): {reps_completed}")
    print(f"  Processing time: {processing_time:.2f}s")
    
    return {
        "video": os.path.basename(video_path),
        "exercise_type": exercise_type.value,
        "posture_validation": {
            "dominant_posture": dominant_posture.value,
            "expected_postures": [p.value for p in expected_postures],
            "posture_valid": posture_valid,
            "match_rate_percent": round(match_rate, 1),
            "posture_distribution": {k.value: v for k, v in posture_counts.items()},
        },
        "rep_detection": {
            "reps_completed": reps_completed,
            "validated": posture_valid,
        },
        "video_info": {
            "total_frames": total_frames,
            "fps": round(fps, 1),
            "duration_seconds": round(duration, 1),
            "frames_analyzed": poses_detected,
        },
        "processing_time_seconds": round(processing_time, 2),
    }


def suggest_exercise_for_posture(posture: BasePosture) -> List[str]:
    """Suggest exercise types that match a detected posture."""
    suggestions = {
        BasePosture.STANDING: ["arm_raise", "squat", "marching", "leg_raise", "wall_pushup"],
        BasePosture.SITTING: ["chair_stand", "seated_leg_raise", "seated_arm_raises"],
        BasePosture.QUADRUPED: [],  # Not currently supported
        BasePosture.LYING_SUPINE: ["leg_raise"],
        BasePosture.KNEELING: [],
    }
    return suggestions.get(posture, [])


def detect_video_exercise(video_path: str, sample_rate: int = 10) -> Tuple[BasePosture, Dict]:
    """
    Detect what exercise/posture is in a video without assuming any type.
    
    Returns the dominant posture and posture statistics.
    """
    print(f"\n{'='*60}")
    print(f"DETECTING POSTURE IN: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    pose_analyzer = get_pose_analyzer()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return BasePosture.UNKNOWN, {"error": "Could not open video"}
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {total_frames} frames @ {fps:.1f}fps")
    
    posture_counts: Dict[BasePosture, int] = {}
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % sample_rate != 0:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = (frame_count / fps) * 1000
        
        pose = pose_analyzer.detect_pose(rgb_frame, timestamp_ms)
        
        if pose and pose.confidence > 0.3 and len(pose.landmarks) > 20:
            posture_result = detect_base_posture(pose.landmarks, pose_analyzer)
            posture = posture_result.posture
            posture_counts[posture] = posture_counts.get(posture, 0) + 1
    
    cap.release()
    
    if posture_counts:
        dominant = max(posture_counts.keys(), key=lambda k: posture_counts[k])
    else:
        dominant = BasePosture.UNKNOWN
    
    print(f"\nDetected postures: {dict((k.value, v) for k, v in posture_counts.items())}")
    print(f"Dominant posture: {dominant.value}")
    
    suggestions = suggest_exercise_for_posture(dominant)
    if suggestions:
        print(f"Compatible exercises: {suggestions}")
    else:
        print(f"[!] No compatible exercises defined for '{dominant.value}' posture")
    
    return dominant, {
        "posture_counts": {k.value: v for k, v in posture_counts.items()},
        "dominant_posture": dominant.value,
        "suggested_exercises": suggestions,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test Movement Analyzer with Posture Validation")
    parser.add_argument("--video", type=str, help="Video file to analyze (e.g., ex1.mp4)")
    parser.add_argument("--exercise", type=str, default=None, 
                        help="Exercise type (chair_stand, arm_raise, squat, etc.). If not specified, auto-detects posture first.")
    parser.add_argument("--detect-only", action="store_true",
                        help="Only detect posture without analyzing exercise")
    parser.add_argument("--all", action="store_true", help="Analyze all videos")
    
    args = parser.parse_args()
    
    video_dir = Path(__file__).parent / "media" / "simulation_footage" / "physio" / "exercise"
    
    if args.all:
        videos = sorted(video_dir.glob("*.mp4"))
        print(f"\n{'#'*60}")
        print(f"POSTURE DETECTION FOR ALL VIDEOS")
        print(f"{'#'*60}")
        
        for video in videos:
            detect_video_exercise(str(video))
        return
    
    if args.video:
        video_path = video_dir / args.video
        if not video_path.exists():
            print(f"[ERROR] Video not found: {video_path}")
            return
        
        if args.detect_only:
            detect_video_exercise(str(video_path))
        elif args.exercise:
            # User specified exercise type
            try:
                exercise = ExerciseType(args.exercise)
                analyze_video_with_validation(str(video_path), exercise)
            except ValueError:
                print(f"[ERROR] Unknown exercise type: {args.exercise}")
                print(f"Available: {[e.value for e in ExerciseType]}")
        else:
            # Auto-detect posture first, then suggest or analyze
            posture, info = detect_video_exercise(str(video_path), sample_rate=10)
            suggestions = info.get("suggested_exercises", [])
            
            if suggestions:
                print(f"\nAuto-analyzing with first suggested exercise: {suggestions[0]}")
                exercise = ExerciseType(suggestions[0])
                analyze_video_with_validation(str(video_path), exercise)
            else:
                print(f"\n[!] Cannot auto-analyze: No supported exercise for '{posture.value}' posture")
    else:
        print("Usage:")
        print("  python test_movement_analyzer_v2.py --video ex1.mp4 --detect-only")
        print("  python test_movement_analyzer_v2.py --video ex1.mp4 --exercise chair_stand")
        print("  python test_movement_analyzer_v2.py --video ex1.mp4  (auto-detect)")
        print("  python test_movement_analyzer_v2.py --all  (detect posture in all videos)")


if __name__ == "__main__":
    main()
