"""
Test script for 3-Layer Movement Analyzer

Tests the enhanced exercise monitoring system with pre-recorded videos.
Videos in media/simulation_footage/physio/exercise/ are named generically (ex1.mp4, ex2.mp4, etc.)
so we test motion detection capabilities rather than exercise-specific accuracy.

Run: python test_movement_analyzer.py
"""

import sys
import os
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import cv2
import time
from pathlib import Path

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
    get_movement_analyzer,
    reset_movement_analyzer
)


def analyze_video_with_movement_analyzer(
    video_path: str,
    exercise_type: ExerciseType = ExerciseType.CHAIR_STAND,
    sample_rate: int = 5  # Process every Nth frame
) -> dict:
    """
    Analyze a video using the 3-layer movement analyzer.
    
    Returns detailed analysis results.
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING: {os.path.basename(video_path)}")
    print(f"Exercise Type: {exercise_type.value}")
    print(f"{'='*60}")
    
    # Initialize analyzers
    pose_analyzer = get_pose_analyzer()
    movement_analyzer = MovementAnalyzer(exercise_type.value)
    
    # Get exercise thresholds
    thresholds = pose_analyzer.EXERCISE_THRESHOLDS.get(exercise_type, {})
    threshold_up = thresholds.get("rep_up", 150)
    threshold_down = thresholds.get("rep_down", 100)
    
    # Determine primary angle
    primary_angle_mapping = {
        ExerciseType.CHAIR_STAND: "avg_knee",
        ExerciseType.SQUAT: "avg_knee",
        ExerciseType.LEG_RAISE: "avg_hip",
        ExerciseType.ARM_RAISE: "avg_shoulder",
        ExerciseType.WALL_PUSHUP: "avg_elbow",
        ExerciseType.MARCHING: "avg_hip",
    }
    primary_angle_name = primary_angle_mapping.get(exercise_type, "avg_knee")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video: {video_path}"}
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video Info: {total_frames} frames @ {fps:.1f}fps ({duration:.1f}s)")
    
    # Track results
    frame_count = 0
    poses_detected = 0
    reps_completed = 0
    phases_seen = set()
    
    # Posture validation tracking
    posture_valid_frames = 0
    posture_invalid_frames = 0
    detected_postures = {}  # posture -> count
    
    # Metrics accumulators
    smoothness_scores = []
    tempo_scores = []
    pain_confidences = []
    phase_transitions = []
    last_phase = MovementPhase.NEUTRAL
    
    # Form guidance tracking
    form_match_scores = []
    form_corrections = []
    
    # Process frames
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sample every Nth frame
        if frame_count % sample_rate != 0:
            continue
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = (frame_count / fps) * 1000
        
        # Detect pose
        pose = pose_analyzer.detect_pose(rgb_frame, timestamp_ms)
        
        if pose and pose.confidence > 0.3 and len(pose.landmarks) > 20:
            poses_detected += 1
            
            # Get joint angles
            angles = pose_analyzer.get_joint_angles(pose)
            
            # Calculate average angles for primary tracking
            if "avg_knee" not in angles:
                angles["avg_knee"] = (angles.get("left_knee", 180) + angles.get("right_knee", 180)) / 2
            if "avg_hip" not in angles:
                angles["avg_hip"] = (angles.get("left_hip", 180) + angles.get("right_hip", 180)) / 2
            if "avg_shoulder" not in angles:
                angles["avg_shoulder"] = (angles.get("left_shoulder", 0) + angles.get("right_shoulder", 0)) / 2
            if "avg_elbow" not in angles:
                angles["avg_elbow"] = (angles.get("left_elbow", 180) + angles.get("right_elbow", 180)) / 2
            
            # Analyze movement (with posture validation via landmarks)
            timestamp = pose.timestamp / 1000 if pose.timestamp > 1e10 else pose.timestamp
            result = movement_analyzer.analyze(
                angles=angles,
                primary_angle_name=primary_angle_name,
                threshold_up=threshold_up,
                threshold_down=threshold_down,
                timestamp=timestamp,
                landmarks=pose.landmarks  # Pass landmarks for posture validation
            )
            
            # Report posture validation issues
            if not result.posture_validation.is_valid:
                posture_invalid_frames += 1
                if frame_count <= 30:  # Only report first few frames
                    print(f"  [MISMATCH] Frame {frame_count}: Detected {result.posture_validation.detected_posture.value}, expected {result.posture_validation.expected_postures}")
            else:
                posture_valid_frames += 1
            
            # Track detected postures
            detected_p = result.posture_validation.detected_posture.value
            detected_postures[detected_p] = detected_postures.get(detected_p, 0) + 1
            
            # Track rep completions
            if result.rep_result.rep_completed:
                reps_completed += 1
                print(f"  [REP] Rep {reps_completed} completed at frame {frame_count}")
            
            # Track phases
            current_phase = result.rep_result.current_phase
            phases_seen.add(current_phase)
            if current_phase != last_phase:
                phase_transitions.append({
                    "frame": frame_count,
                    "from": last_phase.value,
                    "to": current_phase.value
                })
                last_phase = current_phase
            
            # Accumulate metrics
            smoothness_scores.append(result.velocity_metrics.smoothness_score)
            tempo_scores.append(result.velocity_metrics.tempo_score)
            pain_confidences.append(result.pain_indicators.overall_confidence)
            
            # Track form guidance
            if result.form_guidance:
                form_match_scores.append(result.form_guidance.get('match_score', 100))
                priority_fix = result.form_guidance.get('priority_fix', '')
                if priority_fix and priority_fix not in [c['hint'] for c in form_corrections]:
                    form_corrections.append({
                        'frame': frame_count,
                        'phase': result.form_guidance.get('current_phase', ''),
                        'hint': priority_fix,
                        'score': result.form_guidance.get('match_score', 100)
                    })
    
    cap.release()
    processing_time = time.time() - start_time
    
    # Calculate summary statistics
    avg_smoothness = sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 0
    avg_tempo = sum(tempo_scores) / len(tempo_scores) if tempo_scores else 0
    max_pain = max(pain_confidences) if pain_confidences else 0
    avg_pain = sum(pain_confidences) / len(pain_confidences) if pain_confidences else 0
    
    # Calculate posture validation stats
    total_posture_frames = posture_valid_frames + posture_invalid_frames
    posture_match_rate = (posture_valid_frames / total_posture_frames * 100) if total_posture_frames > 0 else 0
    dominant_posture = max(detected_postures.keys(), key=lambda k: detected_postures[k]) if detected_postures else "unknown"
    exercise_is_valid = posture_match_rate > 50  # Majority of frames must match
    
    results = {
        "video": os.path.basename(video_path),
        "exercise_type": exercise_type.value,
        "video_info": {
            "total_frames": total_frames,
            "fps": round(fps, 1),
            "duration_seconds": round(duration, 1),
        },
        "posture_validation": {
            "exercise_valid": exercise_is_valid,
            "dominant_posture": dominant_posture,
            "posture_distribution": detected_postures,
            "match_rate_percent": round(posture_match_rate, 1),
            "valid_frames": posture_valid_frames,
            "invalid_frames": posture_invalid_frames,
        },
        "processing": {
            "frames_analyzed": poses_detected,
            "sample_rate": sample_rate,
            "processing_time_seconds": round(processing_time, 2),
        },
        "rep_detection": {
            "reps_completed": reps_completed,
            "phases_observed": [p.value for p in phases_seen],
            "phase_transitions": len(phase_transitions),
        },
        "velocity_metrics": {
            "avg_smoothness_score": round(avg_smoothness, 1),
            "avg_tempo_score": round(avg_tempo, 1),
        },
        "pain_detection": {
            "max_confidence": round(max_pain, 3),
            "avg_confidence": round(avg_pain, 3),
            "any_detected": max_pain > 0.3,
        },
        "form_guidance": {
            "avg_match_score": round(sum(form_match_scores) / len(form_match_scores), 1) if form_match_scores else 100.0,
            "min_match_score": round(min(form_match_scores), 1) if form_match_scores else 100.0,
            "corrections_needed": len(form_corrections),
            "top_corrections": form_corrections[:5] if form_corrections else [],  # Top 5
        },
    }
    
    return results


def print_results(results: dict):
    """Pretty print analysis results."""
    if "error" in results:
        print(f"[ERROR] {results['error']}")
        return
    
    print(f"\n[RESULTS] {results['video']}")
    print(f"   Exercise Type: {results['exercise_type']}")
    
    # Posture Validation - CRITICAL (show this first!)
    pv = results.get('posture_validation', {})
    if pv:
        is_valid = pv.get('exercise_valid', True)
        if not is_valid:
            print(f"\n   {'!'*50}")
            print(f"   POSTURE MISMATCH DETECTED!")
            print(f"   {'!'*50}")
        print(f"\n   Posture Validation:")
        print(f"     Exercise Valid: {'YES' if is_valid else 'NO - MISMATCH!'}")
        print(f"     Dominant Posture: {pv.get('dominant_posture', 'unknown')}")
        print(f"     Match Rate: {pv.get('match_rate_percent', 0)}%")
        print(f"     Posture Distribution: {pv.get('posture_distribution', {})}")
        if not is_valid:
            print(f"     [!] Rep counts below may be INVALID due to wrong exercise posture!")
    
    print(f"\n   Video Info:")
    print(f"     Duration: {results['video_info']['duration_seconds']}s @ {results['video_info']['fps']}fps")
    print(f"     Frames analyzed: {results['processing']['frames_analyzed']}")
    print(f"\n   Rep Detection:")
    print(f"     Reps Completed: {results['rep_detection']['reps_completed']}")
    print(f"     Phases Seen: {', '.join(results['rep_detection']['phases_observed'])}")
    print(f"     Phase Transitions: {results['rep_detection']['phase_transitions']}")
    print(f"\n   Movement Quality:")
    print(f"     Smoothness: {results['velocity_metrics']['avg_smoothness_score']}%")
    print(f"     Tempo Consistency: {results['velocity_metrics']['avg_tempo_score']}%")
    print(f"\n   Pain Detection:")
    print(f"     Max Confidence: {results['pain_detection']['max_confidence']}")
    print(f"     Any Detected: {'Yes [!]' if results['pain_detection']['any_detected'] else 'No'}")
    
    # Form Guidance (Reference Skeleton Matching)
    fg = results.get('form_guidance', {})
    if fg:
        avg_score = fg.get('avg_match_score', 100)
        print(f"\n   Form Guidance (Reference Matching):")
        print(f"     Average Match Score: {avg_score}%")
        print(f"     Minimum Match Score: {fg.get('min_match_score', 100)}%")
        print(f"     Corrections Needed: {fg.get('corrections_needed', 0)}")
        
        if avg_score < 80:
            print(f"     [!] FORM NEEDS IMPROVEMENT")
        elif avg_score < 95:
            print(f"     [~] Form is acceptable but could be better")
        else:
            print(f"     [✓] Excellent form!")
        
        corrections = fg.get('top_corrections', [])
        if corrections:
            print(f"     Top Corrections:")
            for i, corr in enumerate(corrections[:3], 1):
                print(f"       {i}. [{corr.get('phase', '')}] {corr.get('hint', '')}")
    
    print(f"\n   Processing Time: {results['processing']['processing_time_seconds']}s")


def test_all_videos():
    """Test movement analyzer with all available exercise videos."""
    video_dir = Path(__file__).parent / "media" / "simulation_footage" / "physio" / "exercise"
    
    if not video_dir.exists():
        print(f"[ERROR] Video directory not found: {video_dir}")
        return
    
    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        print(f"[ERROR] No MP4 files found in {video_dir}")
        return
    
    print(f"\n{'#'*60}")
    print(f"# MOVEMENT ANALYZER TEST SUITE")
    print(f"# Found {len(videos)} videos to analyze")
    print(f"{'#'*60}")
    
    # Test with different exercise types to see which works best
    test_exercise_types = [
        ExerciseType.CHAIR_STAND,
        ExerciseType.SQUAT,
        ExerciseType.LEG_RAISE,
        ExerciseType.ARM_RAISE,
    ]
    
    all_results = []
    
    for video_path in sorted(videos):
        # Use CHAIR_STAND as default since we don't know actual exercise
        # The analyzer will still track movement phases and velocity
        results = analyze_video_with_movement_analyzer(
            str(video_path),
            exercise_type=ExerciseType.CHAIR_STAND,
            sample_rate=5
        )
        print_results(results)
        all_results.append(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    total_reps = sum(r.get("rep_detection", {}).get("reps_completed", 0) for r in all_results)
    avg_smoothness = sum(r.get("velocity_metrics", {}).get("avg_smoothness_score", 0) for r in all_results) / len(all_results)
    videos_with_pain = sum(1 for r in all_results if r.get("pain_detection", {}).get("any_detected", False))
    
    print(f"   Videos Analyzed: {len(all_results)}")
    print(f"   Total Reps Detected: {total_reps}")
    print(f"   Average Smoothness: {avg_smoothness:.1f}%")
    print(f"   Videos with Pain Detected: {videos_with_pain}/{len(all_results)}")
    
    return all_results


def test_single_video(video_name: str = "ex1.mp4"):
    """Test a single video with detailed output."""
    video_dir = Path(__file__).parent / "media" / "simulation_footage" / "physio" / "exercise"
    video_path = video_dir / video_name
    
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    results = analyze_video_with_movement_analyzer(
        str(video_path),
        exercise_type=ExerciseType.CHAIR_STAND,
        sample_rate=3  # More detailed analysis
    )
    print_results(results)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test 3-Layer Movement Analyzer")
    parser.add_argument("--video", type=str, help="Test a specific video (e.g., ex1.mp4)")
    parser.add_argument("--all", action="store_true", help="Test all videos")
    
    args = parser.parse_args()
    
    if args.video:
        test_single_video(args.video)
    elif args.all:
        test_all_videos()
    else:
        # Default: test all videos
        test_all_videos()
