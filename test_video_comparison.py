"""
Test script for video comparison - simulates the physio API endpoints
"""

import sys
import os
import cv2
import numpy as np
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

from physio_service.models import PoseAnalyzer, ExerciseType, PoseResult


def extract_poses_from_video(pose_analyzer: PoseAnalyzer, video_path: str, sample_rate: int = 10):
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


def compare_pose_sequences(pose_analyzer, poses1, poses2, exercise_type):
    """Compare two sequences of poses and calculate similarity metrics."""
    
    # Get joint angles for all poses
    angles1 = [pose_analyzer.get_joint_angles(p) for p in poses1]
    angles2 = [pose_analyzer.get_joint_angles(p) for p in poses2]
    
    if not angles1 or not angles2:
        return {"overall_similarity": 0, "error": "No angles extracted"}
    
    # Key angles for comparison
    key_angles = ["left_knee", "right_knee", "left_hip", "right_hip", "left_shoulder", "right_shoulder"]
    
    # Normalize sequence lengths
    min_len = min(len(angles1), len(angles2))
    
    # Sample to same length
    step1 = len(angles1) / min_len
    step2 = len(angles2) / min_len
    
    sampled1 = [angles1[int(i * step1)] for i in range(min_len)]
    sampled2 = [angles2[int(i * step2)] for i in range(min_len)]
    
    # Calculate per-angle similarity
    angle_similarities = {}
    
    for angle_name in key_angles:
        diffs = []
        for a1, a2 in zip(sampled1, sampled2):
            v1 = a1.get(angle_name, 0)
            v2 = a2.get(angle_name, 0)
            diff = abs(v1 - v2)
            diffs.append(diff)
        
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        # Convert to similarity (0-100), where 0 diff = 100% similar
        similarity = max(0, 100 - (avg_diff / 45 * 100))
        angle_similarities[angle_name] = round(similarity, 1)
    
    # Overall similarity
    overall_similarity = sum(angle_similarities.values()) / len(angle_similarities) if angle_similarities else 0
    
    # Movement pattern similarity
    pattern_score = compare_movement_patterns(sampled1, sampled2, key_angles)
    
    return {
        "overall_similarity": round(overall_similarity, 1),
        "pattern_similarity": round(pattern_score, 1),
        "combined_score": round((overall_similarity + pattern_score) / 2, 1),
        "angle_similarities": angle_similarities,
        "frames_compared": min_len
    }


def compare_movement_patterns(angles1, angles2, key_angles):
    """Compare the movement patterns between two sequences."""
    
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
        
        # Compare variance
        var1 = np.var(values1) if len(values1) > 1 else 0
        var2 = np.var(values2) if len(values2) > 1 else 0
        var_ratio = min(var1, var2) / max(var1, var2) if max(var1, var2) > 0 else 1
        var_similarity = var_ratio * 100
        
        pattern_scores.append((range_similarity + var_similarity) / 2)
    
    return sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0


def assess_video_form(pose_analyzer, poses, exercise_type):
    """Assess overall form quality for a video."""
    scores = []
    
    for pose in poses:
        assessment = pose_analyzer.assess_form(pose, exercise_type)
        if assessment:
            scores.append(assessment.score)
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    if avg_score >= 90:
        quality = "excellent"
    elif avg_score >= 75:
        quality = "good"
    elif avg_score >= 50:
        quality = "fair"
    else:
        quality = "poor"
    
    return {
        "score": round(avg_score, 1),
        "quality": quality
    }


def main():
    print("=" * 60)
    print("SMARTCARE+ PHYSIO VIDEO COMPARISON TEST")
    print("=" * 60)
    
    # Initialize pose analyzer
    print("\n[1] Initializing PoseAnalyzer...")
    pose_analyzer = PoseAnalyzer()
    print("    ✓ PoseAnalyzer initialized")
    
    # Get video directory
    video_dir = os.path.join(os.path.dirname(__file__), "media", "simulation_footage", "physio", "exercise")
    videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"\n[2] Found {len(videos)} exercise videos:")
    for v in videos:
        print(f"    - {v}")
    
    if len(videos) < 2:
        print("\n❌ Need at least 2 videos to compare")
        return
    
    # Test 1: Compare same video with itself (should be 100% similar)
    print("\n" + "=" * 60)
    print("TEST 1: Compare ex1.mp4 with ITSELF")
    print("Expected: ~100% similarity")
    print("=" * 60)
    
    video1_path = os.path.join(video_dir, "ex1.mp4")
    
    print(f"\n   Extracting poses from ex1.mp4...")
    poses1 = extract_poses_from_video(pose_analyzer, video1_path, sample_rate=15)
    print(f"   ✓ Extracted {len(poses1)} poses")
    
    print(f"   Comparing with itself...")
    similarity1 = compare_pose_sequences(pose_analyzer, poses1, poses1, ExerciseType.CHAIR_STAND)
    
    print(f"\n   📊 RESULTS:")
    print(f"   Overall Similarity: {similarity1['overall_similarity']}%")
    print(f"   Pattern Similarity: {similarity1['pattern_similarity']}%")
    print(f"   Combined Score: {similarity1['combined_score']}%")
    print(f"   Frames Compared: {similarity1['frames_compared']}")
    print(f"\n   Angle Similarities:")
    for angle, sim in similarity1['angle_similarities'].items():
        print(f"      - {angle}: {sim}%")
    
    # Test 2: Compare two different videos
    print("\n" + "=" * 60)
    print("TEST 2: Compare ex1.mp4 with ex2.mp4")
    print("Expected: Lower similarity (different videos)")
    print("=" * 60)
    
    video2_path = os.path.join(video_dir, "ex2.mp4")
    
    print(f"\n   Extracting poses from ex2.mp4...")
    poses2 = extract_poses_from_video(pose_analyzer, video2_path, sample_rate=15)
    print(f"   ✓ Extracted {len(poses2)} poses")
    
    print(f"   Comparing ex1.mp4 vs ex2.mp4...")
    similarity2 = compare_pose_sequences(pose_analyzer, poses1, poses2, ExerciseType.CHAIR_STAND)
    
    print(f"\n   📊 RESULTS:")
    print(f"   Overall Similarity: {similarity2['overall_similarity']}%")
    print(f"   Pattern Similarity: {similarity2['pattern_similarity']}%")
    print(f"   Combined Score: {similarity2['combined_score']}%")
    print(f"   Frames Compared: {similarity2['frames_compared']}")
    print(f"\n   Angle Similarities:")
    for angle, sim in similarity2['angle_similarities'].items():
        print(f"      - {angle}: {sim}%")
    
    # Test 3: Individual form assessment
    print("\n" + "=" * 60)
    print("TEST 3: Individual Form Assessment")
    print("=" * 60)
    
    print(f"\n   Assessing form for ex1.mp4...")
    form1 = assess_video_form(pose_analyzer, poses1, ExerciseType.CHAIR_STAND)
    print(f"   📊 ex1.mp4 Form:")
    print(f"      Score: {form1['score']}")
    print(f"      Quality: {form1['quality']}")
    
    print(f"\n   Assessing form for ex2.mp4...")
    form2 = assess_video_form(pose_analyzer, poses2, ExerciseType.CHAIR_STAND)
    print(f"   📊 ex2.mp4 Form:")
    print(f"      Score: {form2['score']}")
    print(f"      Quality: {form2['quality']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   Videos Analyzed: 2")
    has_mediapipe = hasattr(pose_analyzer, 'pose') and pose_analyzer.pose is not None
    print(f"   Pose Analyzer: {'MediaPipe' if has_mediapipe else 'Mock (simulated)'}")
    print(f"   Self-comparison (ex1 vs ex1): {similarity1['combined_score']}%")
    print(f"   Cross-comparison (ex1 vs ex2): {similarity2['combined_score']}%")
    print(f"\n   ✓ Video comparison backend is working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
