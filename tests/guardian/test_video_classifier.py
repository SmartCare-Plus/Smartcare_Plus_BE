#!/usr/bin/env python3
"""
🎬 SmartCare+ Guardian - Video Classifier Test Script

Tests the trained guardian model against video files in the 
media/simulation_footage/guardian folder (fall, adl, good_gait, etc.)

Usage:
    python -m tests.guardian.test_video_classifier
    
    Or from backend folder:
    python tests/guardian/test_video_classifier.py

Author: Madhushani (Guardian Service)
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import cv2
import numpy as np
from typing import List, Dict, Tuple
import json
from collections import defaultdict

# Import the video classifier
from guardian_service.models.video_classifier import (
    VideoClassifier, 
    VideoActivityType,
    VideoClassificationResult,
    NUM_FRAMES
)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Video test data directory
MEDIA_DIR = backend_dir / "media" / "simulation_footage" / "guardian"

# Test subdirectories with expected class labels
TEST_FOLDERS = {
    "fall": "fall",           # Expected: fall
    "adl": "adl",             # Expected: adl (Activities of Daily Living)
    "good_gait": "good_gait", # Expected: good_gait
    "arthritis_gait": "arthritis_gait",  # Expected: arthritis_gait
    "tug": "tug",             # Expected: tug (Timed Up and Go)
}


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_video_frames(video_path: str, num_frames: int = NUM_FRAMES) -> List[np.ndarray]:
    """
    Load frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample (default: 32)
        
    Returns:
        List of BGR frames (OpenCV format)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    logger.debug(f"  Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s duration")
    
    # Calculate frame indices to sample uniformly
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    
    # Pad if needed
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1].copy())
    
    return frames[:num_frames]


def discover_test_videos() -> Dict[str, List[str]]:
    """
    Discover all test video files organized by expected class.
    
    Returns:
        Dict mapping expected_class -> list of video file paths
    """
    videos_by_class = defaultdict(list)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    # Check subfolders
    for folder_name, expected_class in TEST_FOLDERS.items():
        folder_path = MEDIA_DIR / folder_name
        if folder_path.exists():
            for video_file in folder_path.iterdir():
                if video_file.suffix.lower() in video_extensions:
                    videos_by_class[expected_class].append(str(video_file))
    
    # Also check root folder for loose videos
    if MEDIA_DIR.exists():
        for video_file in MEDIA_DIR.iterdir():
            if video_file.is_file() and video_file.suffix.lower() in video_extensions:
                # Try to infer class from filename
                name_lower = video_file.stem.lower()
                if 'fall' in name_lower:
                    videos_by_class['fall'].append(str(video_file))
                elif 'adl' in name_lower or 'sitting' in name_lower or 'lying' in name_lower or 'standing' in name_lower:
                    videos_by_class['adl'].append(str(video_file))
                elif 'good_gait' in name_lower or 'normal' in name_lower or 'walking_normal' in name_lower:
                    videos_by_class['good_gait'].append(str(video_file))
                elif 'arthritis' in name_lower or 'abnormal' in name_lower or 'gait_abnormal' in name_lower:
                    videos_by_class['arthritis_gait'].append(str(video_file))
                elif 'tug' in name_lower:
                    videos_by_class['tug'].append(str(video_file))
                else:
                    # Unknown - put in 'unknown' bucket
                    videos_by_class['unknown'].append(str(video_file))
    
    return dict(videos_by_class)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests(max_videos_per_class: int = 5) -> Dict:
    """
    Run classification tests on all discovered videos.
    
    Args:
        max_videos_per_class: Maximum videos to test per class (for speed)
        
    Returns:
        Test results summary dict
    """
    print("\n" + "="*80)
    print("🎬 SmartCare+ Guardian - Video Classifier Test")
    print("="*80)
    
    # Discover test videos
    print("\n📁 Discovering test videos...")
    videos_by_class = discover_test_videos()
    
    total_videos = sum(len(v) for v in videos_by_class.values())
    print(f"   Found {total_videos} videos in {len(videos_by_class)} classes:")
    for cls, videos in videos_by_class.items():
        print(f"   - {cls}: {len(videos)} videos")
    
    if total_videos == 0:
        print("\n❌ No test videos found!")
        print(f"   Expected videos in: {MEDIA_DIR}")
        return {"error": "No test videos found"}
    
    # Initialize classifier
    print("\n🔧 Loading Video Classifier...")
    classifier = VideoClassifier()
    loaded = classifier.load_model()
    
    if not loaded:
        print("❌ Failed to load model!")
        return {"error": "Model load failed"}
    
    if getattr(classifier, '_use_fallback', False):
        print("⚠️  Using FALLBACK rule-based classifier (no DL model loaded)")
    else:
        print("✅ Deep learning model loaded successfully!")
    
    # Run tests
    print("\n" + "-"*80)
    print("📊 Running Classification Tests")
    print("-"*80)
    
    results = {
        "total": 0,
        "correct": 0,
        "by_class": {},
        "confusion_matrix": defaultdict(lambda: defaultdict(int)),
        "detailed_results": []
    }
    
    for expected_class, video_paths in videos_by_class.items():
        if expected_class == 'unknown':
            continue  # Skip unknown videos for accuracy calculation
            
        print(f"\n🎯 Testing class: {expected_class.upper()}")
        
        class_results = {"total": 0, "correct": 0, "predictions": []}
        
        # Limit videos per class for faster testing
        test_videos = video_paths[:max_videos_per_class]
        
        for video_path in test_videos:
            video_name = Path(video_path).name
            print(f"\n   📹 {video_name}")
            
            # Load video frames
            frames = load_video_frames(video_path)
            if len(frames) < 8:
                print(f"      ⚠️ Skipping - not enough frames ({len(frames)})")
                continue
            
            # Classify
            result = classifier.classify_frames(frames)
            
            # Determine correctness
            predicted_class = result.activity.value
            is_correct = predicted_class == expected_class
            
            # Log result
            status = "✅" if is_correct else "❌"
            print(f"      {status} Predicted: {predicted_class} (confidence: {result.confidence:.1%})")
            print(f"         Probabilities: {result.all_probabilities}")
            print(f"         Processing: {result.processing_time_ms:.0f}ms")
            
            if result.is_fall_detected:
                print(f"         🚨 FALL DETECTED!")
            if result.is_gait_abnormal:
                print(f"         ⚠️ Abnormal gait detected")
            
            # Update results
            results["total"] += 1
            class_results["total"] += 1
            
            if is_correct:
                results["correct"] += 1
                class_results["correct"] += 1
            
            results["confusion_matrix"][expected_class][predicted_class] += 1
            
            class_results["predictions"].append({
                "video": video_name,
                "predicted": predicted_class,
                "expected": expected_class,
                "confidence": result.confidence,
                "correct": is_correct,
                "probabilities": result.all_probabilities,
                "processing_time_ms": result.processing_time_ms
            })
            
            results["detailed_results"].append({
                "video": video_path,
                "expected": expected_class,
                "predicted": predicted_class,
                "confidence": result.confidence,
                "correct": is_correct
            })
        
        # Class summary
        class_accuracy = class_results["correct"] / class_results["total"] * 100 if class_results["total"] > 0 else 0
        print(f"\n   📈 {expected_class}: {class_results['correct']}/{class_results['total']} correct ({class_accuracy:.1f}%)")
        
        results["by_class"][expected_class] = class_results
    
    # Overall summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    overall_accuracy = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    print(f"\n   Overall Accuracy: {results['correct']}/{results['total']} ({overall_accuracy:.1f}%)")
    
    print("\n   Per-Class Accuracy:")
    for cls, cls_results in results["by_class"].items():
        acc = cls_results["correct"] / cls_results["total"] * 100 if cls_results["total"] > 0 else 0
        print(f"   - {cls}: {acc:.1f}%")
    
    print("\n   Confusion Matrix:")
    classes = list(results["by_class"].keys())
    header = "Expected\\Predicted".ljust(20) + " | " + " | ".join(c.ljust(12) for c in classes)
    print(f"   {header}")
    print("   " + "-" * len(header))
    for expected in classes:
        row = expected.ljust(20) + " | "
        row += " | ".join(str(results["confusion_matrix"][expected].get(pred, 0)).ljust(12) for pred in classes)
        print(f"   {row}")
    
    results["overall_accuracy"] = overall_accuracy
    results["using_fallback"] = getattr(classifier, '_use_fallback', False)
    
    return results


def test_single_video(video_path: str):
    """
    Test classification on a single video file.
    
    Args:
        video_path: Path to the video file
    """
    print(f"\n📹 Testing single video: {video_path}")
    
    # Initialize classifier
    classifier = VideoClassifier()
    classifier.load_model()
    
    # Load and classify
    frames = load_video_frames(video_path)
    if len(frames) < 8:
        print(f"❌ Not enough frames: {len(frames)}")
        return
    
    result = classifier.classify_frames(frames)
    
    print(f"\n📊 Results:")
    print(f"   Activity: {result.activity.value}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Probabilities:")
    for cls, prob in sorted(result.all_probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
        print(f"      {cls:15} [{bar}] {prob:.1%}")
    print(f"   Fall Detected: {result.is_fall_detected}")
    print(f"   Gait Abnormal: {result.is_gait_abnormal}")
    print(f"   Processing Time: {result.processing_time_ms:.0f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Guardian Video Classifier")
    parser.add_argument("--video", type=str, help="Path to single video to test")
    parser.add_argument("--max-per-class", type=int, default=5, help="Max videos to test per class")
    parser.add_argument("--all", action="store_true", help="Test all videos (no limit)")
    
    args = parser.parse_args()
    
    if args.video:
        test_single_video(args.video)
    else:
        max_videos = 1000 if args.all else args.max_per_class
        results = run_tests(max_videos_per_class=max_videos)
        
        # Save results to file
        output_file = backend_dir / "tests" / "guardian" / "test_results.json"
        with open(output_file, 'w') as f:
            # Convert defaultdict to regular dict for JSON serialization
            results["confusion_matrix"] = {k: dict(v) for k, v in results.get("confusion_matrix", {}).items()}
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Results saved to: {output_file}")
