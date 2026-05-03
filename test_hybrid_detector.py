"""
SMARTCARE+ Hybrid Fall Detection System - Comprehensive Test Script

Tests the 3-layer hybrid fall detection system against all video types:
- Fall videos (should detect as falls)
- ADL videos (should NOT detect as falls)
- Good gait videos (should NOT detect as falls)
- TUG test videos (should NOT detect as falls)
- Arthritis gait videos (should NOT detect as falls)

Usage:
    python test_hybrid_detector.py
"""

import os
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # Fallback to default encoding

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from guardian_service.models import (
    get_hybrid_fall_detector,
    HybridFallResult,
    FallType,
    DetectionSource
)


@dataclass
class TestResult:
    """Result of a single video test."""
    video_path: str
    expected_fall: bool
    detected_fall: bool
    confidence: float
    detection_source: str
    skeleton_score: float
    motion_score: float
    dl_score: float
    processing_time: float
    correct: bool


def get_test_videos() -> Dict[str, List[Path]]:
    """Get all test videos organized by category."""
    video_dir = Path(__file__).parent / "media" / "simulation_footage" / "guardian"
    
    categories = {
        "fall": [],
        "adl": [],
        "good_gait": [],
        "arthritis_gait": [],
        "tug": [],
        "other": []
    }
    
    # Check subfolders
    for category in ["fall", "adl", "good_gait", "arthritis_gait", "tug"]:
        folder = video_dir / category
        if folder.exists():
            videos = list(folder.glob("*.mp4"))
            categories[category] = videos[:10]  # Limit to 10 per category for speed
    
    # Check root folder
    for video in video_dir.glob("*.mp4"):
        name_lower = video.stem.lower()
        if "fall" in name_lower:
            if video not in categories["fall"]:
                categories["fall"].append(video)
        elif "adl" in name_lower or "sitting" in name_lower or "lying" in name_lower or "standing" in name_lower:
            if video not in categories["adl"]:
                categories["adl"].append(video)
        elif "gait" in name_lower and "abnormal" in name_lower:
            if video not in categories["arthritis_gait"]:
                categories["arthritis_gait"].append(video)
        elif "gait" in name_lower or "walking" in name_lower:
            if video not in categories["good_gait"]:
                categories["good_gait"].append(video)
        elif "tug" in name_lower:
            if video not in categories["tug"]:
                categories["tug"].append(video)
        else:
            categories["other"].append(video)
    
    return categories


def test_video(detector, video_path: Path, expected_fall: bool) -> TestResult:
    """Test a single video and return the result."""
    start = time.time()
    result = detector.analyze_video_file(str(video_path))
    elapsed = time.time() - start
    
    correct = result.is_fall == expected_fall
    
    return TestResult(
        video_path=video_path.name,
        expected_fall=expected_fall,
        detected_fall=result.is_fall,
        confidence=result.confidence,
        detection_source=result.detection_source.value,
        skeleton_score=result.skeleton_score,
        motion_score=result.motion_score,
        dl_score=result.dl_score,
        processing_time=elapsed,
        correct=correct
    )


def print_result(result: TestResult, verbose: bool = True):
    """Print a single test result."""
    status = "✅" if result.correct else "❌"
    fall_str = "FALL" if result.detected_fall else "NO FALL"
    expected_str = "FALL" if result.expected_fall else "NO FALL"
    
    print(f"{status} {result.video_path}")
    if verbose:
        print(f"   Expected: {expected_str} | Detected: {fall_str} (conf: {result.confidence:.1%})")
        print(f"   Scores - Skeleton: {result.skeleton_score:.3f}, Motion: {result.motion_score:.3f}, DL: {result.dl_score:.3f}")
        print(f"   Source: {result.detection_source} | Time: {result.processing_time:.2f}s")


def run_tests(max_per_category: int = 5, verbose: bool = True, enable_dl: bool = True):
    """Run comprehensive tests on all video categories."""
    print("=" * 70)
    print("SMARTCARE+ Hybrid Fall Detection System - Comprehensive Test")
    print("=" * 70)
    print()
    
    # Initialize detector
    print(f"Initializing hybrid fall detector (DL enabled: {enable_dl})...")
    detector = get_hybrid_fall_detector(enable_dl=enable_dl)
    print("✅ Detector initialized")
    print()
    
    # Get test videos
    print("Scanning for test videos...")
    categories = get_test_videos()
    
    total_videos = sum(len(v) for v in categories.values())
    print(f"Found {total_videos} videos across {len(categories)} categories")
    for cat, videos in categories.items():
        print(f"  - {cat}: {len(videos)} videos")
    print()
    
    # Run tests
    all_results: List[TestResult] = []
    category_results: Dict[str, List[TestResult]] = {}
    
    for category, videos in categories.items():
        if not videos:
            continue
        
        # Determine expected result
        expected_fall = category == "fall"
        
        print(f"\n{'='*50}")
        print(f"Testing: {category.upper()} videos (expect {'FALL' if expected_fall else 'NO FALL'})")
        print(f"{'='*50}")
        
        category_results[category] = []
        
        for video in videos[:max_per_category]:
            result = test_video(detector, video, expected_fall)
            all_results.append(result)
            category_results[category].append(result)
            print_result(result, verbose)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_correct = sum(1 for r in all_results if r.correct)
    total_tests = len(all_results)
    
    print(f"\nOverall Accuracy: {total_correct}/{total_tests} ({total_correct/total_tests:.1%})")
    print()
    
    print("By Category:")
    for category, results in category_results.items():
        if not results:
            continue
        correct = sum(1 for r in results if r.correct)
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        expected = "FALL" if category == "fall" else "NO FALL"
        
        # Count true positives, false positives, etc.
        if category == "fall":
            tp = sum(1 for r in results if r.detected_fall)
            fn = sum(1 for r in results if not r.detected_fall)
            print(f"  {category:15s}: {accuracy:.1%} accuracy | {tp} detected, {fn} missed")
        else:
            tn = sum(1 for r in results if not r.detected_fall)
            fp = sum(1 for r in results if r.detected_fall)
            print(f"  {category:15s}: {accuracy:.1%} accuracy | {tn} correct, {fp} false alarms")
    
    # Calculate key metrics
    fall_results = category_results.get("fall", [])
    non_fall_results = [r for cat, results in category_results.items() 
                        if cat != "fall" for r in results]
    
    if fall_results:
        true_positives = sum(1 for r in fall_results if r.detected_fall)
        false_negatives = sum(1 for r in fall_results if not r.detected_fall)
        sensitivity = true_positives / (true_positives + false_negatives) if fall_results else 0
        print(f"\nSensitivity (True Positive Rate): {sensitivity:.1%}")
        print(f"  Falls correctly detected: {true_positives}/{len(fall_results)}")
    
    if non_fall_results:
        true_negatives = sum(1 for r in non_fall_results if not r.detected_fall)
        false_positives = sum(1 for r in non_fall_results if r.detected_fall)
        specificity = true_negatives / (true_negatives + false_positives) if non_fall_results else 0
        print(f"\nSpecificity (True Negative Rate): {specificity:.1%}")
        print(f"  Non-falls correctly identified: {true_negatives}/{len(non_fall_results)}")
        
        if false_positives > 0:
            print(f"\nFalse Alarms ({false_positives}):")
            for r in non_fall_results:
                if r.detected_fall:
                    print(f"  ⚠️  {r.video_path} (conf: {r.confidence:.1%}, source: {r.detection_source})")
    
    # Average processing time
    avg_time = sum(r.processing_time for r in all_results) / len(all_results) if all_results else 0
    print(f"\nAverage Processing Time: {avg_time:.2f}s per video")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid fall detection system")
    parser.add_argument("--max-per-category", "-n", type=int, default=5,
                        help="Maximum videos to test per category")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                        help="Show detailed output")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Show minimal output")
    parser.add_argument("--no-dl", action="store_true",
                        help="Disable deep learning for faster testing")
    
    args = parser.parse_args()
    
    run_tests(
        max_per_category=args.max_per_category,
        verbose=not args.quiet,
        enable_dl=not args.no_dl
    )
