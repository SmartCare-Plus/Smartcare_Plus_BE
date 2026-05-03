"""
SMARTCARE+ Fall Detection System - Comprehensive Evaluation & Analysis

Generates detailed metrics, graphs, and analysis for thesis/research publication:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves
- Layer-wise Performance Comparison
- Processing Time Analysis
- Per-category Performance
- Publication-quality visualizations

Usage:
    python evaluate_fall_detection.py --output results/
"""

import os
import sys
from pathlib import Path
import time
import json
import csv
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict

# Visualization libraries
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score, 
    recall_score, f1_score
)

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

sys.path.insert(0, str(Path(__file__).parent))

from guardian_service.models import (
    get_hybrid_fall_detector,
    HybridFallResult,
    FallType,
    DetectionSource
)


@dataclass
class DetailedResult:
    """Detailed test result for a single video."""
    video_name: str
    category: str
    expected_fall: bool
    detected_fall: bool
    confidence: float
    skeleton_score: float
    motion_score: float
    dl_score: float
    detection_source: str
    processing_time: float
    correct: bool
    
    def to_dict(self):
        return asdict(self)


class FallDetectionEvaluator:
    """Comprehensive evaluation of fall detection system."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[DetailedResult] = []
        self.detector = None
        
    def get_test_videos(self) -> Dict[str, List[Path]]:
        """Get all test videos organized by category."""
        video_dir = Path(__file__).parent / "media" / "simulation_footage" / "guardian"
        
        categories = {
            "fall": [],
            "adl": [],
            "good_gait": [],
            "arthritis_gait": [],
            "tug": []
        }
        
        # Scan category folders
        for category in categories.keys():
            folder = video_dir / category
            if folder.exists():
                categories[category] = sorted(folder.glob("*.mp4"))
        
        return categories
    
    def test_video(self, video_path: Path, expected_fall: bool, category: str) -> DetailedResult:
        """Test a single video and return detailed results."""
        start = time.time()
        result = self.detector.analyze_video_file(str(video_path))
        elapsed = time.time() - start
        
        return DetailedResult(
            video_name=video_path.name,
            category=category,
            expected_fall=expected_fall,
            detected_fall=result.is_fall,
            confidence=result.confidence,
            skeleton_score=result.skeleton_score,
            motion_score=result.motion_score,
            dl_score=result.dl_score,
            detection_source=result.detection_source.value,
            processing_time=elapsed,
            correct=result.is_fall == expected_fall
        )
    
    def run_evaluation(self, max_per_category: int = None):
        """Run comprehensive evaluation on all videos."""
        print("=" * 80)
        print("SMARTCARE+ Fall Detection System - Comprehensive Evaluation")
        print("=" * 80)
        print()
        
        # Initialize detector
        print("Initializing hybrid fall detector (3-layer system)...")
        self.detector = get_hybrid_fall_detector(enable_dl=True)
        print("✅ Detector initialized\n")
        
        # Get test videos
        print("Scanning for test videos...")
        categories = self.get_test_videos()
        
        total_videos = sum(len(v) for v in categories.values())
        print(f"Found {total_videos} videos across {len(categories)} categories")
        for cat, videos in categories.items():
            count = len(videos) if not max_per_category else min(len(videos), max_per_category)
            print(f"  - {cat}: {count} videos")
        print()
        
        # Test all videos
        for category, videos in categories.items():
            if not videos:
                continue
            
            expected_fall = (category == "fall")
            test_videos = videos[:max_per_category] if max_per_category else videos
            
            print(f"\nTesting {category.upper()} videos ({len(test_videos)} videos)...")
            
            for i, video in enumerate(test_videos, 1):
                result = self.test_video(video, expected_fall, category)
                self.results.append(result)
                
                status = "✅" if result.correct else "❌"
                print(f"  {status} [{i}/{len(test_videos)}] {video.name} - "
                      f"{'FALL' if result.detected_fall else 'NO FALL'} "
                      f"(conf: {result.confidence:.1%}, time: {result.processing_time:.1f}s)")
        
        print("\n" + "=" * 80)
        print("Evaluation Complete - Generating Analysis & Visualizations")
        print("=" * 80)
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        y_true = [r.expected_fall for r in self.results]
        y_pred = [r.detected_fall for r in self.results]
        y_scores = [r.confidence for r in self.results]
        
        metrics = {
            "overall": {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "sensitivity": recall_score(y_true, y_pred, zero_division=0),  # TPR
                "specificity": recall_score([not y for y in y_true], 
                                           [not y for y in y_pred], 
                                           zero_division=0),  # TNR
            },
            "per_category": {},
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "total_videos": len(self.results),
            "processing_times": {
                "mean": np.mean([r.processing_time for r in self.results]),
                "std": np.std([r.processing_time for r in self.results]),
                "min": np.min([r.processing_time for r in self.results]),
                "max": np.max([r.processing_time for r in self.results]),
            }
        }
        
        # Per-category metrics
        categories = set(r.category for r in self.results)
        for category in categories:
            cat_results = [r for r in self.results if r.category == category]
            cat_y_true = [r.expected_fall for r in cat_results]
            cat_y_pred = [r.detected_fall for r in cat_results]
            
            metrics["per_category"][category] = {
                "total": len(cat_results),
                "correct": sum(r.correct for r in cat_results),
                "accuracy": sum(r.correct for r in cat_results) / len(cat_results),
                "avg_confidence": np.mean([r.confidence for r in cat_results]),
                "avg_processing_time": np.mean([r.processing_time for r in cat_results]),
            }
        
        # Layer-wise scores
        metrics["layer_scores"] = {
            "skeleton": {
                "mean": np.mean([r.skeleton_score for r in self.results]),
                "std": np.std([r.skeleton_score for r in self.results]),
            },
            "motion": {
                "mean": np.mean([r.motion_score for r in self.results]),
                "std": np.std([r.motion_score for r in self.results]),
            },
            "deep_learning": {
                "mean": np.mean([r.dl_score for r in self.results]),
                "std": np.std([r.dl_score for r in self.results]),
            }
        }
        
        return metrics
    
    def generate_confusion_matrix_plot(self, metrics: Dict):
        """Generate confusion matrix visualization."""
        cm = np.array(metrics["confusion_matrix"])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Fall', 'Fall'],
                    yticklabels=['No Fall', 'Fall'],
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Hybrid Fall Detection System', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated: confusion_matrix.png")
    
    def generate_roc_curve(self):
        """Generate ROC curve."""
        y_true = [int(r.expected_fall) for r in self.results]
        y_scores = [r.confidence for r in self.results]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Hybrid System (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
        plt.title('ROC Curve - Fall Detection Performance', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated: roc_curve.png (AUC = {roc_auc:.3f})")
        
        return roc_auc
    
    def generate_precision_recall_curve(self):
        """Generate Precision-Recall curve."""
        y_true = [int(r.expected_fall) for r in self.results]
        y_scores = [r.confidence for r in self.results]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Hybrid System (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall (Sensitivity)', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve - Fall Detection', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(self.output_dir / "precision_recall_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated: precision_recall_curve.png (AUC = {pr_auc:.3f})")
    
    def generate_layer_comparison(self, metrics: Dict):
        """Generate layer-wise performance comparison."""
        # Aggregate scores by category and layer
        categories = sorted(set(r.category for r in self.results))
        layers = ['Skeleton', 'Motion', 'Deep Learning', 'Hybrid']
        
        data = {cat: {layer: [] for layer in layers} for cat in categories}
        
        for result in self.results:
            data[result.category]['Skeleton'].append(result.skeleton_score)
            data[result.category]['Motion'].append(result.motion_score)
            data[result.category]['Deep Learning'].append(result.dl_score)
            data[result.category]['Hybrid'].append(result.confidence)
        
        # Calculate means
        means = {cat: {layer: np.mean(scores) for layer, scores in layers_data.items()} 
                for cat, layers_data in data.items()}
        
        # Plot grouped bar chart
        x = np.arange(len(categories))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, layer in enumerate(layers):
            values = [means[cat][layer] for cat in categories]
            ax.bar(x + i * width, values, width, label=layer, alpha=0.8)
        
        ax.set_xlabel('Video Category', fontsize=14)
        ax.set_ylabel('Average Detection Score', fontsize=14)
        ax.set_title('Layer-wise Performance Comparison Across Categories', 
                     fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "layer_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated: layer_comparison.png")
    
    def generate_performance_metrics_chart(self, metrics: Dict):
        """Generate overall performance metrics bar chart."""
        metric_names = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 
                       'F1-Score', 'Specificity']
        metric_values = [
            metrics['overall']['accuracy'],
            metrics['overall']['precision'],
            metrics['overall']['recall'],
            metrics['overall']['f1_score'],
            metrics['overall']['specificity']
        ]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = sns.color_palette("husl", len(metric_names))
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Overall Performance Metrics - Hybrid Fall Detection System', 
                     fontsize=16, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated: performance_metrics.png")
    
    def generate_processing_time_analysis(self, metrics: Dict):
        """Generate processing time analysis."""
        categories = sorted(set(r.category for r in self.results))
        
        # Processing times by category
        times_by_category = {cat: [r.processing_time for r in self.results if r.category == cat]
                            for cat in categories}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot
        ax1.boxplot([times_by_category[cat] for cat in categories], 
                   labels=[cat.replace('_', ' ').title() for cat in categories])
        ax1.set_ylabel('Processing Time (seconds)', fontsize=14)
        ax1.set_title('Processing Time Distribution by Category', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Average processing time bar chart
        avg_times = [np.mean(times_by_category[cat]) for cat in categories]
        ax2.bar(range(len(categories)), avg_times, color='skyblue', alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Video Category', fontsize=14)
        ax2.set_ylabel('Average Processing Time (seconds)', fontsize=14)
        ax2.set_title('Average Processing Time by Category', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], 
                           rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avg_times):
            ax2.text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "processing_time_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated: processing_time_analysis.png")
    
    def generate_category_accuracy_chart(self, metrics: Dict):
        """Generate per-category accuracy chart."""
        categories = sorted(metrics['per_category'].keys())
        accuracies = [metrics['per_category'][cat]['accuracy'] for cat in categories]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['#2ecc71' if acc == 1.0 else '#3498db' for acc in accuracies]
        bars = ax.bar([cat.replace('_', ' ').title() for cat in categories], 
                     accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_title('Classification Accuracy by Video Category', fontsize=16, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Accuracy')
        plt.xticks(rotation=15, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "category_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated: category_accuracy.png")
    
    def export_results(self, metrics: Dict):
        """Export results to CSV and JSON."""
        # Export detailed results to CSV
        csv_path = self.output_dir / "detailed_results.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.to_dict())
        
        print(f"✅ Exported: detailed_results.csv ({len(self.results)} rows)")
        
        # Export metrics to JSON
        json_path = self.output_dir / "metrics_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Exported: metrics_summary.json")
        
        # Export summary report
        self.generate_text_report(metrics)
    
    def generate_text_report(self, metrics: Dict):
        """Generate comprehensive text report."""
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SMARTCARE+ Fall Detection System - Evaluation Report\n")
            f.write("Three-Layer Hybrid Approach (Skeleton + Motion + Deep Learning)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Videos Tested: {metrics['total_videos']}\n")
            f.write(f"Accuracy:           {metrics['overall']['accuracy']:.2%}\n")
            f.write(f"Precision:          {metrics['overall']['precision']:.2%}\n")
            f.write(f"Recall/Sensitivity: {metrics['overall']['recall']:.2%}\n")
            f.write(f"F1-Score:           {metrics['overall']['f1_score']:.2%}\n")
            f.write(f"Specificity:        {metrics['overall']['specificity']:.2%}\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")
            cm = np.array(metrics['confusion_matrix'])
            f.write(f"True Negatives:  {cm[0][0]}\n")
            f.write(f"False Positives: {cm[0][1]}\n")
            f.write(f"False Negatives: {cm[1][0]}\n")
            f.write(f"True Positives:  {cm[1][1]}\n\n")
            
            f.write("PER-CATEGORY PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            for category, data in sorted(metrics['per_category'].items()):
                f.write(f"{category.upper():<20} Accuracy: {data['accuracy']:.2%} "
                       f"({data['correct']}/{data['total']} correct)\n")
                f.write(f"{'':20} Avg Confidence: {data['avg_confidence']:.2%}\n")
                f.write(f"{'':20} Avg Processing Time: {data['avg_processing_time']:.2f}s\n")
            
            f.write("\nLAYER-WISE AVERAGE SCORES\n")
            f.write("-" * 40 + "\n")
            for layer, scores in metrics['layer_scores'].items():
                f.write(f"{layer.upper():<20} Mean: {scores['mean']:.3f}, "
                       f"Std Dev: {scores['std']:.3f}\n")
            
            f.write("\nPROCESSING TIME STATISTICS\n")
            f.write("-" * 40 + "\n")
            pt = metrics['processing_times']
            f.write(f"Mean:    {pt['mean']:.2f} seconds\n")
            f.write(f"Std Dev: {pt['std']:.2f} seconds\n")
            f.write(f"Min:     {pt['min']:.2f} seconds\n")
            f.write(f"Max:     {pt['max']:.2f} seconds\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Generated: evaluation_report.txt")
    
    def run_complete_evaluation(self, max_per_category: int = None):
        """Run complete evaluation with all analyses."""
        # Run tests
        self.run_evaluation(max_per_category)
        
        # Calculate metrics
        print("\n📊 Calculating performance metrics...")
        metrics = self.calculate_metrics()
        
        # Generate visualizations
        print("\n📈 Generating visualizations...")
        self.generate_confusion_matrix_plot(metrics)
        self.generate_roc_curve()
        self.generate_precision_recall_curve()
        self.generate_performance_metrics_chart(metrics)
        self.generate_layer_comparison(metrics)
        self.generate_category_accuracy_chart(metrics)
        self.generate_processing_time_analysis(metrics)
        
        # Export results
        print("\n💾 Exporting results...")
        self.export_results(metrics)
        
        print(f"\n✅ Complete! All results saved to: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        print("  📊 Metrics & Data:")
        print("     - detailed_results.csv")
        print("     - metrics_summary.json")
        print("     - evaluation_report.txt")
        print("  📈 Visualizations:")
        print("     - confusion_matrix.png")
        print("     - roc_curve.png")
        print("     - precision_recall_curve.png")
        print("     - performance_metrics.png")
        print("     - layer_comparison.png")
        print("     - category_accuracy.png")
        print("     - processing_time_analysis.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of fall detection system for thesis/publication"
    )
    parser.add_argument("--output", "-o", type=str, default="evaluation_results",
                       help="Output directory for results and graphs")
    parser.add_argument("--max-per-category", "-n", type=int, default=None,
                       help="Maximum videos to test per category (default: all)")
    
    args = parser.parse_args()
    
    evaluator = FallDetectionEvaluator(output_dir=args.output)
    evaluator.run_complete_evaluation(max_per_category=args.max_per_category)
