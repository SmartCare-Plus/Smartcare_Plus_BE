#!/usr/bin/env python3
"""
Video Dataset Processor for Human Action Recognition
=====================================================
Converts video files to pose landmark sequences for CNN+LSTM training.

This script processes video files organized in class-labeled folders,
extracts pose landmarks using MediaPipe, and creates fixed-length
sequences suitable for deep learning models.

Author: SmartCare+ Team
Date: January 2026

Usage Examples:
---------------
# Guardian Service (Fall Detection) - Madhushani
python scripts/process_video_dataset.py \
    --input media/simulation_footage/guardian \
    --output ml_models/guardian/guardian_dataset.pkl \
    --sequence-length 30 \
    --overlap 15

# Physio Service (Gait Analysis) - Neelaka
python scripts/process_video_dataset.py \
    --input media/simulation_footage/physio \
    --output ml_models/physio/physio_dataset.pkl \
    --sequence-length 45 \
    --overlap 20
"""

import argparse
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, asdict
import json

import cv2
import numpy as np

# Try new MediaPipe API first, fall back to legacy
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    USE_LEGACY_API = False
except ImportError:
    import mediapipe as mp
    USE_LEGACY_API = True

# Check if legacy API is available
if not USE_LEGACY_API:
    try:
        _ = mp.solutions.pose
        USE_LEGACY_API = True
    except AttributeError:
        USE_LEGACY_API = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline."""
    input_dir: str
    output_file: str
    sequence_length: int = 30
    overlap: int = 15
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv')
    normalize_landmarks: bool = True
    include_visibility: bool = True
    skip_incomplete_sequences: bool = False
    augment_data: bool = False
    max_videos_per_class: Optional[int] = None


@dataclass
class DatasetMetadata:
    """Metadata about the processed dataset."""
    created_at: str
    input_directory: str
    output_file: str
    sequence_length: int
    overlap: int
    total_videos: int
    total_sequences: int
    classes: Dict[str, int]
    landmarks_per_frame: int
    features_per_landmark: int
    total_features_per_frame: int
    processing_config: Dict[str, Any]
    failed_videos: List[str]


class PoseExtractor:
    """Extracts pose landmarks from video frames using MediaPipe."""
    
    # MediaPipe Pose landmark indices for reference
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    NUM_LANDMARKS = 33
    FEATURES_PER_LANDMARK = 4  # x, y, z, visibility
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        normalize: bool = True,
        include_visibility: bool = True
    ):
        """
        Initialize the pose extractor.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            normalize: Whether to normalize landmarks to [0, 1] range
            include_visibility: Whether to include visibility scores
        """
        self.use_legacy = USE_LEGACY_API
        self.normalize = normalize
        self.include_visibility = include_visibility
        self.features_per_landmark = 4 if include_visibility else 3
        self.total_features = self.NUM_LANDMARKS * self.features_per_landmark
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        if self.use_legacy:
            # Legacy MediaPipe API (pre-0.10.14)
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.landmarker = None
        else:
            # New MediaPipe Tasks API (0.10.14+)
            self.pose = None
            self._init_new_api()
    
    def _init_new_api(self):
        """Initialize the new MediaPipe Tasks API."""
        import urllib.request
        import os
        
        # Download model if not exists
        model_path = Path(__file__).parent / 'pose_landmarker_full.task'
        if not model_path.exists():
            logger.info("Downloading MediaPipe Pose Landmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
            urllib.request.urlretrieve(model_url, str(model_path))
            logger.info(f"Model downloaded to {model_path}")
        
        # Create pose landmarker
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            num_poses=1
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[List[float]]:
        """
        Extract pose landmarks from a single frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of flattened landmark coordinates, or None if no pose detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.use_legacy:
            # Legacy API
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks is None:
                return None
            
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])
                if self.include_visibility:
                    landmarks.append(landmark.visibility)
        else:
            # New Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self.frame_timestamp_ms += 33  # Assume ~30fps
            
            results = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            
            if not results.pose_landmarks or len(results.pose_landmarks) == 0:
                return None
            
            landmarks = []
            pose_landmarks = results.pose_landmarks[0]  # First detected pose
            for landmark in pose_landmarks:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])
                if self.include_visibility:
                    # New API uses 'visibility' attribute or default to 1.0
                    vis = getattr(landmark, 'visibility', 1.0) or 1.0
                    landmarks.append(vis)
        
        return landmarks
    
    def extract_from_video(
        self, 
        video_path: str,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Extract pose landmarks from all frames of a video.
        
        Args:
            video_path: Path to the video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (list of frame landmarks, video metadata)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_metadata = {
            'path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration_seconds': total_frames / fps if fps > 0 else 0
        }
        
        all_landmarks = []
        frames_processed = 0
        frames_with_pose = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.extract_from_frame(frame)
            
            if landmarks is not None:
                all_landmarks.append(landmarks)
                frames_with_pose += 1
            else:
                # Append zeros for frames without detected pose
                all_landmarks.append([0.0] * self.total_features)
            
            frames_processed += 1
            
            if progress_callback and frames_processed % 30 == 0:
                progress_callback(frames_processed, total_frames)
        
        cap.release()
        
        video_metadata['frames_processed'] = frames_processed
        video_metadata['frames_with_pose'] = frames_with_pose
        video_metadata['detection_rate'] = frames_with_pose / frames_processed if frames_processed > 0 else 0
        
        return all_landmarks, video_metadata
    
    def close(self):
        """Release MediaPipe resources."""
        if self.use_legacy and self.pose:
            self.pose.close()
        elif self.landmarker:
            self.landmarker.close()


class SequenceCreator:
    """Creates fixed-length sequences from frame data with optional augmentation."""
    
    def __init__(
        self,
        sequence_length: int = 30,
        overlap: int = 15,
        skip_incomplete: bool = False,
        augment: bool = False
    ):
        """
        Initialize the sequence creator.
        
        Args:
            sequence_length: Number of frames per sequence
            overlap: Number of overlapping frames between consecutive sequences
            skip_incomplete: Whether to skip sequences that don't have enough frames
            augment: Whether to apply data augmentation
        """
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.skip_incomplete = skip_incomplete
        self.augment = augment
        self.step = sequence_length - overlap
    
    def create_sequences(
        self, 
        frame_data: List[List[float]], 
        label: str
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Create fixed-length sequences from frame data.
        
        Args:
            frame_data: List of frame landmarks
            label: Class label for the sequences
            
        Returns:
            List of (sequence, label) tuples
        """
        sequences = []
        num_frames = len(frame_data)
        
        if num_frames < self.sequence_length:
            if self.skip_incomplete:
                logger.warning(
                    f"Skipping video with only {num_frames} frames "
                    f"(need {self.sequence_length})"
                )
                return sequences
            else:
                # Pad with zeros if not enough frames
                padding_needed = self.sequence_length - num_frames
                frame_data = frame_data + [[0.0] * len(frame_data[0])] * padding_needed
                num_frames = len(frame_data)
        
        # Create overlapping sequences
        for start_idx in range(0, num_frames - self.sequence_length + 1, self.step):
            end_idx = start_idx + self.sequence_length
            sequence = np.array(frame_data[start_idx:end_idx], dtype=np.float32)
            sequences.append((sequence, label))
            
            # Apply augmentation if enabled
            if self.augment:
                augmented = self._augment_sequence(sequence)
                for aug_seq in augmented:
                    sequences.append((aug_seq, label))
        
        return sequences
    
    def _augment_sequence(self, sequence: np.ndarray) -> List[np.ndarray]:
        """
        Apply data augmentation to a sequence.
        
        Args:
            sequence: Original sequence data
            
        Returns:
            List of augmented sequences
        """
        augmented = []
        
        # Horizontal flip (mirror x-coordinates)
        flipped = sequence.copy()
        # Flip x-coordinates (every 4th value starting from 0)
        flipped[:, 0::4] = 1.0 - flipped[:, 0::4]
        augmented.append(flipped)
        
        # Add small noise
        noisy = sequence.copy()
        noise = np.random.normal(0, 0.01, sequence.shape).astype(np.float32)
        noisy += noise
        noisy = np.clip(noisy, 0, 1)
        augmented.append(noisy)
        
        # Time scaling (speed up/slow down) - interpolation
        # Speed up by taking every other frame and duplicating
        if len(sequence) >= 2:
            fast = sequence[::2]
            fast = np.repeat(fast, 2, axis=0)[:len(sequence)]
            augmented.append(fast)
        
        return augmented


class VideoDatasetProcessor:
    """Main processor for converting video datasets to trainable sequences."""
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the dataset processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.pose_extractor = PoseExtractor(
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            normalize=config.normalize_landmarks,
            include_visibility=config.include_visibility
        )
        self.sequence_creator = SequenceCreator(
            sequence_length=config.sequence_length,
            overlap=config.overlap,
            skip_incomplete=config.skip_incomplete_sequences,
            augment=config.augment_data
        )
        self.failed_videos: List[str] = []
    
    def discover_videos(self) -> Dict[str, List[str]]:
        """
        Discover all video files organized by class label.
        
        Returns:
            Dictionary mapping class labels to list of video paths
        """
        input_path = Path(self.config.input_dir)
        
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")
        
        videos_by_class = {}
        
        for class_folder in sorted(input_path.iterdir()):
            if not class_folder.is_dir():
                continue
            
            class_label = class_folder.name
            video_files = set()  # Use set to avoid duplicates on case-insensitive filesystems
            
            for ext in self.config.video_extensions:
                # Use case-insensitive glob pattern
                for video in class_folder.glob(f"*{ext}"):
                    video_files.add(str(video))
                for video in class_folder.glob(f"*{ext.upper()}"):
                    video_files.add(str(video))
            
            video_files = sorted(list(video_files))
            
            # Apply max videos limit if specified
            if self.config.max_videos_per_class:
                video_files = video_files[:self.config.max_videos_per_class]
            
            if video_files:
                videos_by_class[class_label] = video_files
                logger.info(f"Found {len(video_files)} videos for class '{class_label}'")
        
        if not videos_by_class:
            raise ValueError(f"No video files found in {input_path}")
        
        return videos_by_class
    
    def process_video(
        self, 
        video_path: str, 
        label: str
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Process a single video file.
        
        Args:
            video_path: Path to the video file
            label: Class label for this video
            
        Returns:
            List of (sequence, label) tuples
        """
        try:
            # Extract landmarks from video
            frame_data, video_meta = self.pose_extractor.extract_from_video(video_path)
            
            logger.debug(
                f"Processed {video_meta['frames_processed']} frames, "
                f"detection rate: {video_meta['detection_rate']:.1%}"
            )
            
            # Create sequences
            sequences = self.sequence_creator.create_sequences(frame_data, label)
            
            return sequences
            
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            self.failed_videos.append(video_path)
            return []
    
    def process_dataset(self) -> Tuple[List[Tuple[np.ndarray, str]], DatasetMetadata]:
        """
        Process the entire dataset.
        
        Returns:
            Tuple of (all sequences, dataset metadata)
        """
        logger.info("=" * 60)
        logger.info("Starting Video Dataset Processing")
        logger.info("=" * 60)
        
        # Discover videos
        videos_by_class = self.discover_videos()
        total_videos = sum(len(v) for v in videos_by_class.values())
        
        logger.info(f"Total classes: {len(videos_by_class)}")
        logger.info(f"Total videos to process: {total_videos}")
        logger.info(f"Sequence length: {self.config.sequence_length} frames")
        logger.info(f"Overlap: {self.config.overlap} frames")
        logger.info("=" * 60)
        
        all_sequences = []
        class_counts = {}
        videos_processed = 0
        
        for class_label, video_paths in videos_by_class.items():
            logger.info(f"\nProcessing class: '{class_label}' ({len(video_paths)} videos)")
            class_sequences = 0
            
            for idx, video_path in enumerate(video_paths, 1):
                video_name = Path(video_path).name
                logger.info(f"  [{idx}/{len(video_paths)}] Processing: {video_name}")
                
                sequences = self.process_video(video_path, class_label)
                all_sequences.extend(sequences)
                class_sequences += len(sequences)
                videos_processed += 1
                
                logger.info(f"    Created {len(sequences)} sequences")
            
            class_counts[class_label] = class_sequences
            logger.info(f"  Total sequences for '{class_label}': {class_sequences}")
        
        # Create metadata
        metadata = DatasetMetadata(
            created_at=datetime.now().isoformat(),
            input_directory=str(self.config.input_dir),
            output_file=str(self.config.output_file),
            sequence_length=self.config.sequence_length,
            overlap=self.config.overlap,
            total_videos=videos_processed,
            total_sequences=len(all_sequences),
            classes=class_counts,
            landmarks_per_frame=PoseExtractor.NUM_LANDMARKS,
            features_per_landmark=self.pose_extractor.features_per_landmark,
            total_features_per_frame=self.pose_extractor.total_features,
            processing_config=asdict(self.config),
            failed_videos=self.failed_videos
        )
        
        return all_sequences, metadata
    
    def save_dataset(
        self, 
        sequences: List[Tuple[np.ndarray, str]], 
        metadata: DatasetMetadata
    ) -> None:
        """
        Save the processed dataset to a pickle file.
        
        Args:
            sequences: List of (sequence, label) tuples
            metadata: Dataset metadata
        """
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        dataset = {
            'sequences': sequences,
            'metadata': asdict(metadata)
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"\nDataset saved to: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Save metadata as JSON for easy inspection
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def close(self):
        """Release resources."""
        self.pose_extractor.close()


def print_summary(metadata: DatasetMetadata) -> None:
    """Print a summary of the processed dataset."""
    print("\n" + "=" * 60)
    print("DATASET PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n📁 Input Directory:  {metadata.input_directory}")
    print(f"💾 Output File:      {metadata.output_file}")
    print(f"\n📊 STATISTICS:")
    print(f"   Total Videos:     {metadata.total_videos}")
    print(f"   Total Sequences:  {metadata.total_sequences}")
    print(f"   Sequence Length:  {metadata.sequence_length} frames")
    print(f"   Features/Frame:   {metadata.total_features_per_frame}")
    print(f"\n🏷️  CLASS DISTRIBUTION:")
    for label, count in sorted(metadata.classes.items()):
        percentage = (count / metadata.total_sequences * 100) if metadata.total_sequences > 0 else 0
        print(f"   {label:20s}: {count:5d} sequences ({percentage:.1f}%)")
    
    if metadata.failed_videos:
        print(f"\n⚠️  FAILED VIDEOS ({len(metadata.failed_videos)}):")
        for video in metadata.failed_videos:
            print(f"   - {video}")
    
    print("\n" + "=" * 60)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Process video dataset for Human Action Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Process Guardian (Fall Detection) dataset
  python scripts/process_video_dataset.py \\
      --input media/simulation_footage/guardian \\
      --output ml_models/guardian/guardian_dataset.pkl

  # Process Physio (Gait Analysis) dataset with custom settings
  python scripts/process_video_dataset.py \\
      --input media/simulation_footage/physio \\
      --output ml_models/physio/physio_dataset.pkl \\
      --sequence-length 45 \\
      --overlap 20 \\
      --augment

  # Quick test with limited videos
  python scripts/process_video_dataset.py \\
      --input media/simulation_footage/guardian \\
      --output ml_models/guardian/test_dataset.pkl \\
      --max-videos 5 \\
      --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing class subfolders with videos'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output pickle file path for the processed dataset'
    )
    
    # Sequence parameters
    parser.add_argument(
        '-s', '--sequence-length',
        type=int,
        default=30,
        help='Number of frames per sequence (default: 30)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=15,
        help='Number of overlapping frames between sequences (default: 15)'
    )
    
    # MediaPipe parameters
    parser.add_argument(
        '--detection-confidence',
        type=float,
        default=0.5,
        help='Minimum detection confidence for MediaPipe (default: 0.5)'
    )
    parser.add_argument(
        '--tracking-confidence',
        type=float,
        default=0.5,
        help='Minimum tracking confidence for MediaPipe (default: 0.5)'
    )
    
    # Processing options
    parser.add_argument(
        '--no-visibility',
        action='store_true',
        help='Exclude visibility scores (use only x, y, z coordinates)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip landmark normalization'
    )
    parser.add_argument(
        '--skip-incomplete',
        action='store_true',
        help='Skip videos shorter than sequence length'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Apply data augmentation (flip, noise, time scaling)'
    )
    parser.add_argument(
        '--max-videos',
        type=int,
        default=None,
        help='Maximum videos per class (for testing)'
    )
    
    # Verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose/debug logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = ProcessingConfig(
        input_dir=args.input,
        output_file=args.output,
        sequence_length=args.sequence_length,
        overlap=args.overlap,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        normalize_landmarks=not args.no_normalize,
        include_visibility=not args.no_visibility,
        skip_incomplete_sequences=args.skip_incomplete,
        augment_data=args.augment,
        max_videos_per_class=args.max_videos
    )
    
    # Process dataset
    processor = VideoDatasetProcessor(config)
    
    try:
        sequences, metadata = processor.process_dataset()
        
        if not sequences:
            logger.error("No sequences were created. Check your input videos.")
            sys.exit(1)
        
        processor.save_dataset(sequences, metadata)
        print_summary(metadata)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
    finally:
        processor.close()
    
    logger.info("Processing completed successfully!")


if __name__ == '__main__':
    main()
