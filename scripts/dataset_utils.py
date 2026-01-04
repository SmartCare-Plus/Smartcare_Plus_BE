#!/usr/bin/env python3
"""
Dataset Loader and Inspector Utility
=====================================
Utility functions for loading and inspecting processed pose datasets.

Usage:
------
# As a module
from scripts.dataset_utils import load_dataset, get_train_test_split

# As a standalone script
python scripts/dataset_utils.py --dataset ml_models/guardian/guardian_dataset.pkl --info
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def load_dataset(dataset_path: str) -> Tuple[List[Tuple[np.ndarray, str]], Dict[str, Any]]:
    """
    Load a processed pose dataset from a pickle file.
    
    Args:
        dataset_path: Path to the pickle file
        
    Returns:
        Tuple of (sequences list, metadata dict)
    """
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    sequences = data['sequences']
    metadata = data['metadata']
    
    return sequences, metadata


def get_train_test_split(
    sequences: List[Tuple[np.ndarray, str]],
    test_ratio: float = 0.2,
    shuffle: bool = True,
    random_seed: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing sets.
    
    Args:
        sequences: List of (sequence, label) tuples
        test_ratio: Proportion of data for testing
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        stratify: Whether to maintain class balance
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays
    """
    np.random.seed(random_seed)
    
    # Separate data and labels
    X = np.array([seq for seq, _ in sequences])
    y = np.array([label for _, label in sequences])
    
    if stratify:
        # Stratified split
        unique_labels = np.unique(y)
        train_indices = []
        test_indices = []
        
        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            if shuffle:
                np.random.shuffle(label_indices)
            
            n_test = max(1, int(len(label_indices) * test_ratio))
            test_indices.extend(label_indices[:n_test])
            train_indices.extend(label_indices[n_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
    else:
        # Simple random split
        indices = np.arange(len(sequences))
        if shuffle:
            np.random.shuffle(indices)
        
        n_test = int(len(indices) * test_ratio)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Encode string labels to integers.
    
    Args:
        labels: Array of string labels
        
    Returns:
        Tuple of (encoded labels, label_to_index dict, index_to_label dict)
    """
    unique_labels = sorted(np.unique(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    
    encoded = np.array([label_to_index[label] for label in labels])
    
    return encoded, label_to_index, index_to_label


def one_hot_encode(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode integer labels.
    
    Args:
        labels: Array of integer labels
        num_classes: Number of classes (inferred if None)
        
    Returns:
        One-hot encoded array
    """
    if num_classes is None:
        num_classes = len(np.unique(labels))
    
    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    one_hot[np.arange(len(labels)), labels] = 1.0
    
    return one_hot


def get_dataset_info(metadata: Dict[str, Any]) -> str:
    """
    Format dataset metadata as a readable string.
    
    Args:
        metadata: Dataset metadata dictionary
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        "DATASET INFORMATION",
        "=" * 60,
        f"Created:          {metadata['created_at']}",
        f"Input Directory:  {metadata['input_directory']}",
        f"Output File:      {metadata['output_file']}",
        "",
        "STATISTICS:",
        f"  Total Videos:     {metadata['total_videos']}",
        f"  Total Sequences:  {metadata['total_sequences']}",
        f"  Sequence Length:  {metadata['sequence_length']} frames",
        f"  Overlap:          {metadata['overlap']} frames",
        "",
        "FEATURES:",
        f"  Landmarks/Frame:  {metadata['landmarks_per_frame']}",
        f"  Features/Landmark: {metadata['features_per_landmark']}",
        f"  Total Features:   {metadata['total_features_per_frame']}",
        "",
        "CLASS DISTRIBUTION:",
    ]
    
    total = metadata['total_sequences']
    for label, count in sorted(metadata['classes'].items()):
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"  {label:20s}: {count:5d} ({pct:.1f}%)")
    
    if metadata.get('failed_videos'):
        lines.append("")
        lines.append(f"FAILED VIDEOS ({len(metadata['failed_videos'])}):")
        for video in metadata['failed_videos']:
            lines.append(f"  - {video}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def get_sequence_shape_info(sequences: List[Tuple[np.ndarray, str]]) -> str:
    """
    Get information about sequence shapes.
    
    Args:
        sequences: List of (sequence, label) tuples
        
    Returns:
        Formatted string
    """
    if not sequences:
        return "No sequences available"
    
    sample_seq = sequences[0][0]
    
    lines = [
        "SEQUENCE SHAPE INFO:",
        f"  Number of sequences: {len(sequences)}",
        f"  Sequence shape:      {sample_seq.shape}",
        f"  Data type:           {sample_seq.dtype}",
        f"  Memory per sequence: {sample_seq.nbytes / 1024:.2f} KB",
        f"  Total memory:        {sum(s[0].nbytes for s in sequences) / (1024*1024):.2f} MB",
    ]
    
    return "\n".join(lines)


def print_sample_sequence(sequence: np.ndarray, n_frames: int = 3) -> None:
    """
    Print sample frames from a sequence.
    
    Args:
        sequence: Pose sequence array
        n_frames: Number of frames to print
    """
    print(f"\nSample Sequence (first {n_frames} frames):")
    print("-" * 40)
    
    for i in range(min(n_frames, len(sequence))):
        frame = sequence[i]
        print(f"Frame {i}: shape={frame.shape}")
        # Print first 8 values (x,y,z,vis for first 2 landmarks)
        print(f"  First 8 values: {frame[:8]}")
        print(f"  Min: {frame.min():.4f}, Max: {frame.max():.4f}, Mean: {frame.mean():.4f}")


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Load and inspect processed pose datasets'
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        required=True,
        help='Path to the processed dataset pickle file'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Print dataset information'
    )
    parser.add_argument(
        '--shapes',
        action='store_true',
        help='Print sequence shape information'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Print sample sequence data'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Show train/test split information'
    )
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        return
    
    sequences, metadata = load_dataset(args.dataset)
    
    if args.info or not (args.shapes or args.sample or args.split):
        print(get_dataset_info(metadata))
    
    if args.shapes:
        print(get_sequence_shape_info(sequences))
    
    if args.sample and sequences:
        sample_seq, sample_label = sequences[0]
        print(f"\nSample Label: '{sample_label}'")
        print_sample_sequence(sample_seq)
    
    if args.split:
        X_train, X_test, y_train, y_test = get_train_test_split(sequences)
        print("\nTRAIN/TEST SPLIT:")
        print(f"  Training samples:   {len(X_train)}")
        print(f"  Testing samples:    {len(X_test)}")
        print(f"  Training shape:     {X_train.shape}")
        print(f"  Testing shape:      {X_test.shape}")
        
        print("\n  Training class distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"    {label}: {count}")
        
        print("\n  Testing class distribution:")
        unique, counts = np.unique(y_test, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"    {label}: {count}")


if __name__ == '__main__':
    main()
