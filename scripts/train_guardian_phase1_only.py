#!/usr/bin/env python3
"""
🎬 SmartCare+ Guardian: Phase 1 Only Training Script

Trains ONLY the classification head (EfficientNetV2-S backbone frozen).
This is a memory-efficient script that avoids the OOM crash in Phase 2.

Phase 1 achieved 85.7% validation accuracy - sufficient for demo purposes.

Usage (Kaggle):
    Set KAGGLE_MODE = True, run all cells.

Author: Madhushani (Guardian Service)
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict

# ═══════════════════════════════════════════════════════════════════════════════
# DISABLE PROBLEMATIC OPTIMIZER (Must be set BEFORE importing TensorFlow)
# ═══════════════════════════════════════════════════════════════════════════════
os.environ['TF_ENABLE_LAYOUT_OPTIMIZER'] = '0'
os.environ['TF_ENABLE_GRAPPLER_OPTIMIZERS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import tensorflow as tf

# Disable layout optimizer
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False,
    'constant_folding': True,
    'shape_optimization': True,
    'remapping': True,
    'arithmetic_optimization': True,
    'dependency_optimization': True,
    'loop_optimization': True,
    'function_optimization': True,
    'debug_stripper': True,
})
print("✅ TensorFlow layout optimizer disabled")

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️ GPU memory growth error: {e}")

# Pre-download weights
print("📥 Pre-downloading EfficientNetV2-S weights...")
try:
    from tensorflow.keras.applications import EfficientNetV2S
    _temp = EfficientNetV2S(include_top=False, weights='imagenet', 
                            input_shape=(224, 224, 3), pooling='avg')
    del _temp
    import gc
    gc.collect()
    print("✅ EfficientNetV2-S weights cached successfully!")
except Exception as e:
    print(f"⚠️ Weight pre-download failed: {e}")

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2S
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - PHASE 1 ONLY
# ═══════════════════════════════════════════════════════════════════════════════

KAGGLE_MODE = True  # Set to True for Kaggle notebooks

CONFIG = {
    # Paths
    "dataset_path": "/kaggle/input/smartcare-guardian-videos",
    "model_save_path": "/kaggle/working",
    "checkpoint_path": "/kaggle/working/checkpoints",
    
    # Video processing
    "num_frames": 32,
    "frame_height": 224,
    "frame_width": 224,
    "channels": 3,
    
    # Training - PHASE 1 ONLY with batch_size=4 (proven stable)
    "batch_size": 4,  # ✅ Works for Phase 1
    "epochs_phase1": 20,  # Increased from 15 to 20
    "learning_rate_phase1": 1e-3,
    
    # Data split
    "test_split": 0.15,
    "val_split": 0.15,
    "random_seed": 42,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def discover_dataset(dataset_path: str) -> Tuple[List[str], List[int], Dict[str, int], Dict[int, str]]:
    """Scan dataset directory and create file lists with labels."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    class_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    if len(class_folders) == 0:
        raise ValueError(f"No class folders found in: {dataset_path}")
    
    class_to_idx = {folder.name: idx for idx, folder in enumerate(class_folders)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    video_paths = []
    labels = []
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    
    for class_folder in class_folders:
        class_idx = class_to_idx[class_folder.name]
        for video_file in class_folder.iterdir():
            if video_file.suffix in video_extensions:
                video_paths.append(str(video_file))
                labels.append(class_idx)
    
    return video_paths, labels, class_to_idx, idx_to_class


def load_video_frames(video_path: str, num_frames: int, height: int, width: int) -> np.ndarray:
    """Load and uniformly sample frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return np.zeros((num_frames, height, width, 3), dtype=np.float32)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames))
        if total_frames > 0:
            indices.extend([total_frames - 1] * (num_frames - total_frames))
        else:
            indices = [0] * num_frames
        indices = np.array(indices[:num_frames])
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        else:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((height, width, 3), dtype=np.float32))
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((height, width, 3), dtype=np.float32))
    
    return np.array(frames[:num_frames], dtype=np.float32)


def apply_augmentation(frames: np.ndarray, training: bool = True) -> np.ndarray:
    """Apply data augmentation to video frames."""
    if not training:
        return frames
    
    frames = frames.copy()
    
    if random.random() > 0.5:
        frames = frames[:, :, ::-1, :]
    
    brightness = random.uniform(0.8, 1.2)
    frames = np.clip(frames * brightness, 0, 1)
    
    contrast = random.uniform(0.8, 1.2)
    mean = np.mean(frames, axis=(1, 2, 3), keepdims=True)
    frames = np.clip((frames - mean) * contrast + mean, 0, 1)
    
    if random.random() > 0.5:
        shift = random.randint(-2, 2)
        if shift != 0:
            frames = np.roll(frames, shift, axis=0)
    
    return frames.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════

class VideoDataSequence(tf.keras.utils.Sequence):
    """Keras Sequence for loading video batches."""
    
    def __init__(self, video_paths: List[str], labels: List[int], config: dict, is_training: bool = True):
        self.video_paths = np.array(video_paths)
        self.labels = np.array(labels, dtype=np.int32)
        self.config = config
        self.is_training = is_training
        
        self.batch_size = config["batch_size"]
        self.num_frames = config["num_frames"]
        self.height = config["frame_height"]
        self.width = config["frame_width"]
        
        self.indices = np.arange(len(self.video_paths))
        
        if self.is_training:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        return len(self.video_paths) // self.batch_size
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start_idx = idx * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_frames = []
        batch_labels = []
        
        for i in batch_indices:
            frames = load_video_frames(
                self.video_paths[i], 
                self.num_frames, 
                self.height, 
                self.width
            )
            
            expected_shape = (self.num_frames, self.height, self.width, 3)
            if frames.shape != expected_shape:
                frames = np.zeros(expected_shape, dtype=np.float32)
            
            if self.is_training:
                frames = apply_augmentation(frames, training=True)
            
            batch_frames.append(frames)
            batch_labels.append(self.labels[i])
        
        batch_frames = np.stack(batch_frames, axis=0).astype(np.float32)
        batch_labels = np.array(batch_labels, dtype=np.int32)
        
        return batch_frames, batch_labels
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def build_video_backbone(config: dict) -> models.Model:
    """Build video classification backbone using EfficientNetV2-S + LSTM."""
    print("\n📦 Building Video Classification Backbone...")
    
    input_shape = (config["num_frames"], config["frame_height"],
                   config["frame_width"], config["channels"])
    
    efficientnet = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(config["frame_height"], config["frame_width"], config["channels"]),
        pooling='avg'
    )
    
    # Freeze all layers (Phase 1 = frozen backbone)
    for layer in efficientnet.layers:
        layer.trainable = False
    
    video_input = layers.Input(shape=input_shape, name="video_input")
    frame_features = layers.TimeDistributed(efficientnet, name="frame_encoder")(video_input)
    
    x = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.3),
        name="temporal_lstm_1"
    )(frame_features)
    
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=False, dropout=0.3),
        name="temporal_lstm_2"
    )(x)
    
    features = layers.Dense(512, activation='relu', name="feature_dense")(x)
    features = layers.BatchNormalization(name="feature_bn")(features)
    features = layers.Dropout(0.4, name="feature_dropout")(features)
    
    backbone = models.Model(inputs=video_input, outputs=features, name="video_backbone")
    print(f"✅ Backbone built: {backbone.count_params():,} parameters")
    return backbone


def build_guardian_model(backbone: models.Model, num_classes: int, dropout_rate: float = 0.3) -> models.Model:
    """Build complete Guardian classification model."""
    print(f"\n🏗️ Building Guardian model with {num_classes} classes...")
    
    features = backbone.output
    
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(features)
    x = layers.Dense(256, activation='relu', name="classifier_dense_1")(x)
    x = layers.BatchNormalization(name="classifier_bn")(x)
    x = layers.Dropout(dropout_rate / 2, name="classifier_dropout_2")(x)
    x = layers.Dense(128, activation='relu', name="classifier_dense_2")(x)
    
    outputs = layers.Dense(
        num_classes, activation="softmax", dtype='float32',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name="predictions"
    )(x)
    
    model = models.Model(inputs=backbone.input, outputs=outputs, name="guardian_classifier")
    print(f"✅ Model built: {model.count_params():,} parameters")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN - PHASE 1 ONLY
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main training pipeline - PHASE 1 ONLY."""
    
    config = CONFIG.copy()
    
    print("\n" + "=" * 70)
    print("🎬 SmartCare+ Guardian - PHASE 1 ONLY Training")
    print("=" * 70)
    print("⚠️ This script trains ONLY Phase 1 (frozen backbone)")
    print("   Phase 2 is skipped to avoid RAM OOM issues.\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(config["model_save_path"])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(config["checkpoint_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    tf.random.set_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])
    
    # GPU check
    print(f"✅ TensorFlow {tf.__version__}")
    print(f"✅ GPUs available: {len(gpus)}")
    print(f"\n📁 Dataset: {config['dataset_path']}")
    print(f"📁 Output:  {config['model_save_path']}")
    print(f"🎞️  Frames: {config['num_frames']} @ {config['frame_height']}x{config['frame_width']}")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"🔄 Epochs: {config['epochs_phase1']} (Phase 1 only)")
    
    # ─── Load Dataset ───
    print("\n🔍 Scanning dataset...")
    video_paths, labels, class_to_idx, idx_to_class = discover_dataset(config["dataset_path"])
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total videos: {len(video_paths)}")
    print(f"   Classes: {num_classes}")
    for name, idx in class_to_idx.items():
        count = labels.count(idx)
        print(f"   • {name}: {count} ({100*count/len(labels):.1f}%)")
    
    # Class weights
    class_weight_values = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weight_values))
    
    # Split data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        video_paths, labels, test_size=config["test_split"],
        stratify=labels, random_state=config["random_seed"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=config["val_split"] / (1 - config["test_split"]),
        stratify=y_train_val, random_state=config["random_seed"]
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # ─── Build Model ───
    tf.keras.backend.clear_session()
    backbone = build_video_backbone(config)
    model = build_guardian_model(backbone, num_classes)
    
    # ─── Create Data Sequences ───
    train_sequence = VideoDataSequence(X_train, y_train, config, is_training=True)
    val_sequence = VideoDataSequence(X_val, y_val, config, is_training=False)
    
    steps_per_epoch = len(train_sequence)
    validation_steps = len(val_sequence)
    
    print(f"\n📊 Training Setup:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Validation steps: {validation_steps}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1 ONLY: Train Classification Head (Backbone Frozen)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("🎯 PHASE 1: Training Classification Head (EfficientNet Frozen)")
    print("=" * 70)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate_phase1"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_accuracy')]
    )
    
    # Callbacks - ModelCheckpoint saves the best model
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=1, mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / f"guardian_phase1_{timestamp}.keras"),
            monitor='val_accuracy', save_best_only=True, verbose=1, mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
        )
    ]
    
    print(f"\n   Training for {config['epochs_phase1']} epochs...\n")
    
    history = model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=config["epochs_phase1"],
        callbacks=callbacks,
        verbose=1
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Phase 1 Complete!")
    print(f"   Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE MODEL IN MULTIPLE FORMATS (for compatibility)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n💾 Saving model in multiple formats...")
    
    # 1. Keras format (.keras) - already saved by checkpoint, but save final too
    keras_path = model_dir / f"guardian_phase1_{timestamp}.keras"
    model.save(keras_path)
    print(f"   ✅ Keras format: {keras_path}")
    
    # 2. H5 format (.h5) - legacy format, more compatible across versions
    h5_path = model_dir / f"guardian_phase1_{timestamp}.h5"
    model.save(h5_path, save_format='h5')
    print(f"   ✅ H5 format: {h5_path}")
    
    # 3. SavedModel format - TensorFlow native, most compatible
    savedmodel_path = model_dir / f"guardian_savedmodel_{timestamp}"
    model.export(savedmodel_path)
    print(f"   ✅ SavedModel format: {savedmodel_path}")
    
    # 4. Save weights only (smallest file, requires model architecture to load)
    weights_path = model_dir / f"guardian_weights_{timestamp}.weights.h5"
    model.save_weights(weights_path)
    print(f"   ✅ Weights only: {weights_path}")
    
    # Save class mapping
    class_mapping = {i: name for i, name in enumerate(class_names)}
    with open(model_dir / f"class_mapping_{timestamp}.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Save training summary
    summary = {
        "timestamp": timestamp,
        "model_type": "EfficientNetV2-S + LSTM (Phase 1 Only)",
        "num_classes": len(class_names),
        "class_names": class_names,
        "config": config,
        "training": {
            "phase1_epochs": len(history.history['loss']),
            "phase2_epochs": 0,  # Phase 2 skipped
            "best_val_accuracy": float(best_val_acc)
        },
        "data_split": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        },
        "saved_formats": {
            "keras": str(keras_path),
            "h5": str(h5_path),
            "savedmodel": str(savedmodel_path),
            "weights": str(weights_path)
        }
    }
    
    with open(model_dir / f"training_summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Summary saved to: training_summary_{timestamp}.json")
    
    # ─── Done ───
    print("\n" + "=" * 70)
    print("🎉 PHASE 1 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"   Best Val Accuracy: {best_val_acc*100:.1f}%")
    print(f"\n📁 Models saved in: {model_dir}")
    print(f"\n📦 Download these files:")
    print(f"   • guardian_savedmodel_{timestamp}/ (RECOMMENDED - folder)")
    print(f"   • guardian_phase1_{timestamp}.h5 (backup)")
    print(f"   • class_mapping_{timestamp}.json")
    print(f"\n💡 Next Steps:")
    print(f"   1. Download guardian_savedmodel_{timestamp} folder from /kaggle/working")
    print(f"   2. Copy to backend/ml_models/guardian/")
    print(f"   3. Update video_classifier.py to load SavedModel")


if __name__ == "__main__":
    main()
