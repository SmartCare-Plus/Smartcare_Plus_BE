"""
SMARTCARE+ Threading Module
"""

from .worker_pool import (
    WorkerPool,
    Task,
    TaskStatus,
    video_worker_pool,
    ml_worker_pool,
    get_video_pool,
    get_ml_pool,
    process_video_frame,
    run_ml_inference
)

__all__ = [
    'WorkerPool',
    'Task',
    'TaskStatus',
    'video_worker_pool',
    'ml_worker_pool',
    'get_video_pool',
    'get_ml_pool',
    'process_video_frame',
    'run_ml_inference'
]
