"""
SMARTCARE+ Worker Thread Pool

ThreadPoolExecutor for CPU-intensive video/ML processing
without blocking the async event loop.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, Optional
from queue import Queue, Empty
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import threading

from core.config import settings

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a processing task."""
    task_id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class WorkerPool:
    """
    Thread pool for CPU-intensive operations.
    
    Features:
    - Fixed-size thread pool
    - Task queue with priority
    - Async-compatible execution
    - Task tracking and cancellation
    """
    
    def __init__(
        self,
        max_workers: int = None,
        queue_size: int = None,
        name: str = "worker_pool"
    ):
        self.max_workers = max_workers or settings.THREAD_POOL_SIZE
        self.queue_size = queue_size or settings.VIDEO_FRAME_QUEUE_SIZE
        self.name = name
        
        # Thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=f"{name}_"
        )
        
        # Task tracking
        self._tasks: dict[str, Task] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        
        # Frame queue for video processing
        self._frame_queue: Queue = Queue(maxsize=self.queue_size)
        
        # Stats
        self._completed_count = 0
        self._failed_count = 0
        
        logger.info(
            f"🧵 WorkerPool '{name}' initialized "
            f"(workers: {self.max_workers}, queue: {self.queue_size})"
        )
    
    def submit(
        self,
        func: Callable,
        *args,
        task_id: str = None,
        **kwargs
    ) -> str:
        """
        Submit a task to the thread pool.
        
        Returns:
            task_id for tracking
        """
        task_id = task_id or f"task_{datetime.now().timestamp()}"
        
        task = Task(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs
        )
        
        with self._lock:
            self._tasks[task_id] = task
        
        # Submit to executor
        future = self._executor.submit(self._run_task, task)
        
        with self._lock:
            self._futures[task_id] = future
        
        logger.debug(f"Task {task_id} submitted")
        return task_id
    
    async def submit_async(
        self,
        func: Callable,
        *args,
        task_id: str = None,
        **kwargs
    ) -> Any:
        """
        Submit and await a task result (async-friendly).
        """
        task_id = self.submit(func, *args, task_id=task_id, **kwargs)
        
        # Wait for completion
        loop = asyncio.get_event_loop()
        future = self._futures.get(task_id)
        
        if future:
            result = await loop.run_in_executor(None, future.result)
            return result
        
        return None
    
    def _run_task(self, task: Task) -> Any:
        """Execute a task in the thread pool."""
        task.status = TaskStatus.RUNNING
        
        try:
            result = task.func(*task.args, **task.kwargs)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now(timezone.utc)
            
            with self._lock:
                self._completed_count += 1
            
            logger.debug(f"Task {task.task_id} completed")
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc)
            
            with self._lock:
                self._failed_count += 1
            
            logger.error(f"Task {task.task_id} failed: {e}")
            raise
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a task."""
        task = self._tasks.get(task_id)
        return task.status if task else None
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a completed task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        future = self._futures.get(task_id)
        
        if future and not future.done():
            cancelled = future.cancel()
            if cancelled:
                task = self._tasks.get(task_id)
                if task:
                    task.status = TaskStatus.CANCELLED
            return cancelled
        
        return False
    
    # ========================================
    # Frame Queue Methods (for video processing)
    # ========================================
    
    def enqueue_frame(self, frame_data: dict, block: bool = True, timeout: float = 1.0) -> bool:
        """
        Add a video frame to the processing queue.
        
        frame_data should include:
        - frame: numpy array or base64 string
        - user_id: string
        - timestamp: ISO string
        - metadata: optional dict
        """
        try:
            self._frame_queue.put(frame_data, block=block, timeout=timeout)
            return True
        except Exception:
            return False
    
    def dequeue_frame(self, block: bool = True, timeout: float = 1.0) -> Optional[dict]:
        """Get a frame from the processing queue."""
        try:
            return self._frame_queue.get(block=block, timeout=timeout)
        except Empty:
            return None
    
    @property
    def frame_queue_size(self) -> int:
        return self._frame_queue.qsize()
    
    def clear_frame_queue(self):
        """Clear all pending frames."""
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Empty:
                break
    
    # ========================================
    # Lifecycle
    # ========================================
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool."""
        logger.info(f"Shutting down WorkerPool '{self.name}'...")
        self._executor.shutdown(wait=wait)
        self.clear_frame_queue()
        logger.info(f"WorkerPool '{self.name}' shutdown complete")
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        return {
            "name": self.name,
            "max_workers": self.max_workers,
            "pending_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.PENDING]),
            "running_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]),
            "completed_tasks": self._completed_count,
            "failed_tasks": self._failed_count,
            "frame_queue_size": self.frame_queue_size,
            "frame_queue_capacity": self.queue_size
        }


# ============================================
# Global Worker Pools
# ============================================

# Video processing pool (for gait analysis, fall detection)
video_worker_pool = WorkerPool(name="video_processing")

# ML inference pool (for food recognition, pose estimation)
ml_worker_pool = WorkerPool(name="ml_inference", max_workers=2)


def get_video_pool() -> WorkerPool:
    """Get the video processing worker pool."""
    return video_worker_pool


def get_ml_pool() -> WorkerPool:
    """Get the ML inference worker pool."""
    return ml_worker_pool


async def process_video_frame(
    frame_processor: Callable,
    frame_data: dict
) -> Any:
    """
    Process a video frame using the video worker pool.
    
    Usage:
        result = await process_video_frame(
            analyze_gait_frame,
            {"frame": frame_array, "user_id": "123"}
        )
    """
    return await video_worker_pool.submit_async(frame_processor, frame_data)


async def run_ml_inference(
    model_fn: Callable,
    input_data: Any
) -> Any:
    """
    Run ML inference using the ML worker pool.
    
    Usage:
        result = await run_ml_inference(
            food_classifier.predict,
            preprocessed_image
        )
    """
    return await ml_worker_pool.submit_async(model_fn, input_data)
