"""Task Queue System for managing OCR tasks"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

from .config import MAX_QUEUE_SIZE, BATCH_SIZE, BATCH_TIMEOUT, TASK_TTL, CLEANUP_INTERVAL

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents an OCR task"""
    task_id: str
    status: TaskStatus
    prompt: str
    image_data: Any  # Preprocessed image data
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Optional sampling parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    ngram_size: Optional[int] = None
    window_size: Optional[int] = None
    
    # Optional batch task reference
    batch_task_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API response"""
        response = {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        
        if self.status == TaskStatus.COMPLETED:
            response["result"] = self.result
        elif self.status == TaskStatus.FAILED:
            response["error"] = self.error
            
        return response


@dataclass
class BatchTaskItem:
    """Represents a single item within a batch task"""
    url: str
    task_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class BatchTask:
    """Represents a batch OCR task containing multiple sub-tasks"""
    batch_task_id: str
    items: List[BatchTaskItem]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_dict(self, task_queue: 'TaskQueue') -> Dict[str, Any]:
        """Convert batch task to dictionary for API response"""
        results = []
        total = len(self.items)
        completed = 0
        failed = 0
        pending = 0
        processing = 0
        
        for item in self.items:
            if item.task_id:
                task = None
                # Try to get task status (need to access task_queue, but we'll do it async)
                # For now, we'll get status from the task directly in the async method
                task_dict = {
                    "url": item.url,
                    "task_id": item.task_id,
                }
            else:
                task_dict = {
                    "url": item.url,
                    "error": item.error or "Unknown error",
                }
                failed += 1
            
            results.append(task_dict)
        
        return {
            "batch_task_id": self.batch_task_id,
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "processing": processing,
            "results": results,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    async def to_dict_async(self, task_queue: 'TaskQueue') -> Dict[str, Any]:
        """Convert batch task to dictionary with current task statuses"""
        results = []
        total = len(self.items)
        completed = 0
        failed = 0
        pending = 0
        processing = 0
        
        for item in self.items:
            if item.task_id:
                task = await task_queue.get_task(item.task_id)
                if task:
                    status = task.status
                    if status == TaskStatus.COMPLETED:
                        completed += 1
                        task_dict = {
                            "url": item.url,
                            "task_id": item.task_id,
                            "status": status.value,
                            "result": task.result,
                        }
                    elif status == TaskStatus.FAILED:
                        failed += 1
                        task_dict = {
                            "url": item.url,
                            "task_id": item.task_id,
                            "status": status.value,
                            "error": task.error,
                        }
                    elif status == TaskStatus.PROCESSING:
                        processing += 1
                        task_dict = {
                            "url": item.url,
                            "task_id": item.task_id,
                            "status": status.value,
                        }
                    else:  # PENDING
                        pending += 1
                        task_dict = {
                            "url": item.url,
                            "task_id": item.task_id,
                            "status": status.value,
                        }
                else:
                    # Task not found, treat as failed
                    failed += 1
                    task_dict = {
                        "url": item.url,
                        "task_id": item.task_id,
                        "error": "Task not found",
                    }
            else:
                failed += 1
                task_dict = {
                    "url": item.url,
                    "error": item.error or "Unknown error",
                }
            
            results.append(task_dict)
        
        # Determine overall batch status
        if completed == total:
            batch_status = "completed"
        elif failed == total:
            batch_status = "failed"
        elif processing > 0 or pending > 0:
            batch_status = "processing"
        else:
            batch_status = "pending"
        
        return {
            "batch_task_id": self.batch_task_id,
            "status": batch_status,
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "processing": processing,
            "results": results,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskQueue:
    """Thread-safe task queue with batching and cleanup"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.batch_tasks: Dict[str, BatchTask] = {}
        self.pending_queue: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def create_task(
        self,
        prompt: str,
        image_data: Any,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        ngram_size: Optional[int] = None,
        window_size: Optional[int] = None,
        *,
        allow_overflow: bool = False,
        batch_task_id: Optional[str] = None,
    ) -> Optional[Task]:
        """
        Create a new task and add to queue
        
        Returns:
            Task if created successfully, None if queue is full and overflow not allowed
        """
        async with self.lock:
            # Check if queue is full
            current_size = len(self.tasks)

            if not allow_overflow and current_size >= MAX_QUEUE_SIZE:
                logger.warning(f"Queue is full ({current_size}/{MAX_QUEUE_SIZE})")
                return None

            if allow_overflow and current_size >= MAX_QUEUE_SIZE:
                logger.warning(
                    "Queue size %s exceeds configured MAX_QUEUE_SIZE %s due to overflow allowance",
                    current_size + 1,
                    MAX_QUEUE_SIZE,
                )
            
            task_id = str(uuid.uuid4())
            task = Task(
                task_id=task_id,
                status=TaskStatus.PENDING,
                prompt=prompt,
                image_data=image_data,
                temperature=temperature,
                max_tokens=max_tokens,
                ngram_size=ngram_size,
                window_size=window_size,
                batch_task_id=batch_task_id,
            )
            
            self.tasks[task_id] = task
            await self.pending_queue.put(task)
            
            logger.info(f"Created task {task_id}, queue size: {len(self.tasks)}")
            return task
    
    async def create_batch_task(self, items: List[BatchTaskItem]) -> BatchTask:
        """Create a new batch task"""
        async with self.lock:
            batch_task_id = str(uuid.uuid4())
            batch_task = BatchTask(
                batch_task_id=batch_task_id,
                items=items,
            )
            
            self.batch_tasks[batch_task_id] = batch_task
            logger.info(f"Created batch task {batch_task_id} with {len(items)} items")
            return batch_task
    
    async def get_batch_task(self, batch_task_id: str) -> Optional[BatchTask]:
        """Get batch task by ID"""
        async with self.lock:
            return self.batch_tasks.get(batch_task_id)
    
    async def update_batch_task_items(self, batch_task_id: str, items: List[BatchTaskItem]):
        """Update batch task items (thread-safe)"""
        async with self.lock:
            if batch_task_id in self.batch_tasks:
                batch_task = self.batch_tasks[batch_task_id]
                batch_task.items = items
                batch_task.updated_at = time.time()
                return True
            return False
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        async with self.lock:
            return self.tasks.get(task_id)
    
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Update task status and result/error"""
        async with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = status
                task.updated_at = time.time()
                
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                
                logger.info(f"Task {task_id} updated to status: {status.value}")
    
    async def get_batch(self, batch_size: int = BATCH_SIZE, timeout: float = BATCH_TIMEOUT) -> List[Task]:
        """
        Get a batch of pending tasks
        
        Waits until:
        - batch_size tasks are available, OR
        - timeout seconds have elapsed (and at least 1 task is available)
        
        Returns:
            List of tasks to process (may be less than batch_size)
        """
        batch: List[Task] = []
        start_time = time.time()
        
        while len(batch) < batch_size:
            remaining_time = timeout - (time.time() - start_time)
            
            # If we have at least one task and timeout has elapsed, return what we have
            if batch and remaining_time <= 0:
                logger.info(f"Batch timeout reached with {len(batch)} tasks")
                break
            
            # If timeout elapsed and no tasks, wait a bit longer for at least one task
            wait_time = remaining_time if remaining_time > 0 else 0.1
            
            try:
                task = await asyncio.wait_for(self.pending_queue.get(), timeout=wait_time)
                batch.append(task)
                
                # Mark as processing
                await self.update_task_status(task.task_id, TaskStatus.PROCESSING)
                
            except asyncio.TimeoutError:
                # Timeout reached
                if batch:
                    # We have some tasks, return them
                    logger.info(f"Batch ready with {len(batch)} tasks after timeout")
                    break
                else:
                    # No tasks yet, continue waiting
                    continue
        
        if batch:
            logger.info(f"Returning batch of {len(batch)} tasks")
        
        return batch
    
    async def cleanup_old_tasks(self):
        """Remove tasks older than TTL"""
        async with self.lock:
            current_time = time.time()
            to_remove = []
            
            for task_id, task in self.tasks.items():
                if current_time - task.created_at > TASK_TTL:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
                logger.info(f"Cleaned up old task {task_id}")
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old tasks")
            
            # Clean up old batch tasks
            batch_to_remove = []
            for batch_task_id, batch_task in self.batch_tasks.items():
                if current_time - batch_task.created_at > TASK_TTL:
                    batch_to_remove.append(batch_task_id)
            
            for batch_task_id in batch_to_remove:
                del self.batch_tasks[batch_task_id]
                logger.info(f"Cleaned up old batch task {batch_task_id}")
            
            if batch_to_remove:
                logger.info(f"Cleaned up {len(batch_to_remove)} old batch tasks")
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(CLEANUP_INTERVAL)
                await self.cleanup_old_tasks()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started cleanup task")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped cleanup task")
    
    async def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        async with self.lock:
            stats = {
                "total": len(self.tasks),
                "pending": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
                "processing": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PROCESSING),
                "completed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
                "failed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
                "queue_limit": MAX_QUEUE_SIZE,
            }
            return stats


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create the global task queue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue

