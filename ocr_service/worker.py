"""Background worker for processing OCR tasks in batches"""

import asyncio
import logging
from typing import List, Optional

from .model_manager import get_model_manager
from .task_queue import get_task_queue, Task, TaskStatus
from .config import BATCH_SIZE, BATCH_TIMEOUT

logger = logging.getLogger(__name__)


class BatchWorker:
    """Background worker that processes OCR tasks in batches"""
    
    def __init__(self):
        self.model_manager = get_model_manager()
        self.task_queue = get_task_queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def process_batch(self, tasks: List[Task]):
        """
        Process a batch of tasks using the model
        
        Args:
            tasks: List of tasks to process
        """
        if not tasks:
            return
        
        logger.info(f"Processing batch of {len(tasks)} tasks")
        
        # Prepare inputs for vLLM
        model_inputs = []
        for task in tasks:
            model_input = {
                "prompt": task.prompt,
                "multi_modal_data": {"image": task.image_data}
            }
            model_inputs.append(model_input)
        
        try:
            # Create sampling parameters from the first task (they should all be similar)
            # In practice, we could group tasks by sampling params for optimal batching
            first_task = tasks[0]
            sampling_params = self.model_manager.create_sampling_params(
                temperature=first_task.temperature,
                max_tokens=first_task.max_tokens,
                ngram_size=first_task.ngram_size,
                window_size=first_task.window_size,
            )
            
            # Process batch - run in thread pool to avoid blocking event loop
            # vLLM operations are CPU/GPU bound, not async
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.model_manager.process_batch,
                model_inputs,
                sampling_params
            )
            
            # Update tasks with results
            for task, result in zip(tasks, results):
                await self.task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.COMPLETED,
                    result=result
                )
                logger.info(f"Task {task.task_id} completed successfully")
        
        except Exception as e:
            # Mark all tasks in batch as failed
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            error_message = str(e)
            
            for task in tasks:
                await self.task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    error=error_message
                )
                logger.error(f"Task {task.task_id} failed: {error_message}")
    
    async def worker_loop(self):
        """Main worker loop that continuously processes batches"""
        logger.info("Starting batch worker loop")
        
        while self._running:
            try:
                # Get next batch of tasks
                # This will wait until we have tasks or timeout
                batch = await self.task_queue.get_batch(
                    batch_size=BATCH_SIZE,
                    timeout=BATCH_TIMEOUT
                )
                
                if batch:
                    # Process the batch
                    await self.process_batch(batch)
                else:
                    # No tasks available, short sleep before checking again
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                # Sleep briefly before continuing to avoid tight error loop
                await asyncio.sleep(1.0)
        
        logger.info("Batch worker loop stopped")
    
    async def start(self):
        """Start the background worker"""
        if self._running:
            logger.warning("Worker already running")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self.worker_loop())
        logger.info("Batch worker started")
    
    async def stop(self):
        """Stop the background worker"""
        if not self._running:
            logger.warning("Worker not running")
            return
        
        logger.info("Stopping batch worker...")
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Batch worker stopped")


# Global worker instance
_worker: Optional[BatchWorker] = None


def get_worker() -> BatchWorker:
    """Get or create the global worker instance"""
    global _worker
    if _worker is None:
        _worker = BatchWorker()
    return _worker

