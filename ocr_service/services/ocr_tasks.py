"""Service layer helpers for OCR task management."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..config import DEFAULT_PROMPT
from ..image_processor import (
    ImageProcessingError,
    process_image_from_bytes,
    process_image_from_url,
)
from ..task_queue import Task, BatchTask, BatchTaskItem, get_task_queue

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """Raised when the task queue has reached its capacity."""


@dataclass
class BatchEnqueueItem:
    """Represents enqueue result for a single URL in a batch request."""

    url: str
    task: Optional[Task]
    error: Optional[str] = None


async def enqueue_task_from_bytes(
    file_bytes: bytes,
    prompt: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    ngram_size: Optional[int] = None,
    window_size: Optional[int] = None,
) -> Task:
    """Validate raw bytes and enqueue an OCR task."""

    prompt_to_use = prompt or DEFAULT_PROMPT
    image_data, final_prompt = process_image_from_bytes(file_bytes, prompt_to_use)
    return await _enqueue_task(
        image_data=image_data,
        prompt=final_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        ngram_size=ngram_size,
        window_size=window_size,
    )


async def enqueue_task_from_url(
    url: str,
    prompt: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    ngram_size: Optional[int] = None,
    window_size: Optional[int] = None,
) -> Task:
    """Download an image from URL and enqueue an OCR task."""

    prompt_to_use = prompt or DEFAULT_PROMPT
    image_data, final_prompt = await process_image_from_url(url, prompt_to_use)
    return await _enqueue_task(
        image_data=image_data,
        prompt=final_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        ngram_size=ngram_size,
        window_size=window_size,
    )


async def enqueue_tasks_from_urls(
    urls: Sequence[str],
    prompt: Optional[str] = None,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    ngram_size: Optional[int] = None,
    window_size: Optional[int] = None,
) -> BatchTask:
    """Download multiple URLs concurrently and enqueue tasks as a batch, allowing queue overflow.
    
    Returns:
        BatchTask containing all sub-tasks
    """

    if not urls:
        raise ValueError("URLs list cannot be empty")

    prompt_to_use = prompt or DEFAULT_PROMPT
    task_queue = get_task_queue()
    
    # Create empty batch task first to get batch_task_id
    batch_items: List[BatchTaskItem] = []
    batch_task = await task_queue.create_batch_task(batch_items)

    # Helper function to process a single URL
    async def _process_single_url(url: str) -> BatchTaskItem:
        """Process a single URL and return a BatchTaskItem."""
        try:
            image_data, final_prompt = await process_image_from_url(url, prompt_to_use)
            task = await _enqueue_task(
                image_data=image_data,
                prompt=final_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                ngram_size=ngram_size,
                window_size=window_size,
                allow_overflow=True,
                batch_task_id=batch_task.batch_task_id,
            )
            return BatchTaskItem(url=url, task_id=task.task_id)
        except ImageProcessingError as exc:
            logger.warning("Failed to process image for URL %s: %s", url, exc)
            return BatchTaskItem(url=url, error=str(exc))
        except QueueFullError as exc:
            logger.error("Queue unexpectedly full while allowing overflow for URL %s: %s", url, exc)
            return BatchTaskItem(url=url, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error while enqueueing URL %s", url)
            return BatchTaskItem(url=url, error=str(exc))

    # Process all URLs concurrently
    batch_items = await asyncio.gather(*[_process_single_url(url) for url in urls])

    # Update batch task with items (thread-safe)
    await task_queue.update_batch_task_items(batch_task.batch_task_id, batch_items)

    return batch_task


async def _enqueue_task(
    *,
    image_data,
    prompt: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    ngram_size: Optional[int],
    window_size: Optional[int],
    allow_overflow: bool = False,
    batch_task_id: Optional[str] = None,
) -> Task:
    """Create a task and push it to the queue."""

    task_queue = get_task_queue()
    task = await task_queue.create_task(
        prompt=prompt,
        image_data=image_data,
        temperature=temperature,
        max_tokens=max_tokens,
        ngram_size=ngram_size,
        window_size=window_size,
        allow_overflow=allow_overflow,
        batch_task_id=batch_task_id,
    )

    if task is None:
        raise QueueFullError("Task queue is at full capacity")

    return task

