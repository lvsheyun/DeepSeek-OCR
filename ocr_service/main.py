"""FastAPI application for DeepSeek OCR Service"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, HttpUrl, Field

from .task_queue import get_task_queue, TaskStatus
from .worker import get_worker
from .image_processor import ImageProcessingError
from .config import LOG_LEVEL
from .services import (
    QueueFullError,
    enqueue_task_from_bytes,
    enqueue_task_from_url,
    enqueue_tasks_from_urls,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class OCRURLRequest(BaseModel):
    """Request model for URL-based OCR"""
    url: HttpUrl = Field(..., description="URL of the image to process")
    prompt: Optional[str] = Field(None, description="Custom prompt for OCR")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=16384, description="Maximum tokens to generate")
    ngram_size: Optional[int] = Field(None, ge=1, description="N-gram size for no-repeat processor")
    window_size: Optional[int] = Field(None, ge=1, description="Window size for no-repeat processor")


class OCRBatchURLRequest(BaseModel):
    """Request model for batch URL-based OCR"""

    urls: List[HttpUrl] = Field(
        ...,
        min_items=1,
        description="List of image URLs to process",
    )
    prompt: Optional[str] = Field(None, description="Custom prompt applied to all tasks")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=16384, description="Maximum tokens to generate")
    ngram_size: Optional[int] = Field(None, ge=1, description="N-gram size for no-repeat processor")
    window_size: Optional[int] = Field(None, ge=1, description="Window size for no-repeat processor")


class TaskResponse(BaseModel):
    """Response model for task creation"""
    task_id: str
    status: str


class BatchTaskItemResponse(BaseModel):
    """Response model for an individual URL in a batch submission"""

    url: str
    task_id: Optional[str] = None
    status: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None


class BatchTaskResponse(BaseModel):
    """Response model for batch task creation"""
    batch_task_id: str
    status: str


class BatchTaskResultResponse(BaseModel):
    """Response model for batch task result"""
    batch_task_id: str
    status: str
    total: int
    completed: int
    failed: int
    pending: int
    processing: int
    results: List[BatchTaskItemResponse]
    created_at: float
    updated_at: float


class TaskResultResponse(BaseModel):
    """Response model for task result"""
    task_id: str
    status: str
    created_at: float
    updated_at: float
    result: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str


class StatsResponse(BaseModel):
    """Response model for statistics"""
    total: int
    pending: int
    processing: int
    completed: int
    failed: int
    queue_limit: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    # Startup
    logger.info("Starting DeepSeek OCR Service...")
    
    # Initialize components
    task_queue = get_task_queue()
    worker = get_worker()
    
    # Start background tasks
    await task_queue.start_cleanup_task()
    await worker.start()
    
    logger.info("DeepSeek OCR Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DeepSeek OCR Service...")
    await worker.stop()
    await task_queue.stop_cleanup_task()
    logger.info("DeepSeek OCR Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="DeepSeek OCR Service",
    description="HTTP service for OCR using DeepSeek-OCR model via vLLM",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/ocr", response_model=TaskResponse, status_code=202)
async def ocr_from_file(
    file: UploadFile = File(..., description="Image file to process"),
    prompt: Optional[str] = Form(None, description="Custom prompt for OCR"),
    temperature: Optional[float] = Form(None, ge=0.0, le=2.0, description="Sampling temperature"),
    max_tokens: Optional[int] = Form(None, ge=1, le=16384, description="Maximum tokens to generate"),
    ngram_size: Optional[int] = Form(None, ge=1, description="N-gram size for no-repeat processor"),
    window_size: Optional[int] = Form(None, ge=1, description="Window size for no-repeat processor"),
):
    """
    Upload an image file for OCR processing
    
    Returns:
        - 202 Accepted: Task created successfully
        - 503 Service Unavailable: Queue is full
    """
    try:
        # Read file contents
        file_contents = await file.read()
        
        # Enqueue task via service layer
        task = await enqueue_task_from_bytes(
            file_bytes=file_contents,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            ngram_size=ngram_size,
            window_size=window_size,
        )
        
        return TaskResponse(
            task_id=task.task_id,
            status=task.status.value
        )
    except QueueFullError:
        raise HTTPException(
            status_code=503,
            detail="Service is at full capacity. Please try again later."
        )
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /ocr: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/ocr/url", response_model=TaskResponse, status_code=202)
async def ocr_from_url(request: OCRURLRequest):
    """
    Submit an image URL for OCR processing
    
    Returns:
        - 202 Accepted: Task created successfully
        - 503 Service Unavailable: Queue is full
    """
    try:
        # Enqueue task via service layer
        task = await enqueue_task_from_url(
            url=str(request.url),
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ngram_size=request.ngram_size,
            window_size=request.window_size,
        )
        
        return TaskResponse(
            task_id=task.task_id,
            status=task.status.value
        )
    except QueueFullError:
        raise HTTPException(
            status_code=503,
            detail="Service is at full capacity. Please try again later."
        )
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in /ocr/url: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/ocr/url/batch", response_model=BatchTaskResponse, status_code=202)
async def ocr_from_url_batch(request: OCRBatchURLRequest):
    """
    Submit multiple image URLs for OCR processing in a single batch request.
    
    The batch API allows exceeding the configured MAX_QUEUE_SIZE to keep
    pagination workloads together. Each URL is validated independently.
    
    Returns:
        - 202 Accepted with batch_task_id. Use GET /ocr/batch/{batch_task_id} to retrieve results.
    """

    try:
        batch_task = await enqueue_tasks_from_urls(
            urls=[str(url) for url in request.urls],
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ngram_size=request.ngram_size,
            window_size=request.window_size,
        )
    except ValueError as e:
        logger.error(f"Invalid request in /ocr/url/batch: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.error(f"Unexpected error in /ocr/url/batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    # Determine initial status
    total = len(batch_task.items)
    failed = sum(1 for item in batch_task.items if item.error is not None)
    if failed == total:
        status = "failed"
    else:
        status = "pending"

    return BatchTaskResponse(
        batch_task_id=batch_task.batch_task_id,
        status=status,
    )


@app.get("/ocr/{task_id}", response_model=TaskResultResponse)
async def get_task_result(task_id: str):
    """
    Get the status and result of an OCR task
    
    Returns:
        Task information including status and result (if completed)
    """
    task_queue = get_task_queue()
    
    task = await task_queue.get_task(task_id)
    
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return TaskResultResponse(**task.to_dict())


@app.get("/ocr/batch/{batch_task_id}", response_model=BatchTaskResultResponse)
async def get_batch_task_result(batch_task_id: str):
    """
    Get the status and results of a batch OCR task
    
    Returns:
        Batch task information including all sub-task results
    """
    task_queue = get_task_queue()
    
    batch_task = await task_queue.get_batch_task(batch_task_id)
    
    if batch_task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Batch task {batch_task_id} not found"
        )
    
    # Get full batch task details with current task statuses
    batch_dict = await batch_task.to_dict_async(task_queue)
    
    # Convert results to BatchTaskItemResponse
    results = [
        BatchTaskItemResponse(**item) for item in batch_dict["results"]
    ]
    
    return BatchTaskResultResponse(
        batch_task_id=batch_dict["batch_task_id"],
        status=batch_dict["status"],
        total=batch_dict["total"],
        completed=batch_dict["completed"],
        failed=batch_dict["failed"],
        pending=batch_dict["pending"],
        processing=batch_dict["processing"],
        results=results,
        created_at=batch_dict["created_at"],
        updated_at=batch_dict["updated_at"],
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Service health status
    """
    return HealthResponse(
        status="healthy",
        message="DeepSeek OCR Service is running"
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get queue statistics
    
    Returns:
        Statistics about pending, processing, and completed tasks
    """
    task_queue = get_task_queue()
    stats = await task_queue.get_stats()
    return StatsResponse(**stats)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "DeepSeek OCR Service",
        "version": "0.1.0",
        "endpoints": {
            "POST /ocr": "Upload image file for OCR",
            "POST /ocr/url": "Submit image URL for OCR",
            "POST /ocr/url/batch": "Submit multiple image URLs for OCR (returns batch_task_id)",
            "GET /ocr/{task_id}": "Get task result",
            "GET /ocr/batch/{batch_task_id}": "Get batch task result",
            "GET /health": "Health check",
            "GET /stats": "Queue statistics",
        }
    }

