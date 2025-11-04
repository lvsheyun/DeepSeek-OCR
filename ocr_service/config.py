"""Configuration for DeepSeek OCR Service"""

import os
from typing import Optional

# Model Configuration
# NOTE: These parameters are currently HARD-CODED in vLLM's internal processor
# (vllm/transformers_utils/processors/deepseek_ocr.py)
# They are kept here for reference and future compatibility when vLLM exposes them as mm_kwargs
# Current vLLM defaults (Gundam mode): base_size=1024, image_size=640, crop_mode=True
BASE_SIZE = 1024  # Reference only - hard-coded in vLLM
IMAGE_SIZE = 640  # Reference only - hard-coded in vLLM
CROP_MODE = True  # Reference only - hard-coded in vLLM
MIN_CROPS = 2     # Reference only - hard-coded in vLLM
MAX_CROPS = 6     # Reference only - hard-coded in vLLM (max: 9, recommended: 6 for limited GPU)

# Model Path
MODEL_PATH = os.getenv("OCR_MODEL_PATH", "deepseek-ai/DeepSeek-OCR")

# GPU Settings
GPU_MEMORY_UTILIZATION = float(os.getenv("OCR_GPU_MEMORY_UTIL", "0.8"))
MAX_MODEL_LEN = int(os.getenv("OCR_MAX_MODEL_LEN", "8192"))
TENSOR_PARALLEL_SIZE = int(os.getenv("OCR_TENSOR_PARALLEL_SIZE", "1"))

# Service Configuration
MAX_QUEUE_SIZE = int(os.getenv("OCR_MAX_QUEUE_SIZE", "100"))
BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "16"))
BATCH_TIMEOUT = float(os.getenv("OCR_BATCH_TIMEOUT", "2.0"))  # seconds
TASK_TTL = int(os.getenv("OCR_TASK_TTL", "3600"))  # seconds (1 hour)
CLEANUP_INTERVAL = int(os.getenv("OCR_CLEANUP_INTERVAL", "300"))  # seconds (5 minutes)

# Image Download Settings
MAX_IMAGE_SIZE_MB = int(os.getenv("OCR_MAX_IMAGE_SIZE_MB", "20"))
IMAGE_DOWNLOAD_TIMEOUT = int(os.getenv("OCR_IMAGE_DOWNLOAD_TIMEOUT", "30"))  # seconds

# Default OCR Settings
DEFAULT_PROMPT = os.getenv("OCR_DEFAULT_PROMPT", "<image>\n<|grounding|>Convert the document to markdown.")
DEFAULT_TEMPERATURE = float(os.getenv("OCR_DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("OCR_DEFAULT_MAX_TOKENS", "8192"))

# NGram LogitsProcessor Settings
NGRAM_SIZE = int(os.getenv("OCR_NGRAM_SIZE", "30"))
WINDOW_SIZE = int(os.getenv("OCR_WINDOW_SIZE", "90"))
# Whitelist token IDs for <td>, </td>
WHITELIST_TOKEN_IDS = {128821, 128822}

# Server Settings
HOST = os.getenv("OCR_HOST", "0.0.0.0")
PORT = int(os.getenv("OCR_PORT", "8000"))
WORKERS = int(os.getenv("OCR_WORKERS", "1"))  # Should be 1 for GPU workloads

# Logging
LOG_LEVEL = os.getenv("OCR_LOG_LEVEL", "INFO")

