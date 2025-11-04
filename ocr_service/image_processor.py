"""Image processing utilities for OCR service"""

import io
import logging
from typing import Any, Optional, Tuple
from PIL import Image, ImageOps
import httpx

from .config import (
    MAX_IMAGE_SIZE_MB,
    IMAGE_DOWNLOAD_TIMEOUT,
    BASE_SIZE,
    IMAGE_SIZE,
    CROP_MODE,
    MIN_CROPS,
    MAX_CROPS,
    MODEL_PATH,
)

logger = logging.getLogger(__name__)

# Import DeepseekOCRProcessor from existing codebase
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DeepSeek-OCR-master', 'DeepSeek-OCR-vllm'))

try:
    from process.image_process import DeepseekOCRProcessor
    from transformers import AutoTokenizer
    
    # Initialize processor singleton
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    _processor = DeepseekOCRProcessor(tokenizer=_tokenizer)
    
    logger.info("DeepseekOCRProcessor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize DeepseekOCRProcessor: {e}")
    _processor = None


class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


async def download_image_from_url(url: str) -> Image.Image:
    """
    Download image from URL
    
    Args:
        url: Image URL
        
    Returns:
        PIL Image
        
    Raises:
        ImageProcessingError: If download fails or image is invalid
    """
    try:
        logger.info(f"Downloading image from URL: {url}")
        
        async with httpx.AsyncClient(timeout=IMAGE_DOWNLOAD_TIMEOUT) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ImageProcessingError(f"URL does not point to an image (content-type: {content_type})")
            
            # Check size
            content_length = len(response.content)
            max_size_bytes = MAX_IMAGE_SIZE_MB * 1024 * 1024
            if content_length > max_size_bytes:
                raise ImageProcessingError(
                    f"Image too large: {content_length / 1024 / 1024:.2f}MB (max: {MAX_IMAGE_SIZE_MB}MB)"
                )
            
            # Load image
            image_bytes = io.BytesIO(response.content)
            image = Image.open(image_bytes)
            
            # Handle EXIF orientation
            image = ImageOps.exif_transpose(image)
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"Image downloaded successfully: {image.size}")
            return image
            
    except httpx.HTTPError as e:
        raise ImageProcessingError(f"Failed to download image: {e}")
    except Exception as e:
        raise ImageProcessingError(f"Failed to process downloaded image: {e}")


def validate_and_load_image(image_bytes: bytes) -> Image.Image:
    """
    Validate and load image from bytes
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        PIL Image
        
    Raises:
        ImageProcessingError: If image is invalid
    """
    try:
        # Check size
        size_mb = len(image_bytes) / 1024 / 1024
        if size_mb > MAX_IMAGE_SIZE_MB:
            raise ImageProcessingError(f"Image too large: {size_mb:.2f}MB (max: {MAX_IMAGE_SIZE_MB}MB)")
        
        # Load image
        image_io = io.BytesIO(image_bytes)
        image = Image.open(image_io)
        
        # Handle EXIF orientation
        image = ImageOps.exif_transpose(image)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Image loaded successfully: {image.size}")
        return image
        
    except Exception as e:
        raise ImageProcessingError(f"Failed to load image: {e}")


def preprocess_image(image: Image.Image, prompt: str) -> Tuple[Image.Image, str]:
    """
    Prepare image for vLLM processing
    
    Args:
        image: PIL Image
        prompt: Text prompt (should contain <image> tag)
        
    Returns:
        Tuple of (PIL Image, prompt)
        
    Raises:
        ImageProcessingError: If preprocessing fails
    """
    try:
        # Ensure prompt contains <image> tag
        if '<image>' not in prompt:
            logger.warning(f"Prompt does not contain <image> tag, adding it: {prompt}")
            prompt = f"<image>\n{prompt}"
        
        # Return the PIL Image directly - vLLM will handle the preprocessing
        logger.info(f"Image prepared successfully: {image.size}")
        return image, prompt
        
    except Exception as e:
        logger.error(f"Failed to prepare image: {e}", exc_info=True)
        raise ImageProcessingError(f"Failed to prepare image: {e}")


async def process_image_from_url(url: str, prompt: str) -> Tuple[Image.Image, str]:
    """
    Download and prepare image from URL
    
    Args:
        url: Image URL
        prompt: Text prompt
        
    Returns:
        Tuple of (PIL Image, prompt)
    """
    image = await download_image_from_url(url)
    return preprocess_image(image, prompt)


def process_image_from_bytes(image_bytes: bytes, prompt: str) -> Tuple[Image.Image, str]:
    """
    Load and prepare image from bytes
    
    Args:
        image_bytes: Raw image bytes
        prompt: Text prompt
        
    Returns:
        Tuple of (PIL Image, prompt)
    """
    image = validate_and_load_image(image_bytes)
    return preprocess_image(image, prompt)

