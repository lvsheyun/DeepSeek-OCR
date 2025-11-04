"""Model Manager for DeepSeek OCR - Singleton vLLM instance"""

import logging
from typing import List, Dict, Any, Optional
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

from .config import (
    MODEL_PATH,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    TENSOR_PARALLEL_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    NGRAM_SIZE,
    WINDOW_SIZE,
    WHITELIST_TOKEN_IDS,
)

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton manager for vLLM DeepSeek OCR model"""
    
    _instance: Optional['ModelManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the vLLM model (only once)"""
        if not ModelManager._initialized:
            logger.info("Initializing DeepSeek OCR model...")
            self.llm = LLM(
                model=MODEL_PATH,
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor],
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                max_model_len=MAX_MODEL_LEN,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                trust_remote_code=True,
            )
            logger.info("DeepSeek OCR model initialized successfully")
            ModelManager._initialized = True
    
    def create_sampling_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        ngram_size: Optional[int] = None,
        window_size: Optional[int] = None,
        whitelist_token_ids: Optional[set] = None,
    ) -> SamplingParams:
        """Create sampling parameters for inference"""
        return SamplingParams(
            temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
            extra_args=dict(
                ngram_size=ngram_size if ngram_size is not None else NGRAM_SIZE,
                window_size=window_size if window_size is not None else WINDOW_SIZE,
                whitelist_token_ids=whitelist_token_ids if whitelist_token_ids is not None else WHITELIST_TOKEN_IDS,
            ),
            skip_special_tokens=False,
        )
    
    def process_batch(
        self,
        inputs: List[Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """
        Process a batch of inputs using vLLM
        
        Args:
            inputs: List of dicts with 'prompt' and 'multi_modal_data' keys
            sampling_params: Optional sampling parameters
            
        Returns:
            List of OCR results (strings)
        """
        if not inputs:
            return []
        
        if sampling_params is None:
            sampling_params = self.create_sampling_params()
        
        logger.info(f"Processing batch of {len(inputs)} images")
        
        try:
            outputs = self.llm.generate(inputs, sampling_params)
            results = [output.outputs[0].text for output in outputs]
            logger.info(f"Batch processing completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error during batch processing: {e}", exc_info=True)
            raise
    
    def process_single(
        self,
        prompt: str,
        image_data: Any,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """
        Process a single image (convenience method)
        
        Args:
            prompt: Text prompt
            image_data: Preprocessed image data from DeepseekOCRProcessor
            sampling_params: Optional sampling parameters
            
        Returns:
            OCR result string
        """
        model_input = [{
            "prompt": prompt,
            "multi_modal_data": {"image": image_data}
        }]
        
        results = self.process_batch(model_input, sampling_params)
        return results[0] if results else ""


# Global singleton instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

