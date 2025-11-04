#!/usr/bin/env python3
"""Entry point for DeepSeek OCR HTTP Service"""

import os
import sys
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ocr_service.config import HOST, PORT, WORKERS, LOG_LEVEL


def main():
    """Run the FastAPI service"""
    
    print("=" * 60)
    print("DeepSeek OCR HTTP Service")
    print("=" * 60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Workers: {WORKERS}")
    print(f"Log Level: {LOG_LEVEL}")
    print("=" * 60)
    
    # Run uvicorn server
    uvicorn.run(
        "ocr_service.main:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        log_level=LOG_LEVEL.lower(),
        # Disable reload in production, enable for development
        reload=False,
    )


if __name__ == "__main__":
    main()

