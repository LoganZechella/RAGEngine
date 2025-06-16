#!/usr/bin/env python3
"""
Pre-download ML models for Docker container.
This script downloads and caches models during Docker build time to prevent
runtime download timeouts in containerized environments.
"""

import os
import sys
from loguru import logger

def preload_sentence_transformers():
    """Pre-download the sentence-transformers model used by TextChunker."""
    try:
        logger.info("Pre-downloading sentence-transformers model...")
        
        # Import and initialize the model - this triggers download and caching
        from sentence_transformers import SentenceTransformer
        import time
        
        model_name = 'all-MiniLM-L6-v2'
        logger.info(f"Downloading model: {model_name}")
        
        # Add retry logic for model loading
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # This will download and cache the model
                model = SentenceTransformer(model_name)
                
                # Give the model a moment to fully initialize
                time.sleep(2)
                
                # Test the model to ensure it's working
                test_sentences = ["This is a test sentence.", "This is another test."]
                embeddings = model.encode(test_sentences)
                
                logger.info(f"Successfully pre-loaded {model_name}")
                logger.info(f"Model cache location: {model.cache_folder}")
                logger.info(f"Test embedding shape: {embeddings.shape}")
                
                return True
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise e
        
        return False
        
    except ImportError as e:
        logger.error(f"sentence-transformers library not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to pre-load sentence-transformers model: {e}")
        logger.info("This could be due to:")
        logger.info("  - Network connectivity issues during model download")
        logger.info("  - Insufficient disk space for model cache")
        logger.info("  - File system permissions on cache directory")
        return False

def main():
    """Main function to pre-load all required models."""
    logger.info("Starting model pre-loading for Docker container...")
    
    success = True
    
    # Pre-load sentence-transformers model
    if not preload_sentence_transformers():
        logger.error("Failed to pre-load sentence-transformers model")
        logger.warning("Container will still work, but may experience slower startup on first use")
        success = False
    
    if success:
        logger.info("All models pre-loaded successfully!")
        logger.info("Container should start quickly without model downloads")
    else:
        logger.warning("Model pre-loading incomplete - container will download models at runtime")
        logger.info("This may cause slower startup but functionality will not be affected")
    
    # Always exit successfully to not fail the Docker build
    # The application can handle runtime model downloads as fallback
    sys.exit(0)

if __name__ == "__main__":
    main() 