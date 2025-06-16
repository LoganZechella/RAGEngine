#!/usr/bin/env python3
"""
Simple model preloader - downloads sentence-transformers model with minimal dependencies.
"""

import os
import sys

def simple_preload():
    """Simple model preload with basic error handling."""
    try:
        print("🔄 Downloading sentence-transformers model...")
        
        # Set cache directory to match where sentence-transformers looks by default
        cache_dir = "/root/.cache/sentence_transformers"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Import and download
        from sentence_transformers import SentenceTransformer
        
        # Download the model
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
        
        # Quick test
        test_embedding = model.encode(["test"])
        print(f"✅ Model cached successfully! Embedding shape: {test_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Model preload failed: {e}")
        print("📝 Container will download model at runtime instead")
        return False

if __name__ == "__main__":
    simple_preload()
    # Always exit successfully
    sys.exit(0) 