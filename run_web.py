#!/usr/bin/env python3
"""
RAGEngine Web Interface Launcher
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv
from loguru import logger

def main():
    # Add backend directory to Python path
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    
    # Load environment from backend directory
    env_path = os.path.join(backend_dir, '.env')
    load_dotenv(env_path)
    
    # Configure logging
    logger.add("ragengine_web.log", rotation="10 MB", level="INFO")
    
    # Check required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file in the backend directory.")
        sys.exit(1)
    
    # Optional but recommended
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not set. Knowledge synthesis will not be available.")
        print("⚠️  GOOGLE_API_KEY not set. Knowledge synthesis will not be available.")
    
    print("🚀 Starting RAGEngine Web Interface...")
    print("📍 Dashboard will be available at: http://localhost:8080")
    print("🔧 Configuration:")
    print(f"   • Collection: {os.getenv('COLLECTION_NAME', 'knowledge_base')}")
    print(f"   • Qdrant URL: {os.getenv('QDRANT_URL', 'http://localhost:6333')}")
    print(f"   • Documents: {os.getenv('SOURCE_DOCUMENTS_DIR', './documents')}")
    print("\n💡 Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "web_app:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 RAGEngine Web Interface stopped")
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        print(f"❌ Failed to start web server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 