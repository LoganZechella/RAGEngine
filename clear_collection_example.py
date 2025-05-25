#!/usr/bin/env python3
"""
Example script showing how to clear Qdrant collection data.
"""

import os
from dotenv import load_dotenv
from loguru import logger
from src.api.knowledge_base_api import KnowledgeBaseAPI

def main():
    """Demonstrate collection clearing methods."""
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "collection_name": os.getenv("COLLECTION_NAME", "knowledge_base"),
        "source_paths": [os.getenv("SOURCE_DOCUMENTS_DIR", "./documents")],
        "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "3072")),
    }
    
    print("🔧 Initializing Knowledge Base API...")
    kb_api = KnowledgeBaseAPI(config)
    
    # Get current collection stats
    print("\n📊 Current Collection Statistics:")
    system_info = kb_api.get_system_info()
    vector_db_info = system_info.get('vector_db', {})
    print(f"   Collection Name: {vector_db_info.get('name', 'Unknown')}")
    print(f"   Total Points: {vector_db_info.get('points_count', 0):,}")
    print(f"   Vector Size: {vector_db_info.get('vector_size', 'Unknown')}")
    
    if vector_db_info.get('points_count', 0) == 0:
        print("\n✅ Collection is already empty!")
        return
    
    print("\n🗑️ Collection Clearing Options:")
    print("1. Clear all data (preserves collection structure)")
    print("2. Delete entire collection (collection will be recreated)")
    print("3. Recreate collection (delete + recreate with fresh indexes)")
    print("4. Cancel")
    
    choice = input("\nSelect an option (1-4): ").strip()
    
    if choice == "1":
        print("\n⚠️  This will clear all data but preserve the collection structure.")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            print("\n🗑️  Clearing collection data...")
            success = kb_api.clear_collection()
            if success:
                print("✅ Collection data cleared successfully!")
            else:
                print("❌ Failed to clear collection data.")
    
    elif choice == "2":
        print("\n⚠️  This will completely delete the collection.")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            print("\n🗑️  Deleting collection...")
            success = kb_api.delete_collection()
            if success:
                print("✅ Collection deleted successfully!")
            else:
                print("❌ Failed to delete collection.")
    
    elif choice == "3":
        print("\n⚠️  This will delete and recreate the collection with fresh indexes.")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm in ['yes', 'y']:
            print("\n🔄 Recreating collection...")
            success = kb_api.recreate_collection()
            if success:
                print("✅ Collection recreated successfully!")
            else:
                print("❌ Failed to recreate collection.")
    
    elif choice == "4":
        print("Operation cancelled.")
        return
    
    else:
        print("Invalid choice. Operation cancelled.")
        return
    
    # Show final stats
    print("\n📊 Final Collection Statistics:")
    system_info = kb_api.get_system_info()
    vector_db_info = system_info.get('vector_db', {})
    print(f"   Collection Name: {vector_db_info.get('name', 'Unknown')}")
    print(f"   Total Points: {vector_db_info.get('points_count', 0):,}")
    
    print("\n💡 To reload your documents, run:")
    print("   python interactive_rag.py")
    print("   RAG> ingest")

if __name__ == "__main__":
    main() 