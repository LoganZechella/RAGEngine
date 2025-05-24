#!/usr/bin/env python3
"""
RAGEngine - Standalone Knowledge Base System
Main example demonstrating usage of the RAGEngine system.
"""

import os
from dotenv import load_dotenv
from loguru import logger
from src.api.knowledge_base_api import KnowledgeBaseAPI

def main():
    """Main function demonstrating RAGEngine usage."""
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "google_api_key": os.getenv("GOOGLE_API_KEY"),  # Optional
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
        "collection_name": os.getenv("COLLECTION_NAME", "knowledge_base"),
        "source_paths": [os.getenv("SOURCE_DOCUMENTS_DIR", "./documents")],
        "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "paragraph"),
        "chunk_size_tokens": int(os.getenv("CHUNK_SIZE_TOKENS", "1024")),
        "chunk_overlap_tokens": int(os.getenv("CHUNK_OVERLAP_TOKENS", "200")),
        "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "3072")),
        "top_k_dense": int(os.getenv("TOP_K_DENSE", "10")),
        "top_k_sparse": int(os.getenv("TOP_K_SPARSE", "10")),
        "top_k_rerank": int(os.getenv("TOP_K_RERANK", "5"))
    }
    
    # Validate required configuration
    if not config["openai_api_key"]:
        logger.error("OPENAI_API_KEY is required")
        return
    
    # Create documents directory if it doesn't exist
    os.makedirs(config["source_paths"][0], exist_ok=True)
    
    try:
        # Initialize Knowledge Base API
        logger.info("Initializing RAGEngine Knowledge Base...")
        kb = KnowledgeBaseAPI(config)
        
        # Display system information
        logger.info("System Information:")
        system_info = kb.get_system_info()
        logger.info(f"Collection: {system_info['config']['collection_name']}")
        logger.info(f"Chunking Strategy: {system_info['config']['chunking_strategy']}")
        logger.info(f"Vector Dimensions: {system_info['config']['vector_dimensions']}")
        
        # Check if there are documents to process
        processed_docs = kb.get_processed_documents()
        logger.info(f"Previously processed documents: {len(processed_docs)}")
        
        # Ingest documents
        logger.info("Starting document ingestion...")
        stats = kb.ingest_documents()
        logger.info(f"Ingestion Results:")
        logger.info(f"  Documents processed: {stats['documents_processed']}")
        logger.info(f"  Chunks created: {stats['chunks_created']}")
        logger.info(f"  Embeddings generated: {stats['embeddings_generated']}")
        
        if stats['errors']:
            logger.warning(f"  Errors: {len(stats['errors'])}")
            for error in stats['errors']:
                logger.warning(f"    {error}")
        
        # Example searches
        example_queries = [
            "What is machine learning?",
            "Explain neural networks",
            "How does natural language processing work?",
            "What are the applications of artificial intelligence?"
        ]
        
        for query in example_queries:
            logger.info(f"\n--- Searching: {query} ---")
            
            # Search with synthesis (if Google API key available)
            synthesize = config["google_api_key"] is not None
            results = kb.search(query, synthesize=synthesize)
            
            logger.info(f"Found {results['num_results']} relevant contexts")
            
            # Display top context
            if results['contexts']:
                top_context = results['contexts'][0]
                logger.info(f"Top result (score: {top_context.get('rerank_score', top_context['initial_score']):.3f}):")
                logger.info(f"  Text: {top_context['text'][:200]}...")
            
            # Display synthesis if available
            if results['synthesis']:
                logger.info(f"Synthesis: {results['synthesis']['summary'][:200]}...")
                if results['synthesis']['key_concepts']:
                    logger.info(f"Key concepts: {len(results['synthesis']['key_concepts'])}")
            
            logger.info("-" * 50)
        
        # Demonstrate different search types
        logger.info("\n--- Demonstrating Search Types ---")
        
        test_query = "artificial intelligence"
        
        # Dense search only
        dense_results = kb.dense_search(test_query, top_k=5)
        logger.info(f"Dense search returned {dense_results['num_results']} results")
        
        # Sparse search only
        sparse_results = kb.sparse_search(test_query, top_k=5)
        logger.info(f"Sparse search returned {sparse_results['num_results']} results")
        
        # Hybrid search without synthesis
        hybrid_results = kb.search(test_query, synthesize=False, search_only=True)
        logger.info(f"Hybrid search returned {hybrid_results['num_results']} results")
        
        logger.info("\nRAGEngine demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logger.add("ragengine.log", rotation="10 MB", level="INFO")
    
    main() 