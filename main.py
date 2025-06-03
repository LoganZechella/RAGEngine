#!/usr/bin/env python3
"""
RAGEngine - Standalone Knowledge Base System
Main example demonstrating usage of the RAGEngine system.
"""

import os
from dotenv import load_dotenv
from loguru import logger
from backend.src.api.multi_collection_knowledge_base_api import MultiCollectionKnowledgeBaseAPI

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
        "default_collection": os.getenv("DEFAULT_COLLECTION", "current_documents"),
        "source_paths": [os.getenv("SOURCE_DOCUMENTS_DIR", "./documents")],
        "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "paragraph"),
        "chunk_size_tokens": int(os.getenv("CHUNK_SIZE_TOKENS", "512")),
        "chunk_overlap_tokens": int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
        "vector_dimensions": int(os.getenv("VECTOR_DIMENSIONS", "1536")),
        "top_k_dense": int(os.getenv("TOP_K_DENSE", "10")),
        "top_k_sparse": int(os.getenv("TOP_K_SPARSE", "10")),
        "top_k_rerank": int(os.getenv("TOP_K_RERANK", "5")),
        "auto_collection_assignment": os.getenv("AUTO_COLLECTION_ASSIGNMENT", "true").lower() == "true"
    }
    
    # Validate required configuration
    if not config["openai_api_key"]:
        logger.error("OPENAI_API_KEY is required")
        return
    
    # Create documents directory if it doesn't exist
    os.makedirs(config["source_paths"][0], exist_ok=True)
    
    try:
        # Initialize Knowledge Base API
        logger.info("Initializing RAGEngine Multi-Collection Knowledge Base...")
        kb = MultiCollectionKnowledgeBaseAPI(config)
        
        # Display system information
        logger.info("System Information:")
        system_info = kb.get_system_info()
        logger.info(f"Default Collection: {config['default_collection']}")
        logger.info(f"Chunking Strategy: {config['chunking_strategy']}")
        logger.info(f"Vector Dimensions: {config['vector_dimensions']}")
        
        # Check collection statistics
        collection_stats = kb.get_collection_statistics()
        logger.info(f"Total Collections: {collection_stats.get('total_collections', 0)}")
        logger.info(f"Total Documents: {collection_stats.get('total_documents', 0)}")
        
        # Check if there are documents to process
        try:
            processed_docs = kb.get_processed_documents()
            logger.info(f"Previously processed documents: {len(processed_docs)}")
        except Exception as e:
            logger.warning(f"Could not get processed documents: {e}")
        
        # Ingest documents
        logger.info("Starting document ingestion...")
        stats = kb.ingest_documents()
        logger.info(f"Ingestion Results:")
        logger.info(f"  Documents processed: {stats.get('documents_processed', 0)}")
        logger.info(f"  Chunks created: {stats.get('chunks_created', 0)}")
        logger.info(f"  Embeddings generated: {stats.get('embeddings_generated', 0)}")
        
        if stats.get('errors'):
            logger.warning(f"  Errors: {len(stats['errors'])}")
            for error in stats['errors']:
                logger.warning(f"    {error}")
        
        # Example searches with multi-collection support
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
            results = kb.search_collections(query, synthesize=synthesize)
            
            logger.info(f"Found {results.get('total_results', 0)} relevant contexts")
            logger.info(f"Collections searched: {results.get('collections_searched', [])}")
            
            # Display top context
            if results.get('contexts'):
                top_context = results['contexts'][0]
                logger.info(f"Top result (score: {top_context.get('rerank_score', top_context.get('initial_score', 0)):.3f}):")
                logger.info(f"  Collection: {top_context.get('collection', 'Unknown')}")
                logger.info(f"  Text: {top_context['text'][:200]}...")
            
            # Display synthesis if available
            if results.get('synthesized_knowledge'):
                synthesis = results['synthesized_knowledge']
                if isinstance(synthesis, dict) and synthesis.get('summary'):
                    logger.info(f"Synthesis: {synthesis['summary'][:200]}...")
                elif hasattr(synthesis, 'summary'):
                    logger.info(f"Synthesis: {synthesis.summary[:200]}...")
            
            logger.info("-" * 50)
        
        # Demonstrate different search types
        logger.info("\n--- Demonstrating Search Types ---")
        
        test_query = "artificial intelligence"
        
        # Dense search only
        try:
            dense_results = kb.dense_search(test_query, top_k=5)
            logger.info(f"Dense search returned {dense_results.get('num_results', 0)} results")
        except Exception as e:
            logger.warning(f"Dense search failed: {e}")
        
        # Sparse search only
        try:
            sparse_results = kb.sparse_search(test_query, top_k=5)
            logger.info(f"Sparse search returned {sparse_results.get('num_results', 0)} results")
        except Exception as e:
            logger.warning(f"Sparse search failed: {e}")
        
        # Multi-collection search without synthesis
        hybrid_results = kb.search_collections(test_query, synthesize=False)
        logger.info(f"Multi-collection search returned {hybrid_results.get('total_results', 0)} results")
        
        logger.info("\nRAGEngine multi-collection demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logger.add("ragengine.log", rotation="10 MB", level="INFO")
    
    main() 