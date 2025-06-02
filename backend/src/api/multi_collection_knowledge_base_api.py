"""
Multi-Collection Knowledge Base API for RAGEngine.
Enhanced API with collection-specific search and management capabilities.
"""

import asyncio
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

from ..models.data_models import (
    EmbeddedChunk, RetrievedContext, SynthesizedKnowledge, 
    DocumentCollection, CollectionMetadata, ParsedDocument
)
from ..ingestion.collection_manager import CollectionManager
from ..ingestion.multi_collection_vector_db_manager import MultiCollectionVectorDBManager
from ..ingestion.embedding_generator import EmbeddingGenerator
from ..ingestion.knowledge_ingestion import KnowledgeIngestion
from ..rag.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


class MultiCollectionKnowledgeBaseAPI:
    """Enhanced Knowledge Base API with multi-collection support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_manager = CollectionManager()
        
        # Initialize enhanced vector DB manager
        self.vector_db = MultiCollectionVectorDBManager(
            url=config["qdrant_url"],
            api_key=config.get("qdrant_api_key"),
            default_collection=DocumentCollection(config.get("default_collection", "current_documents")),
            vector_dimensions=config.get("vector_dimensions", 1536)
        )
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            api_key=config["openai_api_key"],
            dimensions=config.get("vector_dimensions", 1536)
        )
        
        # Initialize ingestion with collection support
        self.ingestion = KnowledgeIngestion(
            source_paths=config["source_paths"],
            openai_api_key=config["openai_api_key"],
            qdrant_url=config["qdrant_url"],
            qdrant_api_key=config.get("qdrant_api_key"),
            collection_manager=self.collection_manager,
            vector_db_manager=self.vector_db,
            chunking_strategy=config.get("chunking_strategy", "paragraph"),
            max_chunk_size_tokens=config.get("chunk_size_tokens", 512),
            chunk_overlap_tokens=config.get("chunk_overlap_tokens", 100),
            vector_dimensions=config.get("vector_dimensions", 1536)
        )
        
        # Initialize RAG engine
        self.rag = RAGEngine(
            vector_db=self.vector_db,
            embedding_generator=self.embedding_generator,
            openai_api_key=config.get("openai_api_key"),
            google_api_key=config.get("google_api_key"),
            top_k_dense=config.get("top_k_dense", 10),
            top_k_sparse=config.get("top_k_sparse", 10),
            top_k_rerank=config.get("top_k_rerank", 5)
        )
        
        logger.info("Multi-collection KnowledgeBaseAPI initialized successfully")
    
    def search_collections(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        mode: str = "hybrid",
        top_k: int = 10,
        synthesize: bool = True
    ) -> Dict[str, Any]:
        """
        Search across specified collections.
        """
        try:
            # Process collection selection
            if not collections or "all" in collections:
                target_collections = list(DocumentCollection)
            else:
                target_collections = []
                for coll_name in collections:
                    collection = self.collection_manager.validate_collection(coll_name)
                    if collection:
                        target_collections.append(collection)
            
            if not target_collections:
                target_collections = [DocumentCollection.CURRENT]
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search across collections
            collection_results = self.vector_db.search_collections(
                query_embedding=query_embedding,
                collections=target_collections,
                top_k=top_k
            )
            
            # Combine results
            all_contexts = []
            results_by_collection = {}
            
            for collection, contexts in collection_results.items():
                all_contexts.extend(contexts)
                results_by_collection[collection.value] = len(contexts)
            
            # Sort by score and limit
            all_contexts.sort(key=lambda x: x.initial_score, reverse=True)
            top_contexts = all_contexts[:top_k]
            
            # Rerank if available
            if hasattr(self.rag, 'reranker') and len(top_contexts) > 1:
                try:
                    top_contexts = self.rag.reranker.rerank_contexts(query, top_contexts)
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
            
            # Synthesize if requested
            synthesis = None
            if synthesize and top_contexts and hasattr(self.rag, 'deep_analyzer'):
                try:
                    synthesis = self.rag.deep_analyzer.synthesize_knowledge(query, top_contexts)
                except Exception as e:
                    logger.warning(f"Knowledge synthesis failed: {e}")
            
            return {
                "query": query,
                "mode": mode,
                "collections_searched": [c.value for c in target_collections],
                "total_results": len(top_contexts),
                "results_by_collection": results_by_collection,
                "contexts": [
                    {
                        "chunk_id": ctx.chunk_id,
                        "document_id": ctx.document_id,
                        "text": ctx.text,
                        "initial_score": ctx.initial_score,
                        "rerank_score": ctx.rerank_score,
                        "collection": ctx.collection.value if ctx.collection else None,
                        "metadata": ctx.metadata
                    }
                    for ctx in top_contexts
                ],
                "synthesized_knowledge": synthesis
            }
            
        except Exception as e:
            logger.error(f"Error in collection search: {e}")
            return {
                "error": str(e),
                "query": query,
                "collections_searched": [],
                "total_results": 0,
                "contexts": []
            }
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        try:
            stats = self.vector_db.get_all_collections_info()
            collection_info = self.collection_manager.get_collection_info()
            
            # Combine stats with display information
            enhanced_stats = {}
            for collection in DocumentCollection:
                collection_name = collection.value
                raw_stats = stats.get(collection_name, {})
                display_info = collection_info.get(collection, {})
                
                enhanced_stats[collection_name] = {
                    "display_name": display_info.get("display_name", collection_name),
                    "description": display_info.get("description", ""),
                    "document_count": raw_stats.get("points_count", 0),
                    "vector_size": raw_stats.get("vector_size"),
                    "status": "active" if raw_stats.get("points_count", 0) > 0 else "empty"
                }
            
            return {
                "collections": enhanced_stats,
                "total_collections": len(enhanced_stats),
                "total_documents": sum(s.get("document_count", 0) for s in enhanced_stats.values())
            }
        except Exception as e:
            logger.error(f"Error getting collection statistics: {e}")
            return {"error": str(e), "collections": {}}
    
    def ingest_document_to_collection(
        self,
        file_path: str,
        target_collection: Optional[str] = None,
        auto_classify: bool = True
    ) -> Dict[str, Any]:
        """Ingest document with collection assignment."""
        try:
            # Determine target collection
            if target_collection:
                collection = self.collection_manager.validate_collection(target_collection)
                if not collection:
                    return {
                        "status": "error",
                        "error": f"Invalid collection: {target_collection}",
                        "file_path": file_path
                    }
            elif auto_classify:
                # Read sample for classification
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content_sample = f.read(2000)
                except:
                    content_sample = ""
                
                collection = self.collection_manager.classify_document(
                    file_path, content_sample
                )
            else:
                collection = DocumentCollection.CURRENT
            
            # Process the document
            stats = self.ingestion.process_single_document(file_path)
            
            # Update chunks with collection assignment
            if stats.get("chunks_created", 0) > 0:
                # This would need enhancement in the ingestion process
                # to properly assign collections to chunks
                pass
            
            return {
                "status": "success" if not stats.get("errors") else "error",
                "collection": collection.value,
                "file_path": file_path,
                "chunks_processed": stats.get("chunks_created", 0),
                "errors": stats.get("errors", [])
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path
            }
    
    def clear_collection(self, collection_name: str) -> Dict[str, Any]:
        """Clear all documents from a collection."""
        try:
            collection = self.collection_manager.validate_collection(collection_name)
            if not collection:
                return {
                    "status": "error",
                    "error": f"Invalid collection: {collection_name}"
                }
            
            success = self.vector_db.delete_collection_data(collection)
            
            return {
                "status": "success" if success else "error",
                "collection": collection_name,
                "message": f"Collection {collection_name} cleared" if success else "Failed to clear collection"
            }
        except Exception as e:
            logger.error(f"Error clearing collection {collection_name}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    # Backward compatibility methods
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Backward compatible search method."""
        return self.search_collections(query, **kwargs)
    
    def ingest_documents(self) -> Dict[str, Any]:
        """Backward compatible bulk ingestion."""
        return self.ingestion.process_documents()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "collections": self.get_collection_statistics(),
            "config": self.config
        }