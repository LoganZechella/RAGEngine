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
    DocumentCollection, CollectionMetadata, ParsedDocument, TextChunk
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
            query_chunk = TextChunk(
                chunk_id="query",
                document_id="query",
                text=query,
                metadata={}
            )
            embedded_query = self.embedding_generator.generate_embeddings([query_chunk])
            if not embedded_query or embedded_query[0].embedding_vector is None:
                raise ValueError("Failed to generate query embedding")
            query_embedding = embedded_query[0].embedding_vector
            
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
                    synthesis = self.rag.deep_analyzer.synthesize_knowledge({"query": query}, top_contexts)
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
            total_documents = 0
            total_chunks = 0
            
            for collection in DocumentCollection:
                collection_name = collection.value
                raw_stats = stats.get(collection_name, {})
                display_info = collection_info.get(collection, {})
                
                points_count = raw_stats.get("points_count", 0)
                
                # Get unique document count for this collection
                document_count = self._get_unique_document_count(collection_name)
                
                # Determine status based on points and errors
                if raw_stats.get("error"):
                    status = "error"
                elif points_count > 0:
                    status = "green"  # Template expects 'green' for active
                else:
                    status = "empty"  # Could be 'yellow' for warning if desired
                
                enhanced_stats[collection_name] = {
                    "display_name": display_info.get("display_name", collection_name),
                    "description": display_info.get("description", ""),
                    "document_count": document_count,      # Actual unique documents
                    "chunk_count": points_count,           # Number of chunks/points in the collection
                    "vector_count": points_count,          # Same as chunk count since each chunk has one vector
                    "vector_size": raw_stats.get("vector_size"),
                    "status": status,
                    "last_updated": None,  # Could be enhanced to track this
                    "error": raw_stats.get("error")
                }
                
                total_documents += document_count
                total_chunks += points_count
            
            return {
                "collections": enhanced_stats,
                "total_collections": len(enhanced_stats),
                "total_documents": total_documents,        # Sum of unique documents across all collections
                "total_chunks": total_chunks,              # Sum of all chunks across all collections
                "total_vectors": total_chunks              # Same as total chunks
            }
        except Exception as e:
            logger.error(f"Error getting collection statistics: {e}")
            return {"error": str(e), "collections": {}}
    
    def _get_unique_document_count(self, collection_name: str) -> int:
        """Get the count of unique documents in a collection."""
        try:
            # Use Qdrant's aggregation feature to count unique document_ids
            from qdrant_client.http import models as qmodels
            
            # Scroll through all points and collect unique document_ids
            unique_docs = set()
            offset = None
            
            while True:
                try:
                    result, next_offset = self.vector_db.client.scroll(
                        collection_name=collection_name,
                        limit=100,  # Process in batches
                        offset=offset,
                        with_payload=["document_id"],
                        with_vectors=False  # We don't need vectors
                    )
                    
                    for point in result:
                        if point.payload and "document_id" in point.payload:
                            unique_docs.add(point.payload["document_id"])
                    
                    if next_offset is None:
                        break
                    offset = next_offset
                    
                except Exception as e:
                    logger.warning(f"Error scrolling collection {collection_name}: {e}")
                    # If we can't scroll, return 0
                    return 0
            
            return len(unique_docs)
            
        except Exception as e:
            logger.error(f"Error getting unique document count for {collection_name}: {e}")
            # Fallback: return 0 if we can't count documents
            return 0
    
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
            
            logger.info(f"Document {file_path} assigned to collection: {collection.value}")
            
            # Process the document with collection assignment
            stats = self.ingestion.process_single_document_to_collection(
                file_path=file_path,
                target_collection=collection
            )
            
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
    
    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform dense vector search only."""
        try:
            query_chunk = TextChunk(
                chunk_id="query",
                document_id="query",
                text=query,
                metadata={}
            )
            embedded_query = self.embedding_generator.generate_embeddings([query_chunk])
            if not embedded_query or embedded_query[0].embedding_vector is None:
                raise ValueError("Failed to generate query embedding")
            query_embedding = embedded_query[0].embedding_vector
            contexts = self.vector_db.dense_vector_search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            return {
                "query": query,
                "search_type": "dense",
                "num_results": len(contexts),
                "contexts": [
                    {
                        "chunk_id": ctx.chunk_id,
                        "document_id": ctx.document_id,
                        "text": ctx.text,
                        "initial_score": ctx.initial_score,
                        "collection": ctx.collection.value if ctx.collection else None,
                        "metadata": ctx.metadata
                    }
                    for ctx in contexts
                ]
            }
        except Exception as e:
            logger.error(f"Dense search error: {e}")
            return {"error": str(e), "query": query, "num_results": 0, "contexts": []}

    def sparse_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform sparse keyword search only."""
        try:
            # For sparse search, we'll use the RAG engine's sparse search if available
            if hasattr(self.rag, 'sparse_search_only'):
                contexts = self.rag.sparse_search_only(query, top_k, filters)
            else:
                # Fallback to dense search
                logger.warning("Sparse search not available, falling back to dense search")
                return self.dense_search(query, top_k, filters)
            
            return {
                "query": query,
                "search_type": "sparse",
                "num_results": len(contexts),
                "contexts": [
                    {
                        "chunk_id": ctx.chunk_id,
                        "document_id": ctx.document_id,
                        "text": ctx.text,
                        "initial_score": ctx.initial_score,
                        "collection": ctx.collection.value if ctx.collection else None,
                        "metadata": ctx.metadata
                    }
                    for ctx in contexts
                ]
            }
        except Exception as e:
            logger.error(f"Sparse search error: {e}")
            return {"error": str(e), "query": query, "num_results": 0, "contexts": []}

    def get_processed_documents(self) -> Dict[str, Any]:
        """Get information about processed documents."""
        try:
            if hasattr(self.ingestion, 'document_manager'):
                return self.ingestion.document_manager.get_processed_documents()
            else:
                # Fallback - return empty dict for now
                logger.warning("Document manager not available in ingestion system")
                return {}
        except Exception as e:
            logger.error(f"Error getting processed documents: {e}")
            return {}

    def delete_collection(self) -> bool:
        """Delete the vector database collection."""
        try:
            success = True
            for collection in DocumentCollection:
                collection_success = self.vector_db.delete_collection_data(collection)
                if not collection_success:
                    success = False
                    logger.error(f"Failed to delete collection: {collection.value}")
            return success
        except Exception as e:
            logger.error(f"Error deleting collections: {e}")
            return False

    def recreate_collection(self) -> bool:
        """Delete and recreate all collections with fresh configuration."""
        try:
            # Delete all collections
            delete_success = self.delete_collection()
            if not delete_success:
                logger.warning("Some collections failed to delete during recreation")
            
            # Recreate collections
            self.vector_db._ensure_all_collections_exist()
            logger.info("All collections recreated successfully")
            return True
        except Exception as e:
            logger.error(f"Error recreating collections: {e}")
            return False

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a specific document across all collections."""
        try:
            total_deleted = 0
            for collection in DocumentCollection:
                try:
                    # Use collection-specific deletion if available
                    deleted_count = self.vector_db.delete_by_document_id(collection.value, document_id)
                    total_deleted += deleted_count
                    logger.info(f"Deleted {deleted_count} chunks from {collection.value}")
                except Exception as e:
                    logger.error(f"Error deleting from collection {collection.value}: {e}")
            
            logger.info(f"Total deleted {total_deleted} chunks for document {document_id}")
            return total_deleted
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return 0

    # Backward compatibility methods
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Backward compatible search method."""
        return self.search_collections(query, **kwargs)
    
    def ingest_documents(self) -> Dict[str, Any]:
        """Backward compatible bulk ingestion."""
        return self.ingestion.process_documents()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information including all collections."""
        try:
            # Get collection statistics
            collection_stats = self.get_collection_statistics()
            
            # Get detailed vector DB info for default collection
            default_collection_info = {}
            try:
                default_info = self.vector_db.client.get_collection(self.vector_db.default_collection.value)
                default_collection_info = {
                    "collection_name": self.vector_db.default_collection.value,
                    "points_count": default_info.points_count,
                    "vector_size": default_info.config.params.vectors.size,
                    "distance": str(default_info.config.params.vectors.distance),
                    "indexed_vectors_count": default_info.vectors_count,
                    "name": self.vector_db.default_collection.value
                }
            except Exception as e:
                logger.warning(f"Could not get detailed vector DB info: {e}")
                default_collection_info = {
                    "collection_name": self.vector_db.default_collection.value,
                    "points_count": 0,
                    "vector_size": self.config.get("vector_dimensions", 1536),
                    "distance": "cosine",
                    "name": self.vector_db.default_collection.value
                }
            
            # Build comprehensive system info
            return {
                "config": {
                    "collection_name": self.vector_db.default_collection.value,
                    "chunking_strategy": self.config.get("chunking_strategy"),
                    "vector_dimensions": self.config.get("vector_dimensions"),
                    "source_paths": self.config.get("source_paths", []),
                    "enable_content_filtering": self.config.get("enable_content_filtering", True),
                    "enable_document_type_detection": self.config.get("enable_document_type_detection", True),
                    "default_content_type": self.config.get("default_content_type", "auto")
                },
                "collections": collection_stats,
                "ingestion": {
                    "max_chunk_size_tokens": self.config.get("chunk_size_tokens", 512),
                    "chunk_overlap_tokens": self.config.get("chunk_overlap_tokens", 100),
                    "chunking_strategy": self.config.get("chunking_strategy")
                },
                "rag_engine": {
                    "hybrid_searcher": {
                        "top_k_dense": self.config.get("top_k_dense", 10),
                        "top_k_sparse": self.config.get("top_k_sparse", 10),
                        "rrf_k": 60  # Reciprocal rank fusion constant
                    },
                    "reranker": {
                        "model": "o4-mini" if self.config.get("openai_api_key") else None,
                        "api_available": bool(self.config.get("openai_api_key"))
                    } if hasattr(self.rag, 'reranker') else None,
                    "deep_analyzer": {
                        "model": "gemini-2.5-pro-preview" if self.config.get("google_api_key") else None,
                        "api_available": bool(self.config.get("google_api_key"))
                    } if hasattr(self.rag, 'deep_analyzer') else None
                },
                "vector_db": default_collection_info
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                "config": self.config,
                "collections": self.get_collection_statistics(),
                "error": str(e)
            }