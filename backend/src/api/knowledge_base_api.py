from typing import List, Dict, Any, Optional
from loguru import logger

from backend.src.ingestion.knowledge_ingestion import KnowledgeIngestion
from backend.src.ingestion.embedding_generator import EmbeddingGenerator
from backend.src.ingestion.vector_db_manager import VectorDBManager
from backend.src.rag.rag_engine import RAGEngine

class KnowledgeBaseAPI:
    """Enhanced API interface for the knowledge base with content filtering support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize shared components
        self.embedding_generator = EmbeddingGenerator(
            api_key=config["openai_api_key"],
            dimensions=config.get("vector_dimensions", 1536)
        )
        
        self.vector_db = VectorDBManager(
            url=config["qdrant_url"],
            api_key=config.get("qdrant_api_key"),
            collection_name=config.get("collection_name", "knowledge_base"),
            vector_dimensions=config.get("vector_dimensions", 1536)
        )
        
        # Initialize ingestion with enhanced content filtering support
        self.ingestion = KnowledgeIngestion(
            source_paths=config["source_paths"],
            openai_api_key=config["openai_api_key"],
            qdrant_url=config["qdrant_url"],
            qdrant_api_key=config.get("qdrant_api_key"),
            collection_name=config.get("collection_name", "knowledge_base"),
            chunking_strategy=config.get("chunking_strategy", "paragraph"),
            max_chunk_size_tokens=config.get("chunk_size_tokens", 512),
            chunk_overlap_tokens=config.get("chunk_overlap_tokens", 100),
            vector_dimensions=config.get("vector_dimensions", 1536),
            enable_content_filtering=config.get("enable_content_filtering", True),
            enable_deduplication=config.get("enable_deduplication", True),
            content_type=config.get("default_content_type", "auto"),
            enable_auto_detection=config.get("enable_document_type_detection", True)
        )
        
        # Initialize RAG
        self.rag = RAGEngine(
            vector_db=self.vector_db,
            embedding_generator=self.embedding_generator,
            openai_api_key=config.get("openai_api_key"),
            google_api_key=config.get("google_api_key"),
            top_k_dense=config.get("top_k_dense", 10),
            top_k_sparse=config.get("top_k_sparse", 10),
            top_k_rerank=config.get("top_k_rerank", 5)
        )
        
        logger.info("KnowledgeBaseAPI initialized successfully with content filtering support")
    
    def ingest_documents(self) -> Dict[str, Any]:
        """Ingest all documents from configured sources with content filtering."""
        logger.info("Starting document ingestion with content filtering...")
        return self.ingestion.process_documents()
    
    def ingest_single_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single document with content filtering."""
        logger.info(f"Ingesting single document with content filtering: {file_path}")
        return self.ingestion.process_single_document(file_path)
    
    def search(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        synthesize: bool = True,
        search_only: bool = False
    ) -> Dict[str, Any]:
        """Search the knowledge base with enhanced metadata."""
        logger.info(f"Searching for: {query}")
        
        if search_only:
            # Search without synthesis
            contexts = self.rag.search_only(query, filters)
            return {
                "query": query,
                "num_results": len(contexts),
                "contexts": [
                    {
                        "chunk_id": ctx.chunk_id,
                        "document_id": ctx.document_id,
                        "text": ctx.text,
                        "initial_score": ctx.initial_score,
                        "rerank_score": ctx.rerank_score,
                        "metadata": ctx.metadata
                    }
                    for ctx in contexts
                ],
                "synthesis": None
            }
        else:
            # Full retrieve and analyze
            contexts, synthesis = self.rag.retrieve_and_analyze(query, filters, synthesize)
            
            return {
                "query": query,
                "num_results": len(contexts),
                "contexts": [
                    {
                        "chunk_id": ctx.chunk_id,
                        "document_id": ctx.document_id,
                        "text": ctx.text,
                        "initial_score": ctx.initial_score,
                        "rerank_score": ctx.rerank_score,
                        "metadata": ctx.metadata
                    }
                    for ctx in contexts
                ],
                "synthesis": synthesis if synthesis and synthesize else None
            }
    
    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform dense vector search only."""
        logger.info(f"Dense search for: {query}")
        contexts = self.rag.dense_search_only(query, top_k, filters)
        
        return {
            "query": query,
            "search_type": "dense",
            "num_results": len(contexts),
            "contexts": [
                {
                    "chunk_id": ctx.chunk_id,
                    "document_id": ctx.document_id,
                    "text": ctx.text,
                    "score": ctx.initial_score,
                    "metadata": ctx.metadata
                }
                for ctx in contexts
            ]
        }
    
    def sparse_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform sparse keyword search only."""
        logger.info(f"Sparse search for: {query}")
        contexts = self.rag.sparse_search_only(query, top_k, filters)
        
        return {
            "query": query,
            "search_type": "sparse",
            "num_results": len(contexts),
            "contexts": [
                {
                    "chunk_id": ctx.chunk_id,
                    "document_id": ctx.document_id,
                    "text": ctx.text,
                    "score": ctx.initial_score,
                    "metadata": ctx.metadata
                }
                for ctx in contexts
            ]
        }
    
    def batch_search(
        self,
        queries: List[str],
        filters: Optional[List[Dict[str, Any]]] = None,
        synthesize: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple queries in batch."""
        logger.info(f"Batch search for {len(queries)} queries")
        results = self.rag.batch_retrieve_and_analyze(queries, filters, synthesize)
        
        formatted_results = []
        for i, (contexts, synthesis) in enumerate(results):
            formatted_results.append({
                "query": queries[i],
                "num_results": len(contexts),
                "contexts": [
                    {
                        "chunk_id": ctx.chunk_id,
                        "document_id": ctx.document_id,
                        "text": ctx.text,
                        "initial_score": ctx.initial_score,
                        "rerank_score": ctx.rerank_score,
                        "metadata": ctx.metadata
                    }
                    for ctx in contexts
                ],
                "synthesis": synthesis if synthesis and synthesize else None
            })
        
        return formatted_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information including content filtering details."""
        return {
            "config": {
                "collection_name": self.config.get("collection_name"),
                "chunking_strategy": self.config.get("chunking_strategy"),
                "vector_dimensions": self.config.get("vector_dimensions"),
                "source_paths": self.config.get("source_paths"),
                "enable_content_filtering": self.config.get("enable_content_filtering"),
                "enable_document_type_detection": self.config.get("enable_document_type_detection"),
                "default_content_type": self.config.get("default_content_type")
            },
            "content_filtering": {
                "enabled": self.config.get("enable_content_filtering", True),
                "auto_detection": self.config.get("enable_document_type_detection", True),
                "default_type": self.config.get("default_content_type", "auto"),
                "policy_aggressive": self.config.get("policy_filter_aggressive", False),
                "form_preserve_structure": self.config.get("form_filter_preserve_structure", True),
                "scientific_legacy_mode": self.config.get("scientific_filter_legacy_mode", False)
            },
            "ingestion": self.ingestion.get_ingestion_info(),
            "rag_engine": self.rag.get_engine_info(),
            "vector_db": self.vector_db.get_collection_info()
        }
    
    def get_processed_documents(self) -> Dict[str, Any]:
        """Get information about processed documents with filtering stats."""
        return self.ingestion.document_manager.get_processed_documents()
    
    def get_content_filtering_stats(self) -> Dict[str, Any]:
        """Get aggregated content filtering statistics."""
        processed_docs = self.get_processed_documents()
        
        total_stats = {
            'total_chunks_processed': 0,
            'chunks_filtered_out': 0,
            'chunks_cleaned': 0,
            'boilerplate_removed': 0,
            'duplicates_removed': 0
        }
        
        content_type_counts = {}
        docs_with_stats = 0
        
        for doc_path, metadata in processed_docs.items():
            if 'filtering_stats' in metadata:
                docs_with_stats += 1
                stats = metadata['filtering_stats']
                
                for key in total_stats:
                    if key in stats:
                        total_stats[key] += stats[key]
                
                if 'content_type_detected' in metadata:
                    content_type = metadata['content_type_detected']
                    content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
        
        overall_filter_rate = 0
        if total_stats['total_chunks_processed'] > 0:
            overall_filter_rate = (total_stats['chunks_filtered_out'] / total_stats['total_chunks_processed']) * 100
        
        return {
            "overall_stats": total_stats,
            "overall_filter_rate_percent": round(overall_filter_rate, 2),
            "content_type_distribution": content_type_counts,
            "documents_with_filtering": docs_with_stats,
            "total_documents": len(processed_docs)
        }
    
    def delete_collection(self) -> bool:
        """Delete the vector database collection."""
        logger.warning("Deleting vector database collection")
        return self.vector_db.delete_collection()
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection without deleting the collection itself."""
        logger.warning("Clearing all data from vector database collection")
        return self.vector_db.clear_collection()
    
    def recreate_collection(self) -> bool:
        """Delete and recreate the collection with fresh configuration."""
        logger.warning("Recreating vector database collection")
        return self.vector_db.recreate_collection()
    
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a specific document."""
        logger.info(f"Deleting document: {document_id}")
        return self.vector_db.delete_by_document_id(document_id)
    
    def update_content_filtering_config(self, updates: Dict[str, Any]) -> bool:
        """Update content filtering configuration (requires restart for full effect)."""
        try:
            for key, value in updates.items():
                if key in self.config:
                    self.config[key] = value
                    logger.info(f"Updated config: {key} = {value}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to update content filtering config: {e}")
            return False