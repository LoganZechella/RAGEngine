from typing import List, Dict, Any, Optional
from loguru import logger

from src.ingestion.knowledge_ingestion import KnowledgeIngestion
from src.ingestion.embedding_generator import EmbeddingGenerator
from src.ingestion.vector_db_manager import VectorDBManager
from src.rag.rag_engine import RAGEngine

class KnowledgeBaseAPI:
    """Simple API interface for the knowledge base."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize shared components
        self.embedding_generator = EmbeddingGenerator(
            api_key=config["openai_api_key"],
            dimensions=config.get("vector_dimensions", 3072)
        )
        
        self.vector_db = VectorDBManager(
            url=config["qdrant_url"],
            api_key=config.get("qdrant_api_key"),
            collection_name=config.get("collection_name", "knowledge_base"),
            vector_dimensions=config.get("vector_dimensions", 3072)
        )
        
        # Initialize ingestion
        self.ingestion = KnowledgeIngestion(
            source_paths=config["source_paths"],
            openai_api_key=config["openai_api_key"],
            qdrant_url=config["qdrant_url"],
            qdrant_api_key=config.get("qdrant_api_key"),
            collection_name=config.get("collection_name", "knowledge_base"),
            chunking_strategy=config.get("chunking_strategy", "paragraph"),
            max_chunk_size_tokens=config.get("chunk_size_tokens", 1024),
            chunk_overlap_tokens=config.get("chunk_overlap_tokens", 200)
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
        
        logger.info("KnowledgeBaseAPI initialized successfully")
    
    def ingest_documents(self) -> Dict[str, Any]:
        """Ingest all documents from configured sources."""
        logger.info("Starting document ingestion...")
        return self.ingestion.process_documents()
    
    def ingest_single_document(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single document."""
        logger.info(f"Ingesting single document: {file_path}")
        return self.ingestion.process_single_document(file_path)
    
    def search(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        synthesize: bool = True,
        search_only: bool = False
    ) -> Dict[str, Any]:
        """Search the knowledge base."""
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
                "synthesis": synthesis.dict() if synthesis and synthesize else None
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
                "synthesis": synthesis.dict() if synthesis and synthesize else None
            })
        
        return formatted_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "config": {
                "collection_name": self.config.get("collection_name"),
                "chunking_strategy": self.config.get("chunking_strategy"),
                "vector_dimensions": self.config.get("vector_dimensions"),
                "source_paths": self.config.get("source_paths")
            },
            "ingestion": self.ingestion.get_ingestion_info(),
            "rag_engine": self.rag.get_engine_info(),
            "vector_db": self.vector_db.get_collection_info()
        }
    
    def get_processed_documents(self) -> Dict[str, Any]:
        """Get information about processed documents."""
        return self.ingestion.document_manager.get_processed_documents()
    
    def delete_collection(self) -> bool:
        """Delete the vector database collection."""
        logger.warning("Deleting vector database collection")
        return self.vector_db.delete_collection() 