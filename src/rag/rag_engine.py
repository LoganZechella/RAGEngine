from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from src.rag.hybrid_searcher import HybridSearcher
from src.rag.reranker import ReRanker
from src.rag.deep_analyzer import DeepAnalyzer
from src.models.data_models import RetrievedContext, SynthesizedKnowledge

class RAGEngine:
    """Orchestrates retrieval and knowledge synthesis."""
    
    def __init__(
        self,
        vector_db,
        embedding_generator,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        top_k_rerank: int = 5
    ):
        self.hybrid_searcher = HybridSearcher(
            vector_db=vector_db,
            embedding_generator=embedding_generator,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse
        )
        
        self.reranker = None
        if openai_api_key:
            self.reranker = ReRanker(api_key=openai_api_key)
        
        self.deep_analyzer = None
        if google_api_key:
            self.deep_analyzer = DeepAnalyzer(api_key=openai_api_key)
        
        self.top_k_rerank = top_k_rerank
        
        logger.info(f"Initialized RAGEngine with reranker={self.reranker is not None}, analyzer={self.deep_analyzer is not None}")
    
    def retrieve_and_analyze(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        synthesize: bool = True
    ) -> Tuple[List[RetrievedContext], Optional[SynthesizedKnowledge]]:
        """Retrieve relevant contexts and optionally synthesize knowledge."""
        
        # Hybrid search
        contexts = self.hybrid_searcher.hybrid_search(query, filters=filters)
        logger.info(f"Hybrid search returned {len(contexts)} contexts")
        
        # Re-rank if available
        if self.reranker and contexts:
            contexts = self.reranker.rerank_contexts(query, contexts, self.top_k_rerank)
            logger.info(f"Re-ranking returned {len(contexts)} contexts")
        
        # Synthesize if available and requested
        synthesis = None
        if synthesize and self.deep_analyzer and contexts:
            synthesis = self.deep_analyzer.synthesize_knowledge({"query": query}, contexts)
            if synthesis:
                logger.info("Knowledge synthesis completed")
            else:
                logger.warning("Knowledge synthesis failed")
        
        return contexts, synthesis
    
    def search_only(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        use_reranking: bool = True
    ) -> List[RetrievedContext]:
        """Perform search without synthesis."""
        
        # Hybrid search
        contexts = self.hybrid_searcher.hybrid_search(query, filters=filters)
        
        # Re-rank if available and requested
        if use_reranking and self.reranker and contexts:
            contexts = self.reranker.rerank_contexts(query, contexts, self.top_k_rerank)
        
        return contexts
    
    def dense_search_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """Perform dense vector search only."""
        return self.hybrid_searcher.dense_search_only(query, top_k, filters)
    
    def sparse_search_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """Perform sparse keyword search only."""
        return self.hybrid_searcher.sparse_search_only(query, top_k, filters)
    
    def synthesize_from_contexts(
        self,
        query: str,
        contexts: List[RetrievedContext]
    ) -> Optional[SynthesizedKnowledge]:
        """Synthesize knowledge from provided contexts."""
        if not self.deep_analyzer:
            logger.warning("Deep analyzer not available for synthesis")
            return None
        
        return self.deep_analyzer.synthesize_knowledge({"query": query}, contexts)
    
    def batch_retrieve_and_analyze(
        self,
        queries: List[str],
        filters: Optional[List[Dict[str, Any]]] = None,
        synthesize: bool = True
    ) -> List[Tuple[List[RetrievedContext], Optional[SynthesizedKnowledge]]]:
        """Process multiple queries in batch."""
        results = []
        
        for i, query in enumerate(queries):
            query_filters = filters[i] if filters and i < len(filters) else None
            result = self.retrieve_and_analyze(query, query_filters, synthesize)
            results.append(result)
        
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the RAG engine configuration."""
        return {
            "hybrid_searcher": self.hybrid_searcher.get_search_info(),
            "reranker": self.reranker.get_reranker_info() if self.reranker else None,
            "deep_analyzer": self.deep_analyzer.get_analyzer_info() if self.deep_analyzer else None,
            "top_k_rerank": self.top_k_rerank
        } 