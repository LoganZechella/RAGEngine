"""
Multi-Collection Vector Database Manager for RAG Engine.
Enhanced VectorDBManager supporting multiple collections for document organization.
"""

import os
import time
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger
import qdrant_client
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.src.models.data_models import EmbeddedChunk, RetrievedContext, DocumentCollection


class MultiCollectionVectorDBManager:
    """
    Enhanced VectorDBManager supporting multiple collections.
    Manages storage and retrieval across organized document collections.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_collection: DocumentCollection = DocumentCollection.CURRENT,
        vector_dimensions: int = 1536,
        distance_metric: str = "cosine"
    ):
        """Initialize with multi-collection support."""
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.default_collection = default_collection
        self.vector_dimensions = vector_dimensions
        
        # Convert string distance_metric to qmodels.Distance enum
        try:
            self.distance_metric_enum = qmodels.Distance[distance_metric.upper()]
        except KeyError:
            logger.error(f"Invalid distance metric '{distance_metric}'. Defaulting to COSINE.")
            self.distance_metric_enum = qmodels.Distance.COSINE
        
        # For local Qdrant instances, clear the API key if it looks like a cloud key
        if self.api_key and ("|" in self.api_key or len(self.api_key) > 50):
            if "localhost" in self.url or "127.0.0.1" in self.url:
                logger.info("Detected local Qdrant URL with cloud API key - removing API key for local connection")
                self.api_key = None
        
        # Initialize client
        self._initialize_client()
        
        # Ensure all collections exist
        self._ensure_all_collections_exist()
    
    def _initialize_client(self):
        """Initialize Qdrant client with error handling."""
        logger.info(f"Initializing QdrantClient with URL: '{self.url}'")
        
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(self.url)
            host = parsed.hostname or self.url.replace("http://", "").replace("https://", "")
            port = parsed.port or 6333
            use_https = parsed.scheme == 'https'

            self.client = qdrant_client.QdrantClient(
                host=host,
                port=port,
                api_key=self.api_key,
                timeout=30.0,
                https=use_https,
                prefer_grpc=False
            )
            
            # Test connection
            logger.info("Testing Qdrant connection...")
            collections = self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} existing collections.")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
        
    def _ensure_all_collections_exist(self) -> None:
        """Ensure all document collections exist."""
        for collection in DocumentCollection:
            self._ensure_collection_exists(collection.value)
            
    def _ensure_collection_exists(self, collection_name: str) -> None:
        """Ensure a specific collection exists."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if collection_name not in collection_names:
                logger.info(f"Creating collection '{collection_name}' with {self.vector_dimensions} dimensions")
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_dimensions,
                        distance=self.distance_metric_enum
                    )
                )
                
                self._create_payload_indexes(collection_name)
                logger.info(f"Collection '{collection_name}' created successfully")
            else:
                logger.info(f"Collection '{collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection '{collection_name}' exists: {str(e)}")
            raise
    
    def _create_payload_indexes(self, collection_name: str) -> None:
        """Create payload indexes for efficient filtering."""
        try:
            index_fields = [
                ("document_id", "keyword"),
                ("chunk_type", "keyword"),
                ("collection", "keyword"),
                ("category", "keyword"),
                ("tags", "keyword")
            ]
            
            for field_name, field_type in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type
                    )
                    logger.debug(f"Created index for field '{field_name}' in collection '{collection_name}'")
                except Exception as e:
                    logger.debug(f"Index creation for '{field_name}' in '{collection_name}' returned: {str(e)}")
                
        except Exception as e:
            logger.warning(f"Error creating payload indexes for '{collection_name}': {str(e)}")
    
    def upsert_embeddings(
        self, 
        embedded_chunks: List[EmbeddedChunk], 
        target_collection: Optional[DocumentCollection] = None
    ) -> Dict[DocumentCollection, int]:
        """
        Insert embeddings into appropriate collections.
        
        Returns:
            Dictionary mapping collections to number of chunks upserted
        """
        if not embedded_chunks:
            return {}
            
        # Group chunks by collection
        chunks_by_collection = {}
        for chunk in embedded_chunks:
            collection = target_collection or chunk.collection or self.default_collection
            if collection not in chunks_by_collection:
                chunks_by_collection[collection] = []
            chunks_by_collection[collection].append(chunk)
        
        results = {}
        
        # Process each collection
        for collection, chunks in chunks_by_collection.items():
            try:
                count = self._upsert_to_collection(chunks, collection.value)
                results[collection] = count
                logger.info(f"Upserted {count} chunks to collection '{collection.value}'")
            except Exception as e:
                logger.error(f"Failed to upsert to collection '{collection.value}': {e}")
                results[collection] = 0
                
        return results
    
    def _upsert_to_collection(self, chunks: List[EmbeddedChunk], collection_name: str) -> int:
        """Upsert chunks to a specific collection."""
        valid_chunks = [chunk for chunk in chunks if chunk.embedding_vector is not None]
        
        if not valid_chunks:
            return 0
            
        points = []
        for chunk in valid_chunks:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            
            payload = {
                "document_id": chunk.document_id,
                "text": chunk.text,
                "chunk_type": str(chunk.chunk_type),
                "collection": chunk.collection.value if chunk.collection else collection_name,
                **chunk.metadata
            }
            
            # Clean payload
            cleaned_payload = {k: v for k, v in payload.items() if v is not None}
            
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=chunk.embedding_vector,
                payload=cleaned_payload
            ))
        
        # Upsert with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                return len(points)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        
        return 0
    
    def search_collections(
        self,
        query_embedding: List[float],
        collections: List[DocumentCollection],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[DocumentCollection, List[RetrievedContext]]:
        """
        Search across multiple collections.
        
        Returns:
            Dictionary mapping collections to search results
        """
        results = {}
        
        for collection in collections:
            try:
                collection_results = self.dense_vector_search(
                    query_embedding=query_embedding,
                    collection_name=collection.value,
                    top_k=top_k,
                    filters=filters
                )
                
                # Add collection info to results
                for result in collection_results:
                    result.collection = collection
                    
                results[collection] = collection_results
                
            except Exception as e:
                logger.error(f"Search failed for collection '{collection.value}': {e}")
                results[collection] = []
        
        return results
    
    def dense_vector_search(
        self,
        query_embedding: List[float],
        collection_name: Optional[str] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """Search a specific collection."""
        collection_name = collection_name or self.default_collection.value
        
        try:
            # Prepare filter
            filter_obj = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        should_conditions = [
                            qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=v))
                            for v in value
                        ]
                        filter_conditions.append(qmodels.Filter(should=should_conditions))
                    else:
                        filter_conditions.append(
                            qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
                        )
                filter_obj = qmodels.Filter(must=filter_conditions)
            
            # Execute search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                query_filter=filter_obj
            )
            
            # Convert to RetrievedContext objects
            results = []
            for hit in search_result:
                payload = hit.payload or {}
                
                context = RetrievedContext(
                    chunk_id=str(hit.id),
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    initial_score=hit.score,
                    collection=DocumentCollection(payload.get("collection", collection_name)),
                    metadata={k: v for k, v in payload.items() 
                             if k not in ["text", "document_id", "collection"]}
                )
                results.append(context)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in dense vector search for '{collection_name}': {str(e)}")
            return []
    
    def get_all_collections_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all collections."""
        info = {}
        
        for collection in DocumentCollection:
            try:
                collection_info = self.client.get_collection(collection_name=collection.value)
                info[collection.value] = {
                    "name": collection.value,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": str(collection_info.config.params.vectors.distance),
                    "points_count": collection_info.points_count,
                    "vectors_count": collection_info.vectors_count,
                    "status": str(collection_info.status)
                }
            except Exception as e:
                logger.error(f"Error getting info for collection '{collection.value}': {e}")
                info[collection.value] = {"error": str(e)}
        
        return info
    
    def delete_collection_data(self, collection: DocumentCollection) -> bool:
        """Delete all data from a specific collection."""
        try:
            return self._clear_collection(collection.value)
        except Exception as e:
            logger.error(f"Error deleting collection '{collection.value}': {e}")
            return False
    
    def _clear_collection(self, collection_name: str) -> bool:
        """Clear all points from a collection."""
        try:
            # Delete all points in the collection
            self.client.delete(
                collection_name=collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id",
                                match=qmodels.MatchExcept(except_="__non_existent_document__")
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"Successfully cleared collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection '{collection_name}': {str(e)}")
            return False 