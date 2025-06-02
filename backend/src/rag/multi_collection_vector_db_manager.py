"""
Multi-Collection Vector Database Manager for RAGEngine.
Extends the base VectorDBManager to support multiple Qdrant collections.
"""

import asyncio
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ..models.data_models import (
    EmbeddedChunk, RetrievedContext, DocumentCollection, CollectionMetadata
)
from .collection_manager import CollectionManager

logger = logging.getLogger(__name__)


class MultiCollectionVectorDBManager:
    """Enhanced Vector Database Manager with multi-collection support."""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", vector_size: int = 1536):
        self.client = QdrantClient(url=qdrant_url)
        self.vector_size = vector_size
        self.collection_manager = CollectionManager()
        self._collection_cache = {}
        
        # Initialize all collections
        asyncio.create_task(self._initialize_collections())
    
    async def _initialize_collections(self):
        """Initialize all document collections in Qdrant."""
        for collection in DocumentCollection:
            await self._ensure_collection_exists(collection)
    
    async def _ensure_collection_exists(self, collection: DocumentCollection) -> bool:
        """Ensure a collection exists in Qdrant, create if it doesn't."""
        collection_name = collection.value
        
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name not in existing_names:
                logger.info(f"Creating collection: {collection_name}")
                
                # Create collection with optimized settings
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                        on_disk=True  # Store vectors on disk for better memory usage
                    ),
                    optimizers_config=models.OptimizersConfig(
                        default_segment_number=2,
                        max_segment_size=20000,
                        memmap_threshold=20000,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=2
                    ),
                    hnsw_config=models.HnswConfig(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000,
                        max_indexing_threads=2,
                        on_disk=True
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=False
                        )
                    )
                )
                
                # Create payload index for faster filtering
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="chunk_type",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                
                logger.info(f"Successfully created collection: {collection_name}")
                return True
            else:
                logger.debug(f"Collection already exists: {collection_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name} exists: {e}")
            return False
    
    async def upsert_chunks(self, chunks: List[EmbeddedChunk], collection: Optional[DocumentCollection] = None) -> bool:
        """
        Upsert chunks into their respective collections.
        
        Args:
            chunks: List of embedded chunks to upsert
            collection: Optional specific collection (overrides chunk collection)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Group chunks by collection
            collection_groups = {}
            
            for chunk in chunks:
                target_collection = collection or chunk.collection
                if target_collection not in collection_groups:
                    collection_groups[target_collection] = []
                collection_groups[target_collection].append(chunk)
            
            # Upsert to each collection
            for target_collection, collection_chunks in collection_groups.items():
                await self._ensure_collection_exists(target_collection)
                
                points = []
                for chunk in collection_chunks:
                    if chunk.embedding_vector is None:
                        logger.warning(f"Skipping chunk {chunk.chunk_id} - no embedding vector")
                        continue
                    
                    payload = {
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "chunk_type": chunk.chunk_type.value,
                        "collection": target_collection.value,
                        "metadata": chunk.metadata,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    points.append(models.PointStruct(
                        id=chunk.chunk_id,
                        vector=chunk.embedding_vector,
                        payload=payload
                    ))
                
                if points:
                    self.client.upsert(
                        collection_name=target_collection.value,
                        points=points
                    )
                    logger.info(f"Upserted {len(points)} chunks to collection {target_collection.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error upserting chunks: {e}")
            return False
    
    async def search_collections(
        self,
        query_vector: List[float],
        collections: Optional[List[DocumentCollection]] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedContext]:
        """
        Search across multiple collections.
        
        Args:
            query_vector: The query embedding vector
            collections: List of collections to search (None = all collections)
            limit: Maximum number of results per collection
            score_threshold: Minimum similarity score
            filter_conditions: Additional filter conditions
            
        Returns:
            List of retrieved contexts sorted by score
        """
        if collections is None:
            collections = list(DocumentCollection)
        
        all_results = []
        
        for collection in collections:
            try:
                await self._ensure_collection_exists(collection)
                
                # Build filter
                search_filter = None
                if filter_conditions:
                    conditions = []
                    for key, value in filter_conditions.items():
                        conditions.append(models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        ))
                    
                    if conditions:
                        search_filter = models.Filter(must=conditions)
                
                # Search the collection
                search_results = self.client.search(
                    collection_name=collection.value,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=search_filter,
                    with_payload=True
                )
                
                # Convert to RetrievedContext objects
                for result in search_results:
                    context = RetrievedContext(
                        chunk_id=str(result.id),
                        document_id=result.payload.get("document_id", ""),
                        text=result.payload.get("text", ""),
                        initial_score=result.score,
                        collection=collection,
                        metadata=result.payload.get("metadata", {})
                    )
                    all_results.append(context)
                    
            except Exception as e:
                logger.error(f"Error searching collection {collection.value}: {e}")
                continue
        
        # Sort all results by score (descending)
        all_results.sort(key=lambda x: x.initial_score, reverse=True)
        
        # Return top results across all collections
        return all_results[:limit * len(collections)]
    
    async def get_collection_stats(self, collection: DocumentCollection) -> Dict[str, Any]:
        """Get statistics for a specific collection."""
        try:
            await self._ensure_collection_exists(collection)
            
            collection_info = self.client.get_collection(collection.value)
            
            # Get document count (unique document_ids)
            scroll_result = self.client.scroll(
                collection_name=collection.value,
                limit=10000,  # Adjust based on expected collection size
                with_payload=["document_id"]
            )
            
            unique_documents = set()
            for point in scroll_result[0]:
                if point.payload and "document_id" in point.payload:
                    unique_documents.add(point.payload["document_id"])
            
            metadata = self.collection_manager.get_collection_metadata(collection)
            
            return {
                "collection_name": collection.value,
                "display_name": metadata.display_name,
                "description": metadata.description,
                "document_count": len(unique_documents),
                "chunk_count": collection_info.points_count,
                "vector_count": collection_info.vectors_count,
                "status": "green" if collection_info.status == models.CollectionStatus.GREEN else "yellow",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for collection {collection.value}: {e}")
            return {
                "collection_name": collection.value,
                "display_name": collection.value,
                "description": "Error retrieving collection information",
                "document_count": 0,
                "chunk_count": 0,
                "vector_count": 0,
                "status": "error",
                "error": str(e)
            }
    
    async def get_all_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections."""
        stats = {}
        
        for collection in DocumentCollection:
            collection_stats = await self.get_collection_stats(collection)
            stats[collection.value] = collection_stats
        
        return stats
    
    async def clear_collection(self, collection: DocumentCollection) -> bool:
        """Clear all documents from a collection."""
        try:
            await self._ensure_collection_exists(collection)
            
            # Delete the collection and recreate it
            self.client.delete_collection(collection.value)
            await self._ensure_collection_exists(collection)
            
            logger.info(f"Cleared collection: {collection.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection {collection.value}: {e}")
            return False
    
    async def delete_document(self, document_id: str, collection: Optional[DocumentCollection] = None) -> bool:
        """Delete all chunks for a specific document."""
        try:
            collections_to_search = [collection] if collection else list(DocumentCollection)
            
            for coll in collections_to_search:
                await self._ensure_collection_exists(coll)
                
                # Find all points with this document_id
                scroll_result = self.client.scroll(
                    collection_name=coll.value,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )]
                    ),
                    limit=10000,
                    with_payload=False
                )
                
                # Delete the points
                point_ids = [point.id for point in scroll_result[0]]
                if point_ids:
                    self.client.delete(
                        collection_name=coll.value,
                        points_selector=models.PointIdsList(points=point_ids)
                    )
                    logger.info(f"Deleted {len(point_ids)} chunks for document {document_id} from {coll.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def get_document_collections(self, document_id: str) -> List[DocumentCollection]:
        """Get all collections that contain chunks from a specific document."""
        found_collections = []
        
        for collection in DocumentCollection:
            try:
                await self._ensure_collection_exists(collection)
                
                # Check if document exists in this collection
                scroll_result = self.client.scroll(
                    collection_name=collection.value,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )]
                    ),
                    limit=1,
                    with_payload=False
                )
                
                if scroll_result[0]:  # If any points found
                    found_collections.append(collection)
                    
            except Exception as e:
                logger.error(f"Error checking document {document_id} in collection {collection.value}: {e}")
                continue
        
        return found_collections
    
    async def move_document(
        self,
        document_id: str,
        source_collection: DocumentCollection,
        target_collection: DocumentCollection
    ) -> bool:
        """Move a document from one collection to another."""
        try:
            # Get all chunks from source collection
            scroll_result = self.client.scroll(
                collection_name=source_collection.value,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=True
            )
            
            chunks_to_move = scroll_result[0]
            if not chunks_to_move:
                logger.warning(f"No chunks found for document {document_id} in {source_collection.value}")
                return False
            
            # Convert to EmbeddedChunk objects and update collection
            embedded_chunks = []
            for point in chunks_to_move:
                chunk = EmbeddedChunk(
                    chunk_id=str(point.id),
                    document_id=point.payload["document_id"],
                    text=point.payload["text"],
                    embedding_vector=point.vector,
                    chunk_type=point.payload.get("chunk_type", "text"),
                    collection=target_collection,
                    metadata=point.payload.get("metadata", {})
                )
                embedded_chunks.append(chunk)
            
            # Upsert to target collection
            await self.upsert_chunks(embedded_chunks, target_collection)
            
            # Delete from source collection
            await self.delete_document(document_id, source_collection)
            
            logger.info(f"Moved document {document_id} from {source_collection.value} to {target_collection.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving document {document_id}: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the vector database."""
        try:
            # Check Qdrant connection
            collections = self.client.get_collections()
            
            collection_status = {}
            for collection in DocumentCollection:
                try:
                    info = self.client.get_collection(collection.value)
                    collection_status[collection.value] = {
                        "status": info.status.value if info.status else "unknown",
                        "points_count": info.points_count,
                        "vectors_count": info.vectors_count
                    }
                except:
                    collection_status[collection.value] = {
                        "status": "not_found",
                        "points_count": 0,
                        "vectors_count": 0
                    }
            
            return {
                "status": "healthy",
                "total_collections": len(collections.collections),
                "expected_collections": len(DocumentCollection),
                "collections": collection_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }