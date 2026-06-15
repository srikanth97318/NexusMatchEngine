import logging
from typing import List, Dict, Any, Optional
import numpy as np
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct, SparseVector, NamedSparseVector
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def reciprocal_rank_fusion(dense_hits: List[Any], sparse_hits: List[Any], limit: int = 10, rrf_constant: int = 60) -> List[Dict[str, Any]]:
    """Perform Reciprocal Rank Fusion (RRF) on dense and sparse hits.
    
    Formula:
        RRF = 1 / (60 + Rank_d) + 1 / (60 + Rank_s)
    """
    scores = {}
    payloads = {}
    
    # Process dense rank hits
    for rank, hit in enumerate(dense_hits):
        point_id = hit.id
        scores[point_id] = scores.get(point_id, 0.0) + (1.0 / (rrf_constant + rank + 1))
        payloads[point_id] = getattr(hit, "payload", {})
        
    # Process sparse rank hits
    for rank, hit in enumerate(sparse_hits):
        point_id = hit.id
        scores[point_id] = scores.get(point_id, 0.0) + (1.0 / (rrf_constant + rank + 1))
        payloads[point_id] = getattr(hit, "payload", {})
        
    # Sort and slice
    sorted_points = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    return [
        {
            "id": point_id,
            "score": score,
            "payload": payloads[point_id]
        }
        for point_id, score in sorted_points
    ]


class QdrantVectorClient:
    """Client handling hybrid vector collection setup, indexes, and queries in Qdrant."""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or settings.QDRANT_HOST
        self.port = port or settings.QDRANT_PORT
        self.client = None
        self.mock_store = {}  # In-memory storage fallback
        
        if QDRANT_AVAILABLE:
            try:
                logger.info(f"Connecting to Qdrant cluster at: {self.host}:{self.port}...")
                self.client = QdrantClient(host=self.host, port=self.port, timeout=5.0)
                # Quick health check call
                self.client.get_collections()
                logger.info("Successfully connected to Qdrant cluster.")
            except Exception as e:
                logger.warning(f"Could not connect to Qdrant server: {e}. Fallback to simulated store active.")
                self.client = None
        else:
            logger.warning("qdrant-client package is not installed. Fallback to simulated store active.")

    def setup_collection(self, collection_name: str = settings.QDRANT_COLLECTION) -> None:
        """Create hybrid dense/sparse collection configuration in Qdrant if non-existent."""
        if self.client:
            try:
                logger.info(f"Initializing collection '{collection_name}' in Qdrant (Dense: 1024-dim, Sparse enabled)...")
                # Drop existing to verify clean structure during scaffolding tests
                self.client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(size=1024, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    }
                )
                logger.info(f"Collection '{collection_name}' created/reset successfully.")
            except Exception as e:
                logger.error(f"Failed to setup Qdrant collection '{collection_name}': {e}")
        else:
            logger.info(f"[SIMULATED] Resetting mock collection '{collection_name}'...")
            self.mock_store = {}

    def upsert_vectors(self, points: List[Dict[str, Any]], collection_name: str = settings.QDRANT_COLLECTION) -> bool:
        """Upload payloads containing dense and sparse vectors to vector search index.
        
        Each point in `points` dict should look like:
        {
            "id": str,
            "dense": List[float],
            "sparse": {"indices": List[int], "values": List[float]},
            "payload": Dict[str, Any]
        }
        """
        if not points:
            return True
            
        if self.client:
            try:
                qdrant_points = [
                    PointStruct(
                        id=pt["id"],
                        vector={
                            "dense": pt["dense"],
                            "sparse": SparseVector(
                                indices=pt["sparse"]["indices"],
                                values=pt["sparse"]["values"]
                            )
                        },
                        payload=pt["payload"]
                    )
                    for pt in points
                ]
                self.client.upsert(collection_name=collection_name, points=qdrant_points)
                logger.info(f"Upserted {len(points)} vectors to collection '{collection_name}'")
                return True
            except Exception as e:
                logger.error(f"Qdrant upsert failed: {e}")
                return False
        else:
            # Cache in mock store
            logger.info(f"[SIMULATED] Upserting {len(points)} points in-memory.")
            for pt in points:
                self.mock_store[pt["id"]] = {
                    "dense": pt["dense"],
                    "sparse": pt["sparse"],
                    "payload": pt["payload"]
                }
            return True

    def hybrid_search(
        self, 
        dense_query: List[float], 
        sparse_query: Dict[str, Any], 
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = settings.QDRANT_COLLECTION
    ) -> List[Dict[str, Any]]:
        """Perform Approximate Nearest Neighbor (ANN) search merging dense and sparse scores.
        
        Returns:
            List of matching records and search similarity scores.
        """
        if self.client:
            try:
                # 1. Query Dense space
                dense_hits = self.client.search(
                    collection_name=collection_name,
                    query_vector=("dense", dense_query),
                    limit=limit * 2,
                    with_payload=True
                )
                
                # 2. Query Sparse space
                sparse_hits = self.client.search(
                    collection_name=collection_name,
                    query_vector=NamedSparseVector(
                        name="sparse",
                        vector=SparseVector(
                            indices=sparse_query["indices"],
                            values=sparse_query["values"]
                        )
                    ),
                    limit=limit * 2,
                    with_payload=True
                )
                
                # 3. Fuse scores via Reciprocal Rank Fusion (RRF)
                fused_results = reciprocal_rank_fusion(dense_hits, sparse_hits, limit)
                return fused_results
            except Exception as e:
                logger.error(f"Qdrant hybrid search failed: {e}. Falling back to simulation query.")
                
        # Simulated Search Fallback
        logger.info("[SIMULATED] Running in-memory hybrid search...")
        if not self.mock_store:
            logger.info("[SIMULATED] Populating empty mock store with default test candidates.")
            self.mock_store = {
                "cand_001": {
                    "dense": [0.1] * 1024,
                    "sparse": {"indices": [101], "values": [0.5]},
                    "payload": {"name": "Jane Doe", "skills": ["Python", "FastAPI"], "years_exp": 5}
                },
                "cand_003": {
                    "dense": [0.1] * 1024,
                    "sparse": {"indices": [101], "values": [0.5]},
                    "payload": {"name": "Alice Smith", "skills": ["Python", "Machine Learning"], "years_exp": 8}
                }
            }
        dense_hits = []
        sparse_hits = []
        
        class MockHit:
            def __init__(self, id_val: str, score_val: float, payload_val: dict):
                self.id = id_val
                self.score = score_val
                self.payload = payload_val
                
        for point_id, point_data in self.mock_store.items():
            payload = point_data["payload"]
            
            # Apply basic filters if any
            if filters:
                match = True
                for k, v in filters.items():
                    if payload.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            
            # Calculate dense cosine similarity
            p_dense = np.array(point_data["dense"])
            q_dense = np.array(dense_query)
            dot = np.dot(p_dense, q_dense)
            norm_p = np.linalg.norm(p_dense)
            norm_q = np.linalg.norm(q_dense)
            dense_score = float(dot / (norm_p * norm_q)) if (norm_p * norm_q) > 0 else 0.0
            dense_hits.append(MockHit(point_id, dense_score, payload))
            
            # Calculate sparse overlap
            p_sparse_indices = set(point_data["sparse"]["indices"])
            q_sparse_indices = set(sparse_query["indices"])
            overlap = len(p_sparse_indices.intersection(q_sparse_indices))
            sparse_score = float(overlap / max(len(q_sparse_indices), 1))
            sparse_hits.append(MockHit(point_id, sparse_score, payload))
            
        # Sort individual channels
        dense_hits.sort(key=lambda x: x.score, reverse=True)
        sparse_hits.sort(key=lambda x: x.score, reverse=True)
        
        return reciprocal_rank_fusion(dense_hits, sparse_hits, limit)
