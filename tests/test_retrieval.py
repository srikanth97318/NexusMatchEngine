from src.member2_retrieval.embedder import BGEM3Embedder
from src.member2_retrieval.qdrant_client import QdrantVectorClient


def test_bge_m3_embedder_dimensions():
    embedder = BGEM3Embedder()
    results = embedder.generate_embeddings(["Python Developer Resume"])
    
    assert len(results) == 1
    assert "dense" in results[0]
    assert len(results[0]["dense"]) == 1024
    assert "sparse" in results[0]


def test_qdrant_hybrid_search():
    client = QdrantVectorClient()
    mock_dense = [0.1] * 1024
    mock_sparse = {"indices": [101], "values": [0.5]}
    
    results = client.hybrid_search(mock_dense, mock_sparse, limit=5)
    assert len(results) > 0
    assert "score" in results[0]
    assert "payload" in results[0]
