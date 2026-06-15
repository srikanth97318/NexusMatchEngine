from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.member4_orchestration.main import app

client = TestClient(app)


def test_gateway_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


@patch('src.member4_orchestration.main.get_llm_ranker')
@patch('src.member4_orchestration.main.get_postgres_client')
@patch('src.member4_orchestration.main.get_qdrant_client')
@patch('src.member4_orchestration.main.get_embedder')
def test_end_to_end_match_endpoint(mock_get_embedder, mock_get_qdrant, mock_get_pg, mock_get_llm):
    # Setup mock behaviors
    mock_embedder_inst = MagicMock()
    mock_embedder_inst.generate_embeddings.return_value = [{"dense": [0.1] * 1024, "sparse": {"indices": [101], "values": [0.5]}}]
    mock_get_embedder.return_value = mock_embedder_inst
    
    mock_qdrant_inst = MagicMock()
    mock_qdrant_inst.hybrid_search.return_value = [
        {"id": "cand_001", "score": 0.892, "payload": {"name": "Jane Doe", "skills": ["Python", "FastAPI"]}}
    ]
    mock_get_qdrant.return_value = mock_qdrant_inst
    
    mock_pg_inst = MagicMock()
    mock_pg_inst.get_candidate.return_value = {
        "id": "cand_001",
        "name": "Jane Doe",
        "skills": ["Python", "FastAPI"],
        "experience": [{"company": "Google", "role": "Engineer", "duration_months": 24}]
    }
    mock_get_pg.return_value = mock_pg_inst
    
    mock_llm_inst = MagicMock()
    mock_llm_inst.rerank_list.return_value = [
        {
            "id": "cand_001",
            "name": "Jane Doe",
            "ltr_score": 0.82,
            "listwise_rank": 1,
            "llm_rationale": "Perfect fit."
        }
    ]
    mock_get_llm.return_value = mock_llm_inst

    query_payload = {
        "job_description_id": "job_101",
        "job_text": "Need a senior backend engineer skilled in Python and FastAPI",
        "required_skills": ["Python", "FastAPI"],
        "min_experience_years": 5,
        "top_k": 3
    }
    response = client.post("/match", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["job_id"] == "job_101"
    assert "results" in data
    assert len(data["results"]) > 0
    assert "ltr_score" in data["results"][0]
