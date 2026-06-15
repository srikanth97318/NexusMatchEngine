from fastapi.testclient import TestClient
from src.member4_orchestration.main import app

client = TestClient(app)


def test_gateway_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_end_to_end_match_endpoint():
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
