import os
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from src.member1_ingestion.tasks import ingest_resume_pipeline
from src.member2_retrieval.qdrant_client import QdrantVectorClient
from src.member2_retrieval.postgres_client import PostgresStateClient
from src.member2_retrieval.embedder import BGEM3Embedder
from src.member3_ranking.feature_engineering import build_ltr_features
from src.member3_ranking.ranker import LightGBMRanker
from src.member3_ranking.listwise_llm import ListwiseLLMRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NexusMatch Engine API", version="1.0.0")

# Lazy loading client cache
_qdrant_client = None
_postgres_client = None
_embedder = None
_ranker = None
_llm_ranker = None


def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantVectorClient()
    return _qdrant_client


def get_postgres_client():
    global _postgres_client
    if _postgres_client is None:
        _postgres_client = PostgresStateClient()
    return _postgres_client


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = BGEM3Embedder()
    return _embedder


def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = LightGBMRanker()
    return _ranker


def get_llm_ranker():
    global _llm_ranker
    if _llm_ranker is None:
        _llm_ranker = ListwiseLLMRanker()
    return _llm_ranker


class MatchQuery(BaseModel):
    job_description_id: str
    job_text: str
    required_skills: List[str]
    min_experience_years: int
    top_k: int = 5


@app.get("/")
def read_root():
    return {"status": "online", "system": "NexusMatch Engine Gateway"}


@app.post("/ingest/resume")
async def ingest_resume(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    """Asynchronous endpoint queuing resume files for layout analysis."""
    logger.info(f"API request to ingest resume: {file.filename}")

    # Save uploaded file bytes to workspace directory so Celery workers can access it
    upload_dir = Path("data/raw/candidate_resumes")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / file.filename

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info(f"Successfully saved uploaded file bytes to: {temp_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    # Queue task via Celery worker
    task = ingest_resume_pipeline.delay(str(temp_path))
    return {"status": "queued", "task_id": task.id, "filename": file.filename}


@app.post("/match")
def match_candidates(query: MatchQuery):
    """Perform end-to-end matching: hybrid retrieval -> LambdaMART ranker -> LLM refinement."""
    logger.info(
        f"Executing end-to-end matching query for Job: {query.job_description_id}..."
    )

    postgres_client = get_postgres_client()
    qdrant_client = get_qdrant_client()
    embedder = get_embedder()
    ranker = get_ranker()
    llm_ranker = get_llm_ranker()

    # 1. Save Job Description to PostgreSQL database for dashboard metrics aggregation
    try:
        jd_data = {
            "title": query.job_description_id.replace("_", " ").title(),
            "department": "Engineering",
            "required_skills": query.required_skills,
            "preferred_skills": [],
            "min_experience_years": query.min_experience_years,
            "full_text": query.job_text,
        }
        postgres_client.store_job_description(query.job_description_id, jd_data)
    except Exception as e:
        logger.error(
            f"Failed to persist job description {query.job_description_id}: {e}"
        )

    # 2. Embed query JD using BGE-M3
    embedded_query = embedder.generate_embeddings([query.job_text])[0]

    # 3. Retrieve top candidates via Qdrant Hybrid search
    retrieved_candidates = qdrant_client.hybrid_search(
        dense_query=embedded_query["dense"],
        sparse_query=embedded_query["sparse"],
        limit=query.top_k * 2,
    )

    # 4. Feature Engineering and LTR Scoring (LightGBM)
    scored_candidates = []
    job_meta = {
        "required_skills": query.required_skills,
        "min_experience_years": query.min_experience_years,
    }

    for cand in retrieved_candidates:
        # Query detailed candidate profile dynamically from PostgreSQL relational layer
        cand_db = postgres_client.get_candidate(cand["id"])
        if cand_db:
            candidate_data = {
                "skills": cand_db.get("skills", []),
                "experience": cand_db.get("experience", []),
            }
            cand_name = cand_db.get("name", cand["payload"].get("name", "Unknown"))
        else:
            candidate_data = {
                "skills": cand["payload"].get("skills", []),
                "experience": [
                    {
                        "role": "Engineer",
                        "duration_months": int(
                            cand["payload"].get("years_exp", 0) * 12
                        ),
                    }
                ],
            }
            cand_name = cand["payload"].get("name", "Unknown")

        features = build_ltr_features(candidate_data, job_meta, cand["score"])
        score = ranker.predict_ranking([features])[0]
        shap = ranker.generate_shap_values(features)

        scored_candidates.append(
            {
                "id": cand["id"],
                "name": cand_name,
                "initial_score": cand["score"],
                "ltr_score": score,
                "shap_values": shap,
            }
        )

    # Sort by LTR score
    scored_candidates = sorted(
        scored_candidates, key=lambda x: x["ltr_score"], reverse=True
    )

    # 5. Final LLM Refinement Loop
    refined = llm_ranker.rerank_list(scored_candidates[: query.top_k], query.job_text)

    return {"job_id": query.job_description_id, "results": refined}


@app.get("/metrics")
def get_metrics():
    """Fetch aggregated recruiter metrics from PostgreSQL database."""
    logger.info("Fetching dashboard metrics from PostgreSQL...")
    postgres_client = get_postgres_client()
    return postgres_client.get_dashboard_metrics()
