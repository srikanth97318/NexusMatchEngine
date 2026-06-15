import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from src.member1_ingestion.tasks import ingest_resume_pipeline
from src.member2_retrieval.qdrant_client import QdrantVectorClient
from src.member2_retrieval.embedder import BGEM3Embedder
from src.member3_ranking.feature_engineering import build_ltr_features
from src.member3_ranking.ranker import LightGBMRanker
from src.member3_ranking.listwise_llm import ListwiseLLMRanker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NexusMatch Engine API", version="1.0.0")

# Scaffolding clients
qdrant_client = QdrantVectorClient()
embedder = BGEM3Embedder()
ranker = LightGBMRanker()
llm_ranker = ListwiseLLMRanker()


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
async def ingest_resume(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Asynchronous endpoint queuing resume files for layout analysis."""
    logger.info(f"API request to ingest resume: {file.filename}")
    
    # Save file temporarily to trigger parsing
    temp_path = f"/tmp/{file.filename}"
    # In production, write file bytes to storage
    
    # Queue task via Celery worker
    task = ingest_resume_pipeline.delay(temp_path)
    return {
        "status": "queued",
        "task_id": task.id,
        "filename": file.filename
    }


@app.post("/match")
def match_candidates(query: MatchQuery):
    """Perform end-to-end matching: hybrid retrieval -> LambdaMART ranker -> LLM refinement."""
    logger.info("Executing end-to-end matching query pipeline...")
    
    # 1. Embed query JD using BGE-M3
    embedded_query = embedder.generate_embeddings([query.job_text])[0]
    
    # 2. Retrieve top candidates via Qdrant Hybrid search
    retrieved_candidates = qdrant_client.hybrid_search(
        dense_query=embedded_query["dense"],
        sparse_query=embedded_query["sparse"],
        limit=query.top_k * 2
    )
    
    # 3. Feature Engineering and LTR Scoring (LightGBM)
    scored_candidates = []
    job_meta = {
        "required_skills": query.required_skills,
        "min_experience_years": query.min_experience_years
    }
    
    for cand in retrieved_candidates:
        candidate_data = {
            "skills": cand["payload"].get("skills", []),
            "experience": [{"role": "Engineer", "duration_months": cand["payload"].get("years_exp", 0) * 12}]
        }
        features = build_ltr_features(candidate_data, job_meta, cand["score"])
        score = ranker.predict_ranking([features])[0]
        shap = ranker.generate_shap_values(features)
        
        scored_candidates.append({
            "id": cand["id"],
            "name": cand["payload"]["name"],
            "initial_score": cand["score"],
            "ltr_score": score,
            "shap_values": shap
        })
        
    # Sort by LTR score
    scored_candidates = sorted(scored_candidates, key=lambda x: x["ltr_score"], reverse=True)
    
    # 4. Final LLM Refinement Loop
    refined = llm_ranker.rerank_list(scored_candidates[:query.top_k], query.job_text)
    
    return {
        "job_id": query.job_description_id,
        "results": refined
    }
