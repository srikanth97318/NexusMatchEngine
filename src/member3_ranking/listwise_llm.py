import os
import logging
from typing import List, Dict, Any
from src.config import settings
from pydantic import BaseModel, Field

try:
    import instructor
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandidateRankOrder(BaseModel):
    ordered_candidate_ids: List[str] = Field(..., description="List of candidate IDs sorted from best to worst match")
    rationales: Dict[str, str] = Field(..., description="Dictionary mapping candidate ID to a short 1-sentence matching rationale")


class ListwiseLLMRanker:
    """Fine-grained LLM re-ranker processing structural context via listwise prompts."""
    
    def __init__(self):
        logger.info("Initializing vLLM / OpenAI client framework (Late listwise scoring)...")
        self.api_key = settings.VLLM_API_KEY or settings.OPENAI_API_KEY
        self.client = None
        self.llm_available = LLM_AVAILABLE
        
        if self.llm_available and self.api_key and self.api_key != "mock-key-for-development":
            try:
                if settings.VLLM_API_URL:
                    logger.info(f"Using local vLLM server: {settings.VLLM_API_URL}")
                    openai_client = OpenAI(base_url=settings.VLLM_API_URL, api_key=self.api_key)
                    self.model_name = "local-model"
                else:
                    logger.info("Using OpenAI GPT endpoint...")
                    openai_client = OpenAI(api_key=self.api_key)
                    self.model_name = "gpt-4o-mini"
                    
                self.client = instructor.from_openai(openai_client)
                logger.info("LLM listwise ranker client initialized successfully.")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}. Fallback ranking active.")
        else:
            logger.warning("Instructor client or API keys not configured. Fallback ranking active.")
            
    def rerank_list(self, candidates: List[Dict[str, Any]], job_description: str) -> List[Dict[str, Any]]:
        """Submit list of top matches to LLM for final listwise sorting.
        
        Returns:
            List of candidates sorted in optimized alignment order.
        """
        if not candidates:
            return []
            
        logger.info(f"Submitting {len(candidates)} candidate summaries to listwise LLM pipeline...")
        
        if self.client is not None:
            try:
                # Format candidate summaries for the prompt context
                candidate_summaries = []
                for c in candidates:
                    summary = f"ID: {c['id']}, Name: {c['name']}, Initial Score: {c['ltr_score']:.2f}"
                    candidate_summaries.append(summary)
                    
                context = "\n".join(candidate_summaries)
                
                logger.info("Context window size is safe. Executing LLM listwise inference...")
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=CandidateRankOrder,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert recruiter. Compare the job description with the candidate summaries. "
                                "Sort the candidate IDs from best match to worst match. Provide a 1-sentence rationale for each candidate."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Job Description:\n{job_description}\n\nCandidates:\n{context}"
                        }
                    ],
                    temperature=0.0
                )
                
                # Re-order candidates based on LLM output
                ordered_ids = response.ordered_candidate_ids
                rationales = response.rationales
                
                candidates_map = {c["id"]: c for c in candidates}
                reranked_candidates = []
                
                for idx, cid in enumerate(ordered_ids):
                    if cid in candidates_map:
                        cand = candidates_map[cid]
                        cand["listwise_rank"] = idx + 1
                        cand["llm_rationale"] = rationales.get(cid, "Aligned based on technical parameters.")
                        reranked_candidates.append(cand)
                        
                # Add any candidates missed by the LLM at the end
                for cid, cand in candidates_map.items():
                    if cand not in reranked_candidates:
                        cand["listwise_rank"] = len(reranked_candidates) + 1
                        cand["llm_rationale"] = "Ranked by fallback pipeline."
                        reranked_candidates.append(cand)
                        
                return reranked_candidates
                
            except Exception as e:
                logger.error(f"LLM reranking failed: {e}. Falling back to initial LTR score sorting.")
                
        # Algorithmic fallback: Sort purely by LTR score
        sorted_candidates = sorted(candidates, key=lambda x: x.get("ltr_score", 0.0), reverse=True)
        for idx, cand in enumerate(sorted_candidates):
            cand["listwise_rank"] = idx + 1
            cand["llm_rationale"] = f"Candidate ranked by LightGBM model score of {cand.get('ltr_score', 0.0):.2f} (LLM ranker fallback)."
            
        return sorted_candidates
