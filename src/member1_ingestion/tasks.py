import logging
import uuid
import hashlib
from pathlib import Path
from celery import Celery
from src.config import settings
from src.member1_ingestion.parser import DoclingLayoutParser
from src.member1_ingestion.schemas import CandidateProfile
from src.member2_retrieval.postgres_client import PostgresStateClient
from src.member2_retrieval.embedder import BGEM3Embedder
from src.member2_retrieval.qdrant_client import QdrantVectorClient

try:
    import instructor
    from openai import OpenAI

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery app routing
celery_app = Celery(
    "ingestion_tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL
)


@celery_app.task(name="tasks.ingest_resume_pipeline")
def ingest_resume_pipeline(file_path_str: str) -> dict:
    """Background Celery task mapping unstructured files through parsing and schema validations.

    Args:
        file_path_str: Path string to the resume.

    Returns:
        Dict representing validated CandidateProfile data.
    """
    if not INSTRUCTOR_AVAILABLE:
        raise ImportError(
            "The 'instructor' and 'openai' libraries are required to run the ingest_resume_pipeline task. Please install them using pip."
        )

    logger.info(f"Triggered Celery task to parse resume: {file_path_str}")
    path = Path(file_path_str)

    # 1. Parse layout via Docling (OCR enabled)
    parser = DoclingLayoutParser()
    raw_payload = parser.parse_document(path)
    raw_text = raw_payload["raw_text"]

    # 2. Extract structured entities (Instructor LLM)
    logger.info("Initializing Instructor client for profile extraction...")

    # Check for valid API keys
    api_key = settings.VLLM_API_KEY or settings.OPENAI_API_KEY
    if not api_key or api_key == "mock-key-for-development":
        raise ValueError(
            "A valid OPENAI_API_KEY or VLLM_API_KEY must be set in the environment for LLM extraction."
        )

    try:
        if settings.VLLM_API_URL:
            openai_client = OpenAI(
                base_url=settings.VLLM_API_URL, api_key=settings.VLLM_API_KEY
            )
            model_name = "local-model"
        else:
            openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            model_name = "gpt-4o-mini"

        instructor_client = instructor.from_openai(openai_client)

        # Enforce exact Instructor schema matching
        profile = instructor_client.chat.completions.create(
            model=model_name,
            response_model=CandidateProfile,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional recruiting assistant. Extract the structured candidate profile from the raw resume text. Extract skills, duration details, companies, and work/education timelines accurately.",
                },
                {"role": "user", "content": f"Resume Text:\n\n{raw_text}"},
            ],
            temperature=0.0,
        )
        logger.info(f"LLM successfully extracted profile for: {profile.name}")
    except Exception as e:
        logger.critical(f"Instructor LLM extraction failed: {e}")
        raise RuntimeError(f"Instructor extraction failed: {e}") from e

    # Generate unique candidate ID based on email
    email_key = profile.email or str(uuid.uuid4())
    candidate_id = (
        "cand_"
        + hashlib.md5(email_key.lower().strip().encode("utf-8")).hexdigest()[:12]
    )

    profile_dict = profile.model_dump()
    profile_dict["id"] = candidate_id

    # 3. Store in PostgreSQL database
    pg_client = PostgresStateClient()
    pg_client.store_candidate(candidate_id, profile_dict, raw_text)

    # 4. Generate Embeddings (BGE-M3 Dense + Sparse)
    embedder = BGEM3Embedder()
    skills_summary = ", ".join(profile.skills)
    exp_summary = " ".join(
        [
            f"{exp.role} at {exp.company} - {exp.description or ''}"
            for exp in profile.experience
        ]
    )
    summary_text = (
        f"Name: {profile.name}. Skills: {skills_summary}. Experience: {exp_summary}"
    )

    embeddings_list = embedder.generate_embeddings([summary_text])
    if embeddings_list:
        dense_vector = embeddings_list[0]["dense"]
        sparse_vector = embeddings_list[0]["sparse"]

        # 5. Index in Qdrant Vector database
        qdrant_client = QdrantVectorClient()
        qdrant_client.upsert_vectors(
            [
                {
                    "id": candidate_id,
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                    "payload": {
                        "name": profile.name,
                        "email": profile.email,
                        "skills": profile.skills,
                        "years_exp": sum(
                            exp.duration_months for exp in profile.experience
                        )
                        / 12.0,
                    },
                }
            ]
        )

    return profile_dict
