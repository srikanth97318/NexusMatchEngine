import logging
from celery import Celery
from pathlib import Path
from src.config import settings
from src.member1_ingestion.parser import DoclingLayoutParser
from src.member1_ingestion.schemas import CandidateProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery app routing
celery_app = Celery("ingestion_tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL)


@celery_app.task(name="tasks.ingest_resume_pipeline")
def ingest_resume_pipeline(file_path_str: str) -> dict:
    """Background Celery task mapping unstructured files through parsing and schema validations.
    
    Args:
        file_path_str: Path string to the resume.
        
    Returns:
        Dict representing validated CandidateProfile data.
    """
    logger.info(f"Triggered Celery task to parse resume: {file_path_str}")
    path = Path(file_path_str)
    
    # 1. Parse layout via Docling
    parser = DoclingLayoutParser()
    raw_payload = parser.parse_document(path)
    
    # 2. Mock entity schema mapping (normally done with LLMs + Instructor)
    # This yields a type-safe validated CandidateProfile model
    mock_profile = CandidateProfile(
        name="Jane Doe",
        email="jane.doe@example.com",
        skills=["Python", "FastAPI", "Qdrant", "Machine Learning"],
        experience=[
            {
                "company": "Tech Innovations",
                "role": "Senior Engineer",
                "duration_months": 36,
                "description": "Led backend architecture migrations."
            }
        ],
        education=[
            {
                "institution": "State University",
                "degree": "B.S.",
                "field_of_study": "Computer Science"
            }
        ]
    )
    
    logger.info(f"Successfully validated profile for: {mock_profile.name}")
    return mock_profile.model_dump()
