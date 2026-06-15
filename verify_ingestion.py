import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env if present
load_dotenv()

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent))

from src.member1_ingestion.parser import DoclingLayoutParser
from src.member1_ingestion.schemas import CandidateProfile

def run_verification():
    print("==================================================")
    print("Running Ingestion Verification Script...")
    print("==================================================")
    
    # 1. Create a sample resume text file if it doesn't exist
    sample_path = Path("data/raw/candidate_resumes/jane_doe_resume.txt")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    
    sample_text = """
JANE DOE
jane.doe@example.com | (123) 456-7890 | Boston, MA

PROFESSIONAL SUMMARY
Experienced Senior Software Engineer with a strong background in building scalable web APIs and machine learning systems.

SKILLS
Python, FastAPI, PostgreSQL, Redis, Docker, Machine Learning, Qdrant

EXPERIENCE
Tech Innovations, Boston, MA
Senior Software Engineer (January 2021 - Present)
- Designed and built a real-time analytics API using FastAPI and PostgreSQL.
- Migrated legacy data pipeline to Celery and Redis, reducing processing time by 45%.
- Implemented vector search using Qdrant.

EDUCATION
State University, Boston, MA
B.S. in Computer Science (September 2016 - May 2020)
"""
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_text.strip())
    print(f"Created sample resume file at: {sample_path}")
    
    # 2. Run Docling Parsing
    print("\n--- [Step 1: Docling Layout Parsing] ---")
    try:
        parser = DoclingLayoutParser()
        raw_payload = parser.parse_document(sample_path)
        print("Docling Parsing Successful!")
        print(f"File Name: {raw_payload['metadata']['file_name']}")
        print(f"Headers Found: {raw_payload['headers']}")
        print("Raw Converted Text Snippet:")
        print("-" * 40)
        print(raw_payload['raw_text'][:250] + "...")
        print("-" * 40)
    except Exception as e:
        print(f"Docling Parsing Failed: {e}")
        print("Make sure 'docling' is installed in the environment to run the real parser.")
        return
        
    # 3. Run LLM Extraction (requires API Key)
    print("\n--- [Step 2: Instructor LLM Extraction] ---")
    api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "mock-key-for-development":
        print("Skipped LLM Extraction: OPENAI_API_KEY or VLLM_API_KEY not configured in the environment.")
        print("Please configure your API key in .env or system environment variables to test LLM extraction.")
        return
        
    try:
        import instructor
        from openai import OpenAI
        
        vllm_url = os.environ.get("VLLM_API_URL")
        if vllm_url:
            print(f"Using local vLLM endpoint: {vllm_url}")
            openai_client = OpenAI(base_url=vllm_url, api_key=api_key)
            model_name = "local-model"
        else:
            print("Using OpenAI GPT-4o-mini endpoint...")
            openai_client = OpenAI(api_key=api_key)
            model_name = "gpt-4o-mini"
            
        instructor_client = instructor.from_openai(openai_client)
        
        print("Submitting text to LLM for CandidateProfile schema extraction...")
        profile = instructor_client.chat.completions.create(
            model=model_name,
            response_model=CandidateProfile,
            messages=[
                {"role": "system", "content": "You are a professional recruiting assistant. Extract the structured candidate profile from the raw resume text. Extract skills, duration details, companies, and work/education timelines accurately."},
                {"role": "user", "content": f"Resume Text:\n\n{raw_payload['raw_text']}"}
            ],
            temperature=0.0
        )
        print("LLM Extraction Successful!")
        print("-" * 40)
        print(json.dumps(profile.model_dump(), indent=2))
        print("-" * 40)
        
    except Exception as e:
        print(f"LLM Extraction Failed: {e}")
        
    print("\nVerification execution finished.")

if __name__ == "__main__":
    run_verification()
