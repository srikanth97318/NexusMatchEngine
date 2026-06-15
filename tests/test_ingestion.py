from pathlib import Path
from src.member1_ingestion.parser import DoclingLayoutParser
from src.member1_ingestion.schemas import CandidateProfile


def test_docling_parser_stub(tmp_path):
    # Setup temporary file as text so Docling parses it correctly without PDF binary expectations
    test_file = tmp_path / "resume_jane.txt"
    test_file.write_text("Unstructured resume content")
    
    parser = DoclingLayoutParser()
    result = parser.parse_document(test_file)
    
    assert "metadata" in result
    assert result["metadata"]["file_name"] == "resume_jane.txt"
    assert "raw_text" in result


def test_candidate_profile_validation():
    profile = CandidateProfile(
        name="Jane Doe",
        skills=["Python", "SQL"],
        experience=[
            {"company": "Google", "role": "Software Engineer", "duration_months": 24}
        ],
        education=[]
    )
    assert profile.name == "Jane Doe"
    assert len(profile.experience) == 1
    assert profile.experience[0].company == "Google"
