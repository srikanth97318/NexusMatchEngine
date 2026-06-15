import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="NexusMatch Engine Dashboard", layout="wide")

st.title("🎯 NexusMatch Engine Analytics Panel")
st.subheader("Interactive Talent Matching, LTR Re-ranking, and Explainability Studio")

st.markdown("""
This interface interacts with the FastAPI backend orchestration layer. 
Upload resumes to ingest profiles or enter job requirements below to evaluate candidates.
""")

# Backend server address
API_URL = "http://localhost:8000"

# Sidebar controls
st.sidebar.header("📁 Ingestion Panel")
uploaded_file = st.sidebar.file_uploader("Upload Candidate Resume (PDF / DOCX)", type=["pdf", "docx"])
if uploaded_file is not None:
    if st.sidebar.button("Process Resume"):
        st.sidebar.info("Queued Celery background ingestion pipeline...")

# Main dashboard columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 Job Requirements")
    job_id = st.text_input("Job ID", "job_data_science_01")
    job_title = st.text_input("Target Job Title", "Senior Data Scientist")
    jd_text = st.text_area("Unstructured Job Description", 
                           "We are looking for a Senior Data Scientist with 5+ years of experience in Python, ML pipelines, and vector databases like Qdrant.")
    skills_required = st.text_input("Required Skills (Comma separated)", "Python, Machine Learning, Qdrant")
    min_exp = st.slider("Min Years of Experience", 0, 15, 5)

with col2:
    st.header("🏆 Matched Candidates")
    if st.button("Run Match Engine Execution Loop", type="primary"):
        # Mock request to FastAPI backend
        skills_list = [s.strip() for s in skills_required.split(",")]
        
        st.write("🔍 Running BGE-M3 Dense/Sparse embedding generations...")
        st.write("🛰️ Querying Qdrant index vector partitions...")
        st.write("⚡ Computing LambdaMART feature engineering matrices...")
        
        # Mock response mimicking FastAPI backend output
        mock_results = [
            {
                "rank": 1,
                "id": "cand_001",
                "name": "Jane Doe",
                "ltr_score": 0.82,
                "llm_rationale": "Strong fit. Over 5 years of Python & ML. Mentions Qdrant explicitly.",
                "shap": {"skill_match_score": 0.35, "semantic_similarity": 0.25, "trajectory_velocity": 0.15, "exp_ratio": 0.07}
            },
            {
                "rank": 2,
                "id": "cand_003",
                "name": "Alice Smith",
                "ltr_score": 0.71,
                "llm_rationale": "High experience match. Lacks explicit Qdrant experience but strong ML history.",
                "shap": {"skill_match_score": 0.20, "semantic_similarity": 0.30, "trajectory_velocity": 0.12, "exp_ratio": 0.09}
            }
        ]
        
        for res in mock_results:
            with st.container():
                st.write(f"### Rank {res['rank']}: {res['name']} ({res['id']})")
                st.write(f"**Final Score:** `{res['ltr_score']:.2f}`")
                st.write(f"**LLM Explainability Rationale:** {res['llm_rationale']}")
                
                # Render feature importance charts
                shap_df = pd.DataFrame(list(res['shap'].items()), columns=["Feature", "Impact (SHAP)"])
                st.bar_chart(shap_df.set_index("Feature"))
                st.markdown("---")
