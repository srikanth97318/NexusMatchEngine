# NexusMatch Engine
### AI-Powered Resume Ranking & Candidate Matching Platform

NexusMatch Engine is an advanced AI-driven recruitment system that intelligently matches candidates with job descriptions using Hybrid Retrieval, Machine Learning Re-ranking, and Explainable AI.

The system goes beyond keyword matching by combining semantic search, structured parsing, ranking algorithms, and LLM reasoning to generate highly accurate candidate shortlists.

---

# 🌟 Key Features

✅ Layout-aware Resume Parsing

✅ Structured JSON Extraction using LLMs

✅ Hybrid Dense + Sparse Retrieval

✅ Metadata Filtering

✅ Machine Learning Re-ranking (LambdaMART)

✅ Explainable AI using SHAP

✅ Hallucination Prevention

✅ Fraud Detection & Anti-cheat Mechanisms

✅ High-performance FastAPI Backend

✅ Interactive Recruiter Dashboard

---

# 🏗 System Architecture

```text
Recruiter UI
      │
      ▼
FastAPI Gateway
      │
 ┌────┴───────────────┐
 │                    │
 ▼                    ▼
Celery Workers      Qdrant Vector DB
 │                    │
 ▼                    ▼
PostgreSQL      Hybrid Retrieval
                      │
                      ▼
              LightGBM Ranker
                      │
                      ▼
                 vLLM Engine
                      │
                      ▼
               Final Candidate List
```

---

# 📌 Problem Statement

Traditional Applicant Tracking Systems (ATS) rely heavily on keyword matching, leading to:

- Poor semantic understanding
- Resume formatting issues
- High false positives
- Lack of explainability
- Vulnerability to keyword stuffing

NexusMatch solves these problems using state-of-the-art AI pipelines.

---

# ⚙️ Tech Stack

## Backend
- FastAPI
- Python
- Celery
- Redis
- PostgreSQL

## AI/ML
- BGE-M3 Embeddings
- LightGBM (LambdaMART)
- SHAP Explainability
- vLLM
- Llama 3

## Retrieval
- Qdrant Vector Database
- Hybrid Search
- Reciprocal Rank Fusion (RRF)

## Parsing
- Docling
- Instructor
- Pydantic

## Frontend
- Streamlit / React

## Deployment
- Docker
- Docker Compose

---

# 🔄 End-to-End Workflow

## Step 1: Resume Parsing

The system accepts resumes in:

- PDF
- DOCX

Docling performs layout-aware parsing to preserve:

- Headers
- Tables
- Lists
- Multi-column structures

---

## Step 2: Structured Extraction

LLMs convert parsed text into strongly typed JSON schemas.

Example:

```json
{
  "name": "John Doe",
  "skills": ["Python", "Docker"],
  "experience": 4,
  "education": "B.Tech"
}
```

---

## Step 3: Vector Embedding

BGE-M3 generates:

- Dense embeddings
- Sparse embeddings
- Token-level embeddings

This enables both semantic and exact keyword matching.

---

## Step 4: Hybrid Retrieval

Candidate retrieval uses:

### Dense Search

Captures semantic meaning.

### Sparse Search

Captures exact keywords.

### Reciprocal Rank Fusion

\[
RRF = \frac{1}{60+Rank_{dense}} + \frac{1}{60+Rank_{sparse}}
\]

Top-100 candidates are retrieved in milliseconds.

---

## Step 5: Machine Learning Re-ranking

Features engineered:

### Semantic Match Score

Measures similarity with job description.

### Career Velocity

\[
Velocity=\frac{\Delta Title}{\Delta Years}
\]

### Skill Recency Decay

\[
Weight=Base\times e^{-\lambda t}
\]

### Tenure Stability

Detects job hopping.

These features are passed into a LightGBM LambdaMART model.

---

# 📊 Ranking Metric

The model optimizes:

\[
NDCG@K=\frac{DCG@K}{IDCG@K}
\]

where

\[
DCG@K=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}
\]

---

# 🧠 Explainable AI

NexusMatch uses SHAP values to explain rankings.

Example:

| Feature | Contribution |
|---------|-------------|
| Skill Match | +22% |
| Career Growth | +18% |
| Stability | +10% |
| Skill Recency | +12% |

This improves recruiter trust and transparency.

---

# 🛡 Hallucination Prevention

To eliminate LLM hallucinations:

- Exact text-span verification
- Source grounding
- Character index matching
- Automatic re-validation

Generated explanations must map back to original resumes.

---

# 🔐 Fraud Detection

The system detects:

### Keyword Stuffing

\[
Density=\frac{Unique\ Technical\ Terms}{Total\ Tokens}
\]

### Chronological Overlap

Flags impossible employment histories.

---

# 📈 Performance

| Stage | Time |
|-------|------|
| Retrieval | <20 ms |
| Re-ranking | <5 ms |
| Final Ranking | Real-time |

Supports repositories with:

- 50,000+ resumes

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/srikanth97318/NexusMatchEngine.git
cd NexusMatch
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Services

```bash
docker-compose up --build
```

## Start API

```bash
uvicorn app.main:app --reload
```

---

# 🎯 Future Enhancements

- Multilingual Resume Parsing
- Interview Scheduling Agent
- Resume Feedback Generation
- Bias Detection Module
- Agentic AI Recruiter Assistant
- Real-time Analytics Dashboard

---

# 👨‍💻 Contributors

- Member 1 — Parsing & Extraction
- Member 2 — Retrieval Engine
- Member 3 — Ranking & Explainability
- Member 4 — Infrastructure & Deployment

---

# 📜 License

MIT License

---

## ⭐ If you found this project useful, please star the repository!
