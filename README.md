# NexusMatch Engine

NexusMatch Engine is a production-grade talent acquisition and skill-matching engine, structured to support high-performance parsing, hybrid retrieval, machine learning ranking (LTR), and LLM validation.

## Architecture Component Blueprint

The system is composed of four modular engine layers:

### 1. File Ingestion & Parsing Core (`src/member1_ingestion`)
- **Document Scaffolding:** Utilizes layout-aware extraction models (`docling` framework) to parsing multi-column resume PDFs and JDs.
- **Data Validation:** Employs Pydantic schemas paired with `instructor` to enforce type-safety and structural definitions on candidates and job requirements.
- **Queue Task Routing:** Offloads CPU-intensive processing tasks to **Celery workers** backed by a **Redis broker**.

### 2. Embedding Vector Generation & Discovery (`src/member2_retrieval`)
- **Multi-Vector Representations:** Configured with `BGE-M3` embeddings generating Dense (1024-dim), Sparse (token-based weights), and Late-interaction ColBERT tensors.
- **Relational Storage:** Maintains candidate logs, matching session audits, and profile entities in a persistent PostgreSQL layer.
- **Vector Search Index:** Performs ultra-low latency hybrid approximate nearest neighbor (ANN) retrieval using **Qdrant**'s vector filtering and fusion engines.

### 3. Predictive Re-ranking & Contextual Refinement (`src/member3_ranking`)
- **Feature Engineering:** Computes normalized matching factors including years of experience delta, candidate trajectory velocity, and skill decay scales.
- **Learning to Rank (LTR):** Implements offline training for a **LightGBM LambdaMART** pairwise model using relative candidate feedback logs.
- **Listwise Refinement:** Routes retrieved top candidates to local **vLLM** hosted instances to execute listwise re-ordering prompts with strict JSON schema constraints.
- **Attribution & Transparency:** Outputs feature impact scores utilizing **SHAP (SHapley Additive exPlanations)** coefficients per ranking query.

### 4. Production API Systems Gateway & User Interface (`src/member4_orchestration`)
- **FastAPI Core Gateway:** Orchestrates end-to-end processing loops (Ingestion -> Retrieval -> LTR -> LLM -> Output).
- **Streamlit Analytics Panel:** Renders uploaded resumes, job search criteria, LTR match dashboards, and interactive SHAP charts.

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (if running bare-metal)

### Running via Docker Compose (Recommended)
This provisions services for FastAPI, Streamlit UI, Postgres, Qdrant, Redis, and Celery workers:

```bash
# Clone the repository
cd nexusmatch-engine

# Start the cluster
docker-compose up --build
```
- **FastAPI API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Web Interface:** [http://localhost:8501](http://localhost:8501)

### Bare-Metal Setup
1. **Initialize Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Configure Environment Variables:**
   ```bash
   cp .env.example .env
   # Modify values as needed
   ```
3. **Execute Test Suite:**
   ```bash
   pytest tests/
   ```
