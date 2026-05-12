<p align="center">
  <picture>
    <img src="https://img.shields.io/badge/AGENTIC_RAG-MULTI_AGENT_SYSTEM-00d4ff?style=for-the-badge" alt="Agentic RAG">
  </picture>
</p>

<h1 align="center">Marine Structures AI Tutor<br><sub>Agentic RAG · Multi-Agent Exam Tutoring System</sub></h1>

<p align="center">
  <b>LangGraph · DeepSeek V4 Pro · BGE-M3 · ChromaDB · FastAPI · Docker</b>
</p>

<p align="center">
  <a href="https://zshyc.github.io/cer-rag/"><img src="https://img.shields.io/badge/🌐_Website-zshyc.github.io/cer--rag-00d4ff?style=flat-square"></a>
  <a href="https://github.com/ZSHYC/cer-rag"><img src="https://img.shields.io/github/stars/ZSHYC/cer-rag?style=flat-square&color=yellow"></a>
  <a href="https://github.com/ZSHYC/cer-rag/commits/main"><img src="https://img.shields.io/github/last-commit/ZSHYC/cer-rag?style=flat-square&color=green"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square">
  <img src="https://img.shields.io/badge/tests-19/19_passed-brightgreen?style=flat-square">
</p>

<p align="center">
  📖 <a href="README.md">中文版</a> · 🌐 <a href="https://zshyc.github.io/cer-rag/">Project Website</a>
</p>

---

> An **Agentic RAG** multi-agent exam tutoring system powered by **LangGraph + DeepSeek V4 Pro**.
> Not a "plug data into LangChain" demo — it traces the **complete evolution** from Naive RAG → Advanced RAG → Agentic RAG → Production deployment.
>
> Built for **SESS3026/JEIS3005 "Marine Structures"** at the University of Southampton.

---

## 📑 Table of Contents

1. [What & Why](#1-what--why)
2. [Architecture](#2-architecture)
3. [Quick Start](#3-quick-start)
4. [Usage Guide](#4-usage-guide)
   - [CLI](#41-cli) · [Gradio UI](#42-gradio-ui) · [FastAPI](#43-fastapi) · [Docker](#44-docker) · [RAGAS Evaluation](#45-ragas-evaluation) · [Knowledge Graph API](#46-knowledge-graph-api) · [Custom Agent Flow](#47-custom-agent-flow)
5. [Data Pipeline](#5-data-pipeline)
6. [Tech Stack](#6-tech-stack)
7. [Project Structure](#7-project-structure)
8. [Lessons Learned](#8-lessons-learned)
9. [Development Roadmap](#9-development-roadmap)

---

## 1. What & Why

### The Evolution of RAG — In One Project

This project demonstrates the **full RAG maturity curve**:

| Stage | Architecture | Key Innovation |
|-------|-------------|---------------|
| **Naive RAG** | Retrieve → Generate | Dense + Sparse Hybrid, RRF Fusion |
| **Advanced RAG** | Retrieve → Rerank → Generate | BGE Cross-encoder, Multi-strategy Chunking |
| **Agentic RAG** | Supervisor → 5 Specialist Agents | Self-RAG Reflection Loop, Tool-use |
| **Production** | FastAPI SSE + Docker | RAGAS Evaluation, A/B Testing |

### Project Stats

| Metric | Value |
|--------|-------|
| Python Source Files | 31 |
| Lines of Code | ~2,800 |
| Agent Nodes (LangGraph) | 7 |
| Indexed Chunks | 738 |
| Knowledge Graph Nodes / Edges | 79 / 269 |
| Development Iterations | 5 |
| Tests Passed | 19/19 |

### Key Technical Highlights

- **6-Agent Multi-Agent System** — Supervisor orchestrates Retrieval (Self-RAG), Concept, Calculation, Exam, and Critic agents via LangGraph state machine
- **Self-RAG / CRAG Pattern** — Retrieval agent evaluates sufficiency, rewrites queries, and re-searches — paper-backed (Self-RAG '23, CRAG '24)
- **Three-Layer Indexing** — Dense vectors (BGE-M3 · 1024-dim) + Sparse keywords (BM25) + Knowledge Graph (NetworkX · 79 nodes, 269 edges)
- **Cross-lingual** — English lectures + Chinese study notes in one unified semantic space
- **Production-grade** — FastAPI SSE streaming, Docker Compose one-command deploy, LangSmith tracing
- **Evaluation-driven** — RAGAS metrics (faithfulness, context precision/recall, answer relevancy) with A/B strategy comparison

### 🎤 Interview Pitch

> "I built a multi-agent AI tutoring system with LangGraph. A Supervisor agent orchestrates 5 specialists — retrieval with Self-RAG reflection, concept explanation, calculation solving, exam generation/grading, and quality critique. I use a three-layer indexing system: dense vectors, sparse keywords, and a knowledge graph. The system has a FastAPI SSE backend, Gradio UI, Docker deployment, and a RAGAS evaluation pipeline. This project gave me a deep understanding of the full RAG evolution from naive retrieval to agentic orchestration."

---

## 2. Architecture

```
  Interface Layer:       CLI (Rich) · Gradio UI · FastAPI SSE
  ─────────────────────────────────────────────────────────
  LangGraph Multi-Agent:
     Supervisor Agent (Plan → Delegate → Synthesize)
        ├── Retrieval Agent  —  Self-RAG reflection loop
        ├── Concept Agent    —  Cross-source verification
        ├── Calc Agent       —  Step-by-step formula solving
        ├── Exam Agent       —  Generate + Grade
        └── Critic Agent     —  Hallucination detection
  ─────────────────────────────────────────────────────────
  Shared Tool Layer:
     dense_search · sparse_search · graph_traverse · rerank
  ─────────────────────────────────────────────────────────
  Three-Layer Indexing:
     Layer 1: Dense Vectors   — ChromaDB + BGE-M3 (1024-dim)
     Layer 2: Sparse Keywords — BM25 (exact formula/term match)
     Layer 3: Knowledge Graph — NetworkX
              Concept ↔ Formula ↔ Lecture ↔ Exam ↔ Solution
  ─────────────────────────────────────────────────────────
  Infrastructure:     Docker · LangSmith · RAGAS
```

---

## 3. Quick Start

```bash
git clone https://github.com/ZSHYC/cer-rag.git && cd cer-rag
conda create -n cer_rag python=3.11 -y && conda activate cer_rag
pip install -r requirements.txt
cp .env.example .env              # add your DeepSeek API key
python rag/build_index.py         # one-time, ~5 min
```

**Four launch options:**

```bash
python rag/cli.py                 # ① CLI with Rich formatting
python rag/ui/app.py              # ② Gradio UI → http://localhost:7860
python -m rag.server.main         # ③ FastAPI → http://localhost:8000
docker-compose up -d              # ④ Docker deployment
```

---

## 4. Usage Guide

### 4.1 CLI

```bash
python rag/cli.py
```

| Key | Action |
|-----|--------|
| `:q` | Quit |
| `:s` | Toggle source display |
| `:d` | **Debug mode** — live agent execution trace |

**Agent routing table:**

| Query Type | Example | LangGraph Path |
|-----------|---------|---------------|
| Conceptual | "What is Rational Ship Structural Design?" | supervisor → retrieval → concept |
| Calculation | "Calculate the shape factor" | supervisor → retrieval → calc |
| Formula Lookup | "What is Faulkner's effective width?" | supervisor → retrieval → concept |
| Exam Generation | "Generate 3 questions about buckling" | supervisor → exam_generate |
| Cross-lingual | "用中文解释 plastic collapse" | Auto-detect, bilingual |

<details>
<summary>🔍 Debug mode output example (click to expand)</summary>

```
You: Calculate the shape factor of a square section
[calculation] thinking...
  │ [supervisor] plan=['retrieve formula', 'calculate'] docs=0 retries=0
  │ [retrieval] plan=[] docs=8 retries=1
  │ [calc] plan=[] docs=0 retries=0

Tutor: Z = a³/6, Zp = a³/4, φ = Zp/Z = 1.5

Plan: retrieve formula › calculate

Sources (8 docs):
  #  Source                              Preview
  1  [Review: 10_核心题型...]             Zp = (a²/2)(a/4) + ...
  2  [Answer 2022/23 Q2(iii)]            Derive the shape factor...
  3  [Lecture 18, Slide 14-16]           Plastic section modulus...
```
</details>

### 4.2 Gradio UI

```bash
python rag/ui/app.py
# → http://localhost:7860
```

- Full Agentic RAG chat panel
- Debug Mode toggle for real-time agent trace
- Pre-set example questions
- Auto source citation display
- Multi-turn conversation memory

### 4.3 FastAPI

```bash
python -m rag.server.main
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | SSE streaming chat |
| `POST` | `/exam/generate` | Generate mock exam questions |
| `POST` | `/exam/grade` | Grade student answers |

```bash
# Health check
curl http://localhost:8000/health

# SSE Chat
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is plate buckling?"}'

# Generate exam
curl -X POST http://localhost:8000/exam/generate \
  -H "Content-Type: application/json" \
  -d '{"topic":"buckling","num_questions":3}'

# Grade answer
curl -X POST http://localhost:8000/exam/grade \
  -H "Content-Type: application/json" \
  -d '{"question":"Calculate shape factor","student_answer":"φ = 1.5"}'
```

<details>
<summary>📡 SSE event stream format (click to expand)</summary>

```
data: {"node":"supervisor","plan":["search","explain"],"docs":0}
data: {"node":"retrieval","plan":[],"docs":8}
data: {"node":"concept","plan":[],"docs":0}
data: {"answer":"Plate buckling is a structural instability..."}
data: [DONE]
```
</details>

### 4.4 Docker

```bash
docker-compose up -d          # start (port 8000)
docker-compose logs -f        # view logs
docker-compose down           # stop
```

```yaml
# docker-compose.yml key config
services:
  tutor-api:
    build: .                    # Build from Dockerfile
    ports: ["8000:8000"]
    environment:                # Env vars from .env
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
    volumes:
      - ./data:/app/data        # Persistent index data
      - ./.env:/app/.env:ro     # Read-only config
    restart: unless-stopped
```

### 4.5 RAGAS Evaluation

```bash
python rag/evaluation/ragas_eval.py
```

Uses **5 real exam questions** as ground truth. Outputs per-case metrics: retrieved doc count, Self-RAG retry count, answer preview.

### 4.6 Knowledge Graph API

```python
from rag.indexing.knowledge_graph import get_kg, graph_traverse

kg = get_kg()                         # 79 nodes, 269 edges
results = graph_traverse("buckling")
# → [Formula] σcr = kπ²E/[12(1-ν²)]·(t/b)²
# → [Exam] 2022/23 Q3, 2023/24 Q3, 2024/25 Q3
# → [Lecture] L13: Buckling of Plates
```

Graph relations:

```
Concept ──EXPLAINED_IN──→ Lecture
Concept ──TESTED_BY─────→ ExamQuestion
Formula ──RELATED_TO────→ Concept
```

### 4.7 Custom Agent Flow

```python
from rag.graph.build_graph import build_tutor_graph

# Two compilation modes
graph = build_tutor_graph(enable_critic=False)   # Direct output
graph = build_tutor_graph(enable_critic=True)    # With Critic validation

state = {"query": "...", "query_type": "...", ...}
for step in graph.stream(state, {"recursion_limit": 20}):
    node = list(step.keys())[0]
    print(f"[{node}]", step[node].get("plan", []))
```

---

## 5. Data Pipeline

**738 chunks** across 5 heterogeneous sources, each with a tailored chunking strategy:

| Source | Scale | Chunks | Strategy | Metadata |
|--------|-------|--------|----------|----------|
| Lecture Slides | 21 PDFs · 582 pages | 200 | 3-slide merged windows | lecture_number, slide_range, topic_tags |
| Exam Papers | 9 years (2014–2025) | 25 | Question-boundary split | exam_year, question_number, marks |
| Answer Keys | 3 years (2022–2025) | 15 | Sub-question paired | exam_year, question_number, sub_question |
| Seminar Solutions | 4 Tutorials | 23 | Tutorial→Question hierarchy | tutorial_number |
| Chinese Study Notes | 12 Markdown files | 475 | Markdown header-aware | file_topic, breadcrumb, language |

---

## 6. Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **LLM** | DeepSeek V4 Pro | Strong Chinese+math reasoning, OpenAI-compatible API |
| **Agent Framework** | LangGraph | State machine multi-agent orchestration, conditional routing |
| **Embedding** | BGE-M3 (FlagEmbedding) | 1024-dim, 100+ languages, free & local |
| **Vector DB** | ChromaDB | Metadata filtering, zero-ops |
| **Sparse Search** | BM25 (rank_bm25) | Exact formula/term matching |
| **Reranker** | BGE-reranker-v2-m3 | Cross-encoder precision boost |
| **Knowledge Graph** | NetworkX | Lightweight, 79 nodes, 269 edges |
| **Backend** | FastAPI + SSE | Async streaming responses |
| **UI** | Gradio + Rich CLI | Rapid prototyping + polished terminal |
| **Evaluation** | RAGAS | Faithfulness, precision, recall, relevancy |
| **Deploy** | Docker Compose | One-command production |

---

## 7. Project Structure

```
rag/
├── cli.py · config.py · build_index.py
├── ITERATION_0.md ~ ITERATION_4.md        # Iteration docs
│
├── preprocess/      # PyMuPDF extraction + normalization + 5 chunking strategies
├── indexing/        # ChromaDB + BM25 + RRF fusion + Knowledge Graph (NetworkX)
├── agents/          # 7 Agents: supervisor, retrieval, concept, calc, exam, critic + tools
├── graph/           # LangGraph compiler + router
├── prompts/         # 8 prompt templates
├── evaluation/      # RAGAS evaluation + A/B strategy comparison
├── server/          # FastAPI (SSE streaming)
├── ui/              # Gradio Web UI
└── tests/
```

---

## 8. Lessons Learned

<details open>
<summary><b>1. BGE-M3 belongs to FlagEmbedding, not sentence-transformers</b></summary>

Loading via `sentence-transformers` triggers `TextEncodeInput must be Union[...]` errors. BGE-M3 is a **FlagEmbedding** model — use `FlagEmbedding.BGEM3FlagModel` directly, and implement a custom LangChain `Embeddings` wrapper.
</details>

<details>
<summary><b>2. Chunking strategy is the #1 RAG hyperparameter</b></summary>

Switching from fixed-size (1000-token) to source-aware chunking improved retrieval precision by **2-3×**. Different source types need fundamentally different granularities.
</details>

<details>
<summary><b>3. PDF slide extraction is non-trivial</b></summary>

`pdftotext` produces garbled Greek letters and subscripts (`𝞮𝞮` → should be `EI`). Requires a **100+ character normalization map**. Diagrams embedded in slides are lost — **cite slide numbers** as references instead.
</details>

<details>
<summary><b>4. Exam paper formats vary significantly across years</b></summary>

Across 9 years of exams, at least **3 different question marking formats** exist. Multi-pattern regex compatibility is essential for correct question-boundary splitting.
</details>

---

## 9. Development Roadmap

This project was built over 5 deliberate iterations, each with its own technical document covering design rationale, debugging experiences, and test results:

| # | Focus | Key Output | Full Report |
|---|-------|------------|-------------|
| 0 | Basic RAG | Document processing, 738 chunks, 3-layer indexing (dense+sparse+KG placeholder), hybrid RRF retrieval, Rich CLI | [📄 ITERATION_0.md](rag/ITERATION_0.md) |
| 1 | Agentic Upgrade | LangGraph state graph, Supervisor Agent (planning+routing), Retrieval Agent Self-RAG reflection loop, Concept & Calc specialist agents | [📄 ITERATION_1.md](rag/ITERATION_1.md) |
| 2 | Knowledge Graph | 79-node 269-edge KG (Concept↔Formula↔Lecture↔Exam), graph-traversal enhanced search, Exam Agent (generate+grade), Critic Agent (hallucination detection) | [📄 ITERATION_2.md](rag/ITERATION_2.md) |
| 3 | Productionization | FastAPI + SSE streaming backend, Docker Compose one-command deploy, RAGAS evaluation (5 ground-truth cases), A/B strategy comparison | [📄 ITERATION_3.md](rag/ITERATION_3.md) |
| 4 | Polish | Gradio Web UI, 19/19 comprehensive tests passed, bilingual documentation, project website | [📄 ITERATION_4.md](rag/ITERATION_4.md) |

Each iteration doc includes: design rationale, pitfalls encountered and their solutions, complete test results, and a new-file manifest.

---

## License

MIT — free to use, adapt, and learn from.

---

<p align="center">
  <sub>Built with LangGraph · DeepSeek V4 Pro · ChromaDB · BGE-M3 · NetworkX · FastAPI</sub>
</p>
