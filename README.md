# Marine Structures AI Tutor

> An **Agentic RAG** multi-agent exam tutoring system powered by LangGraph + DeepSeek V4 Pro.
> Built for SESS3026/JEIS3005 "Marine Structures" at the University of Southampton.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-Agentic%20RAG-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/LLM-DeepSeek%20V4%20Pro-purple.svg" alt="DeepSeek">
  <img src="https://img.shields.io/badge/Embedding-BGE--M3-orange.svg" alt="BGE-M3">
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-red.svg" alt="ChromaDB">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

---

##  What Makes This Project Stand Out

This is **not** a simple "plug data into LangChain" demo. It demonstrates the **full evolution path of modern RAG systems**:

| Stage | Architecture | Key Innovation |
|-------|-------------|---------------|
| Naive RAG | Retrieve → Generate | Dense + Sparse Hybrid, RRF Fusion |
| Advanced RAG | Retrieve → Rerank → Generate | BGE Cross-encoder, Multi-strategy Chunking |
| Agentic RAG | **Supervisor → Specialist Agents** | Self-RAG Reflection Loop, Tool-use |
| Production | FastAPI + Streaming + Docker | RAGAS Evaluation, A/B Testing, Observability |

### Key Technical Highlights

- **6-Agent Multi-Agent System**: Supervisor orchestrates Retrieval (self-reflection), Concept, Calculation, Exam, and Critic agents via LangGraph state machine
- **Self-RAG / CRAG Pattern**: Retrieval agent evaluates retrieval sufficiency and rewrites queries for re-retrieval — inspired by Self-RAG and Corrective RAG papers
- **Three-Layer Indexing**: Dense vectors (BGE-M3) + Sparse keywords (BM25) + Knowledge Graph (NetworkX) for structured concept-formula-exam traversal
- **Cross-lingual**: Seamlessly handles English course materials and Chinese study notes in a unified semantic space (BGE-M3 multilingual embeddings)
- **RAG Evaluation**: Integrated RAGAS metrics (faithfulness, context precision/recall, answer relevancy) with A/B strategy comparison
- **Production-grade**: FastAPI + SSE streaming + Docker Compose + LangSmith observability

---

##  Architecture

```
┌──────────────────────────────────────────────────┐
│              CLI (Rich) / FastAPI / Gradio         │
├──────────────────────────────────────────────────┤
│           LangGraph Multi-Agent Layer              │
│                                                    │
│   ┌──────────── Supervisor Agent ────────────┐    │
│   │   "Plan → Delegate → Synthesize"          │    │
│   └──┬─────────┬──────────┬──────────┬───────┘    │
│      │         │          │          │             │
│   ┌──▼───┐ ┌──▼───┐ ┌───▼──┐ ┌───▼────┐ ┌───▼──┐ │
│   │Retrv │ │Concpt│ │Calc  │ │Exam    │ │Critic│ │
│   │Agent │ │Agent │ │Agent │ │Agent   │ │Agent │ │
│   └──┬───┘ └──────┘ └──────┘ └────────┘ └──────┘ │
│      │                                            │
├──────┼────────────────────────────────────────────┤
│   ┌──▼────────────────────────────────────────┐   │
│   │          Tool Layer                        │   │
│   │  dense_search | sparse_search | graph_trav │   │
│   │  formula_lookup | calculator | rerank      │   │
│   └───────────────────────────────────────────┘   │
├──────────────────────────────────────────────────┤
│         Three-Layer Indexing                      │
│  Dense (ChromaDB+BGE-M3) | Sparse (BM25) | KG     │
├──────────────────────────────────────────────────┤
│    Infra: Docker | LangSmith | RAGAS | CI/CD      │
└──────────────────────────────────────────────────┘
```

---

##  Data Processing Pipeline

**738 indexed chunks** across 5 heterogeneous sources with source-aware chunking:

| Source | Documents | Chunks | Strategy |
|--------|-----------|--------|----------|
| Lecture Slides | 21 PDFs (582 pages) | 200 | 3-slide merged windows |
| Exam Papers | 9 years (2014-2025) | 25 | Question-boundary split |
| Answer Keys | 3 years (2022-2025) | 15 | Sub-question paired chunks |
| Seminar Solutions | 4 Tutorials | 23 | Tutorial → Question hierarchical |
| Chinese Study Notes | 12 Markdown files | 475 | Markdown header-aware |

Each chunk carries rich metadata: source type, year, lecture number, topic tags, content type, language.

---

##  Quick Start

### Prerequisites
- Conda (Miniforge)
- DeepSeek API key ([get one here](https://platform.deepseek.com))

### Setup

```bash
# 1. Clone
git clone https://github.com/ZSHYC/cer-rag.git
cd cer-rag

# 2. Environment
conda create -n cer_rag python=3.11 -y
conda activate cer_rag
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your DeepSeek API key

# 4. Build index (one-time, ~5 min)
python rag/build_index.py

# 5. Launch CLI
python rag/cli.py
```

### Usage

```
You: What is Rational Ship Structural Design?

Tutor: Rational Ship Structural Design is a mechanics-based, optimization-driven
design approach that directly uses structural theory, FEA, and computational
optimization rather than empirical rule-based formulas...

Sources:
  1. [Review: 09_核心题型完整答题模板_理论论述题.md] (score: 0.620)
  2. [Exam 2021/22 Q1] — exact exam question match
  3. [Lecture 1, Slide 5-7] — foundational definition
```

---

##  Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **LLM** | DeepSeek V4 Pro (API) | Strong Chinese+math reasoning, OpenAI-compatible |
| **Agent Framework** | LangGraph | State machine orchestration for multi-agent |
| **Embedding** | BGE-M3 (FlagEmbedding) | Multilingual (100+ langs), 1024-dim, local & free |
| **Vector DB** | ChromaDB | Zero-ops, metadata filtering, Python-native |
| **Sparse Retrieval** | BM25 (rank_bm25) | Exact keyword matching for formulas/terms |
| **Reranking** | BGE-reranker-v2-m3 | Cross-encoder precision boost |
| **Evaluation** | RAGAS | Faithfulness, relevancy, precision/recall metrics |
| **Backend** | FastAPI + SSE | Async streaming responses |
| **UI** | Gradio + Rich CLI | Quick prototyping + polished terminal UX |
| **Deploy** | Docker Compose | One-command production deployment |
| **Observability** | LangSmith | Full agent call chain tracing |

---

##  Project Structure

```
rag/
├── cli.py                     # Interactive CLI (Rich)
├── config.py                  # Global configuration
├── build_index.py             # Index construction entry
│
├── preprocess/                # Document processing
│   ├── extract_lectures.py    # PyMuPDF PDF extraction
│   ├── normalize_txt.py       # Symbol normalization (garbled text fix)
│   ├── chunk_lectures.py      # Lecture slide chunking
│   ├── chunk_exams.py         # Exam/solution chunking
│   ├── chunk_markdown.py      # Study notes chunking
│   └── run_chunking.py        # Full chunking pipeline
│
├── indexing/                  # Retrieval layer
│   ├── vector_index.py        # BGE-M3 + ChromaDB
│   ├── sparse_index.py        # BM25 sparse index
│   ├── build_sparse.py        # BM25 persistence
│   └── fusion.py              # RRF + Cross-encoder rerank
│
├── agents/                    # Agentic RAG
│   ├── state.py               # LangGraph shared state
│   ├── supervisor.py          # Query planning & routing
│   ├── retrieval_agent.py     # Search + self-reflection
│   ├── concept_agent.py       # Theoretical explanations
│   ├── calc_agent.py          # Step-by-step calculations
│   └── tools.py               # Shared tool functions
│
├── graph/                     # LangGraph orchestration
├── prompts/                   # Prompt templates
├── evaluation/                # RAGAS evaluation
├── tracking/                  # Knowledge coverage tracking
├── server/                    # FastAPI backend
└── ui/                        # Gradio interface
```

---

##  Development Roadmap

- [x] **Iteration 0**: Basic RAG — document processing, hybrid retrieval, CLI
- [ ] **Iteration 1**: Agentic upgrade — LangGraph multi-agent, Self-RAG reflection
- [ ] **Iteration 2**: Knowledge graph + Exam/Critic agents
- [ ] **Iteration 3**: Productionization — FastAPI, Docker, RAGAS evaluation
- [ ] **Iteration 4**: Polish — Gradio UI, docs, testing

---

##  Key Design Decisions & Trade-offs

| Decision | Why |
|----------|-----|
| Custom pipeline over LangChain LCEL | More control over multi-agent orchestration; LangGraph for state management |
| FlagEmbedding over sentence-transformers | BGE-M3 is a FlagEmbedding model; sentence-transformers has tokenizer incompatibility |
| ChromaDB over FAISS | Metadata filtering needed for source-type scoped queries |
| Source-aware chunking over fixed-size | Critical for retrieval quality — lectures, exams, and formulas need different granularities |
| Unicode math over LaTeX | BGE-M3 embeds Unicode well; LLMs read it natively |
| DeepSeek V4 Pro over GPT-4o | Superior Chinese quality, lower cost, equally strong math reasoning |

---

##  License

MIT License — feel free to use and adapt.

---

<p align="center">
  <sub>Built with LangGraph · DeepSeek · ChromaDB · BGE-M3</sub>
</p>
