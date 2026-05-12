# Marine Structures AI Tutor · 船舶结构物 AI 导师

<p align="center">
  <b>Chinese</b> · <a href="#english">English</a>
</p>

---

# 中文版

## 一句话介绍

**基于 LangGraph 多智能体的 Agentic RAG 考试辅导系统**。不是简单的"调个 LangChain 就完了"的 Demo，而是从朴素 RAG → 高级 RAG → Agentic RAG → 生产部署的**完整演进路径**。

服务于南安普顿大学 SESS3026/JEIS3005「Marine Structures / 船舶结构物」课程期末备考。

## 为什么这个项目值得关注

| 你现在会什么 | 你做完这个项目后能聊什么 |
|-------------|----------------------|
| 调 API 做问答 | 「我设计了一个 6-Agent 多智能体协作系统」 |
| 用 LangChain 搭 RAG | 「我对比了朴素/高级/Agentic 三种 RAG 架构的优劣」 |
| 调 embedding 做搜索 | 「我构建了稠密+稀疏+知识图谱三层索引体系」 |
| 写脚本 | 「我交付了一个 FastAPI + SSE 流式 + Docker 的生产级系统」 |
| 凭感觉说"效果好" | 「我有 RAGAS 量化评估 + A/B 策略对比数据支撑结论」 |

### 面试时这个故事怎么讲

> "我做了一个多智能体 AI 考试导师系统。它不只是一个 RAG——我用 LangGraph 构建了一个 Supervisor Agent 协调 5 个专业 Agent（检索、概念讲解、计算求解、出题批改、质量审查）。检索 Agent 有自省回路——搜完会评估信息是否充分，不够就改写查询重新搜。底层用了三层索引：稠密向量（BGE-M3）+ 稀疏关键词（BM25）+ 知识图谱（NetworkX），能按概念→公式→考题的关系图做结构化推理。整个系统有 FastAPI 流式后端、Gradio 交互界面、Docker 一键部署、RAGAS 评估体系。这个项目让我完整理解了从 Naive RAG 到 Agentic RAG 的演进路径。"

## 架构总览

```
┌──────────────────────────────────────────────────────┐
│            CLI (Rich) · FastAPI · Gradio              │
├──────────────────────────────────────────────────────┤
│              LangGraph 多智能体层                       │
│                                                       │
│  ┌────────── Supervisor Agent ─────────────┐         │
│  │  分析问题 → 制定计划 → 分派任务 → 汇总     │         │
│  └──┬────────┬────────┬────────┬───────────┘         │
│     │        │        │        │                      │
│  ┌──▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼───┐ ┌──▼───┐          │
│  │检索 │ │概念 │ │计算 │ │出题  │ │审查  │          │
│  │Agent│ │Agent│ │Agent│ │Agent │ │Agent │          │
│  └──┬──┘ └─────┘ └─────┘ └──────┘ └──────┘          │
│     │                                                 │
│  ┌──▼─────────────────────────────────┐              │
│  │         共享工具层                    │              │
│  │  语义搜索 · 关键词搜索 · 图谱游走     │              │
│  │  公式查找 · 计算器 · 重排序           │              │
│  └────────────────────────────────────┘              │
├──────────────────────────────────────────────────────┤
│              三层索引体系                               │
│  稠密向量 (ChromaDB+BGE-M3)                            │
│  稀疏关键词 (BM25)                                      │
│  知识图谱 (NetworkX — 概念↔公式↔课件↔考题)             │
├──────────────────────────────────────────────────────┤
│      基础设施: Docker · LangSmith · RAGAS              │
└──────────────────────────────────────────────────────┘
```

## 数据处理管线

**738 个索引块**，横跨 5 种异构数据源，每种源有自己的分块策略：

| 数据源 | 规模 | Chunk 数 | 分块策略 |
|--------|------|---------|---------|
| 课件 | 21 PDF, 582页 | 200 | 3页合并窗口 |
| 历年试卷 | 9年 (2014-2025) | 25 | 按题号边界切分 |
| 参考答案 | 3年 (2022-2025) | 15 | 子题成对切分 |
| 研讨题解答 | 4次 Tutorial | 23 | Tutorial → Question 层级 |
| 中文复习笔记 | 12个 Markdown | 475 | Markdown 标题感知 |

每个 chunk 携带丰富元数据：来源类型、年份、课件编号、知识点标签、题型、语言。

## 快速开始

### 环境要求
- Conda (Miniforge)
- DeepSeek API Key（[点此获取](https://platform.deepseek.com)）

### 安装

```bash
# 1. 克隆
git clone https://github.com/ZSHYC/cer-rag.git
cd cer-rag

# 2. 创建环境
conda create -n cer_rag python=3.11 -y
conda activate cer_rag
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

# 4. 构建索引（一次性，约5分钟）
python rag/build_index.py

# 5. 启动 CLI
python rag/cli.py
```

### 使用演示

```
You: 板的屈曲临界应力怎么算？

Tutor: 板屈曲临界应力公式为：
σcr = (k·π²·E)/[12(1-ν²)]·(t/b)²

其中：
- k: 屈曲系数，与边界条件和长宽比 a/b 有关
- E: 弹性模量（钢取 210,000 N/mm²）
- ν: 泊松比（钢取 0.3）
- t: 板厚 (mm)
- b: 板的较短边 (mm)

[Review: 03_公式与计算题套路]
[Lecture 13, Slide 18-20]
[Exam 2023/24 Q3]
```

```
You: 给我出3道 buckling 相关的题目

Tutor: 以下是根据历年考题改编的 3 道模拟题...
```

## 技术栈

| 组件 | 选型 | 理由 |
|------|------|------|
| **LLM** | DeepSeek V4 Pro | 中英文+数学推理强，兼容 OpenAI 格式 |
| **Agent 框架** | LangGraph | 状态机编排多 Agent，条件边路由 |
| **嵌入模型** | BGE-M3 (FlagEmbedding) | 免费、本地、1024维、中英双语 |
| **向量库** | ChromaDB | 零运维、元数据过滤、Python 原生 |
| **稀疏检索** | BM25 (rank_bm25) | 公式名/术语精确匹配 |
| **重排序** | BGE-reranker-v2-m3 | Cross-encoder 精度提升 |
| **知识图谱** | NetworkX | 概念-公式-考题结构化关联 |
| **后端** | FastAPI + SSE | 异步流式响应 |
| **UI** | Gradio + Rich CLI | 快速原型 + 终端美化 |
| **评估** | RAGAS | 忠实度/相关性/精确度/召回率 |
| **部署** | Docker Compose | 一键生产部署 |
| **可观测** | LangSmith | Agent 全链路追踪 |

## 项目结构

```
rag/
├── cli.py                   # CLI 交互入口 (Rich)
├── config.py                # 全局配置
├── build_index.py           # 索引构建入口
├── ITERATION_0.md ~ ITERATION_4.md  # 5篇迭代文档
│
├── preprocess/              # 文档预处理
│   ├── extract_lectures.py  # PyMuPDF PDF 提取
│   ├── normalize_txt.py     # pdftotext 乱码修复
│   ├── chunk_lectures.py    # 课件分块
│   ├── chunk_exams.py       # 试卷/答案分块
│   ├── chunk_markdown.py    # 复习笔记分块
│   └── run_chunking.py      # 分块流水线
│
├── indexing/                # 索引层
│   ├── vector_index.py      # BGE-M3 + ChromaDB
│   ├── sparse_index.py      # BM25 稀疏索引
│   ├── build_sparse.py      # BM25 持久化
│   ├── fusion.py            # RRF 融合 + 重排序
│   └── knowledge_graph.py   # 知识图谱构建 + 游走
│
├── agents/                  # 多智能体
│   ├── state.py             # LangGraph 共享状态
│   ├── supervisor.py        # Supervisor Agent
│   ├── retrieval_agent.py   # Retrieval Agent (含自省)
│   ├── concept_agent.py     # Concept Agent
│   ├── calc_agent.py        # Calc Agent
│   ├── exam_agent.py        # Exam Agent (出题+批改)
│   ├── critic_agent.py      # Critic Agent (质量审查)
│   └── tools.py             # 共享工具层
│
├── graph/                   # LangGraph 编排
│   ├── build_graph.py       # 图编译器
│   └── router.py            # 条件路由
│
├── prompts/templates.py     # Prompt 模板管理
├── evaluation/ragas_eval.py # RAGAS 评估 + A/B 对比
├── server/main.py           # FastAPI 后端
├── ui/app.py                # Gradio Web UI
└── tests/                   # 测试
```

## 踩过的坑（重要经验的沉淀）

### 1. sentence-transformers 与 BGE-M3 不兼容

BGE-M3 是 **FlagEmbedding 家族的模型**，不能用 `sentence-transformers` 加载。新版本 tokenizer 接口不兼容，会报 `TextEncodeInput must be Union...`。

**解法**：直接用 `FlagEmbedding.BGEM3FlagModel`，自己实现 LangChain Embeddings 接口。

### 2. 幻灯片 PDF 文本提取的格式问题

`pdftotext` 对希腊字母、上下标输出乱码（`𝞮𝞮` → 应该是 `EI`）。需要建立符号映射表归一化。

### 3. 试卷格式跨年份不一致

9 年试卷，至少 3 种题目格式：`[Total marks N]`、`Question N`、`[N marks]`。正则需兼容所有年份。

### 4. 分块策略影响检索质量远超预期

同一个问题，用固定 1000-token 切分 vs 按题目边界切分，检索精度差 2-3 倍。**分块是 RAG 系统最重要的超参数**。

## 开发迭代

| 迭代 | 内容 | 状态 |
|------|------|------|
| 迭代 0 | 基础 RAG：文档处理 + 738chunk 索引 + 混合检索 + CLI | ✅ |
| 迭代 1 | Agentic 升级：LangGraph 多智能体 + Self-RAG 自省回路 | ✅ |
| 迭代 2 | 知识图谱 + Exam/Critic Agent | ✅ |
| 迭代 3 | 工程化：FastAPI + Docker + RAGAS 评估 | ✅ |
| 迭代 4 | 打磨：Gradio UI + 全面测试 + 文档 | ✅ |

---

# English {#english}

## What is this?

An **Agentic RAG** multi-agent exam tutoring system powered by **LangGraph + DeepSeek V4 Pro**. It demonstrates the complete evolution path of modern RAG systems — from naive retrieval-augmented generation through to agentic multi-agent orchestration with production-grade deployment.

Built for **SESS3026/JEIS3005 "Marine Structures"** at the University of Southampton.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-Agentic%20RAG-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/LLM-DeepSeek%20V4%20Pro-purple.svg" alt="DeepSeek">
  <img src="https://img.shields.io/badge/Embedding-BGE--M3-orange.svg" alt="BGE-M3">
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-red.svg" alt="ChromaDB">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

## Why this project matters

This is **not** a simple "plug data into LangChain" demo. It shows the **full evolution of modern RAG**:

| Stage | Architecture | Key Innovation |
|-------|-------------|---------------|
| Naive RAG | Retrieve → Generate | Dense + Sparse Hybrid, RRF Fusion |
| Advanced RAG | Retrieve → Rerank → Generate | BGE Cross-encoder, Multi-strategy Chunking |
| Agentic RAG | **Supervisor → Specialist Agents** | Self-RAG Reflection Loop, Tool-use |
| Production | FastAPI + Streaming + Docker | RAGAS Evaluation, A/B Testing, Observability |

### Key Technical Highlights

- **6-Agent Multi-Agent System**: Supervisor orchestrates Retrieval (with self-reflection), Concept, Calculation, Exam, and Critic agents via LangGraph state machine
- **Self-RAG / CRAG Pattern**: Retrieval agent evaluates sufficiency and rewrites queries for re-retrieval — inspired by Self-RAG (Asai et al., 2023) and Corrective RAG (Yan et al., 2024)
- **Three-Layer Indexing**: Dense vectors (BGE-M3) + Sparse keywords (BM25) + Knowledge Graph (NetworkX) for structured concept-formula-exam traversal
- **Cross-lingual**: Seamlessly handles English course materials and Chinese study notes in a unified semantic space (BGE-M3 multilingual embeddings)
- **Production-grade**: FastAPI + SSE streaming + Docker Compose + LangSmith observability
- **Evaluation-driven**: RAGAS metrics (faithfulness, context precision/recall, answer relevancy) with A/B strategy comparison framework

### Interview pitch

> "I built a multi-agent AI exam tutoring system. It's not just a basic RAG — I used LangGraph to build a Supervisor agent that coordinates 5 specialist agents (retrieval with self-reflection, concept explanation, calculation solving, exam generation/grading, and quality critique). The retrieval agent has a Self-RAG loop — after searching, it evaluates whether the information is sufficient, and if not, rewrites the query and retries. Under the hood, I have a three-layer indexing system: dense vectors (BGE-M3), sparse keywords (BM25), and a knowledge graph (NetworkX) for structured concept-formula-exam traversal. The entire system has a FastAPI streaming backend, Gradio UI, Docker deployment, and a RAGAS evaluation pipeline. This project gave me a deep understanding of the full RAG evolution path."

## Architecture

```
┌──────────────────────────────────────────────────┐
│           CLI (Rich) / FastAPI / Gradio            │
├──────────────────────────────────────────────────┤
│          LangGraph Multi-Agent Layer               │
│                                                    │
│  ┌─────────── Supervisor Agent ───────────┐       │
│  │   "Plan → Delegate → Synthesize"        │       │
│  └──┬────────┬─────────┬────────┬─────────┘       │
│     │        │         │        │                  │
│  ┌──▼──┐ ┌──▼───┐ ┌──▼──┐ ┌──▼────┐ ┌──▼────┐   │
│  │Retrv│ │Concpt│ │Calc │ │Exam   │ │Critic│    │
│  │Agent│ │Agent │ │Agent│ │Agent  │ │Agent │    │
│  └──┬──┘ └──────┘ └─────┘ └───────┘ └──────┘    │
│     │                                              │
├─────┼──────────────────────────────────────────────┤
│  ┌──▼──────────────────────────────────────────┐   │
│  │             Tool Layer                       │   │
│  │  dense_search | sparse_search | graph_trav   │   │
│  │  formula_lookup | calculator | rerank        │   │
│  └─────────────────────────────────────────────┘   │
├────────────────────────────────────────────────────┤
│           Three-Layer Indexing                      │
│   Dense (ChromaDB+BGE-M3)                           │
│   Sparse (BM25)                                     │
│   Knowledge Graph (Concept↔Formula↔Lecture↔Exam)    │
├────────────────────────────────────────────────────┤
│      Infra: Docker | LangSmith | RAGAS | CI/CD      │
└────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/ZSHYC/cer-rag.git && cd cer-rag

# Environment
conda create -n cer_rag python=3.11 -y && conda activate cer_rag
pip install -r requirements.txt

# Configure API key
cp .env.example .env  # edit with your DeepSeek API key

# Build index (one-time, ~5 min)
python rag/build_index.py

# Launch
python rag/cli.py            # CLI with Rich
python rag/ui/app.py         # Gradio Web UI
python -m rag.server.main    # FastAPI server
docker-compose up -d         # Docker deployment
```

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| **LLM** | DeepSeek V4 Pro | Strong Chinese+math reasoning, OpenAI-compatible API |
| **Agent Framework** | LangGraph | State machine orchestration with conditional routing |
| **Embedding** | BGE-M3 (FlagEmbedding) | Multilingual, 1024-dim, local & free |
| **Vector DB** | ChromaDB | Zero-ops, metadata filtering, Python-native |
| **Sparse Search** | BM25 (rank_bm25) | Exact keyword matching for formulas/terms |
| **Reranker** | BGE-reranker-v2-m3 | Cross-encoder precision boost |
| **Knowledge Graph** | NetworkX | Lightweight graph traversal |
| **Backend** | FastAPI + SSE | Async streaming responses |
| **UI** | Gradio + Rich CLI | Rapid prototyping + polished terminal |
| **Evaluation** | RAGAS | Faithfulness, relevancy, precision/recall |
| **Deploy** | Docker Compose | One-command production |

## Key Design Decisions

| Decision | Why |
|----------|-----|
| FlagEmbedding over sentence-transformers | BGE-M3 is a FlagEmbedding model; tokenizer incompatibility with sentence-transformers |
| Source-aware chunking over fixed-size | Critical for retrieval quality — different sources need different granularities |
| ChromaDB over FAISS | Metadata filtering required for source-type scoped queries |
| LangGraph over LangChain LCEL | Fine-grained control over multi-agent state management and conditional routing |
| Unicode math over LaTeX | BGE-M3 embeds Unicode well; LLMs read it natively without rendering overhead |
| DeepSeek V4 Pro over GPT-4o | Superior Chinese quality, lower cost, equally strong math reasoning |

## Lessons Learned

1. **sentence-transformers ≠ BGE-M3**: BGE-M3 belongs to the FlagEmbedding family. Loading it via sentence-transformers triggers `TextEncodeInput` errors. Use `FlagEmbedding.BGEM3FlagModel` directly.
2. **Chunking is the most important hyperparameter**: Switching from fixed-size to source-aware chunking improved retrieval precision by 2-3x.
3. **Self-reflection drives retrieval quality**: A simple "is this sufficient?" check with re-query dramatically reduces "I don't know" answers.
4. **PDF slide extraction isn't trivial**: Greek letters and subscripts from `pdftotext` require normalization mappings. Diagrams in slides are lost — cite slide numbers instead.

## Development Roadmap

| Iteration | Focus | Status |
|-----------|-------|--------|
| 0 | Basic RAG: doc processing, hybrid retrieval, CLI | ✅ |
| 1 | Agentic upgrade: LangGraph multi-agent, Self-RAG | ✅ |
| 2 | Knowledge Graph + Exam/Critic agents | ✅ |
| 3 | Productionization: FastAPI, Docker, RAGAS | ✅ |
| 4 | Polish: Gradio UI, testing, docs | ✅ |

## License

MIT License — feel free to use and adapt.

---

<p align="center">
  <sub>Built with LangGraph · DeepSeek V4 Pro · ChromaDB · BGE-M3 · NetworkX</sub>
</p>
