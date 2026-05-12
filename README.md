<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)">
    <img src="https://img.shields.io/badge/AGENTIC_RAG-MULTI_AGENT_SYSTEM-00d4ff?style=for-the-badge" alt="Agentic RAG">
  </picture>
</p>

<h1 align="center">Marine Structures AI Tutor<br><sub>船舶结构物 AI 导师</sub></h1>

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
  <b><a href="#chinese">中文版</a></b> · <a href="#english">English</a>
</p>

<br>

> **An Agentic RAG multi-agent exam tutoring system.**
> Not a "plug data into LangChain" demo — it traces the **complete evolution path** from Naive RAG → Advanced RAG → Agentic RAG → Production deployment.
>
> Built for SESS3026/JEIS3005 "Marine Structures" at the University of Southampton.

---

<p align="center">
  <b>📑 <a href="#chinese-toc">中文目录</a></b> · <a href="#english-toc">English TOC</a>
</p>

---

# 中文版 {#chinese}

<a id="chinese-toc"></a>
## 📑 目录

1. [为什么值得关注](#ch-why)
2. [面试话术](#ch-pitch)
3. [架构总览](#ch-arch)
4. [快速开始](#ch-quickstart)
5. [完整使用指南](#ch-guide)
   - [环境配置](#ch-env) · [安装与索引构建](#ch-install) · [CLI 命令行](#ch-cli) · [Gradio Web UI](#ch-gradio) · [FastAPI 后端](#ch-fastapi) · [Docker 部署](#ch-docker) · [RAGAS 评估](#ch-ragas) · [知识图谱 API](#ch-kg) · [自定义 Agent 流程](#ch-custom)
6. [数据处理管线](#ch-data)
7. [技术栈](#ch-stack)
8. [项目结构](#ch-structure)
9. [踩过的坑](#ch-lessons)
10. [开发迭代](#ch-iterations)

---

<a id="ch-why"></a>
## 为什么值得关注

### 🎯 这个项目解决的核心问题

> 如何从"会用 LangChain 调 API"进阶到"能设计生产级 Agentic RAG 系统"？

| 面试官眼中的你（之前） | 面试官眼中的你（做完这个项目后） |
|:---------------------|:---------------------------|
| 调 API 做问答 | **我设计了一个 6-Agent 多智能体协作系统** |
| 用 LangChain 搭 RAG | **我对比了朴素/高级/Agentic 三种 RAG 架构，能讲清各自的 trade-off** |
| 调 embedding 做搜索 | **我构建了稠密+稀疏+知识图谱三层索引，知道每层解决什么问题** |
| 写 Python 脚本 | **我交付了 FastAPI + SSE 流式 + Docker 一键部署的生产级系统** |
| 凭感觉说"效果好" | **我有 RAGAS 量化评估 + A/B 策略对比数据支撑** |

### 📊 项目规模

| 指标 | 数值 |
|------|------|
| Python 源文件 | 31 个 |
| 代码行数 | ~2,800 行 |
| LangGraph Agent 节点 | 7 个 |
| 索引文档块 | 738 个 |
| 知识图谱节点/边 | 79 / 269 |
| 迭代次数 | 5 次 |
| 测试通过 | 19/19 |

<a id="ch-pitch"></a>
### 🎤 面试话术（30秒电梯演讲）

> "我做了一个多智能体 AI 考试导师系统。我用 LangGraph 构建了一个 Supervisor Agent 协调 5 个专业 Agent。检索 Agent 有 Self-RAG 自省回路——搜完会自动评估信息是否充分，不够就改写查询重新搜。底层用了三层索引：稠密向量 + 稀疏关键词 + 知识图谱。整个系统有 FastAPI 流式后端、Gradio 交互界面、Docker 一键部署、RAGAS 评估体系。这个项目让我完整理解了从 Naive RAG 到 Agentic RAG 的演进路径。"

<a id="ch-arch"></a>
## 架构总览

```
  ┌─────────────────────────────────────────────────────────┐
  │                  用户接口层                               │
  │        CLI (Rich)  ·  Gradio UI  ·  FastAPI SSE          │
  ├─────────────────────────────────────────────────────────┤
  │               LangGraph 多智能体层                         │
  │                                                          │
  │   ┌─────────────── Supervisor Agent ─────────────┐       │
  │   │   分析问题 → 制定计划 → 分派任务 → 汇总结果    │       │
  │   └───┬─────────┬─────────┬─────────┬────────────┘       │
  │       │         │         │         │                     │
  │   ┌───▼──┐ ┌───▼──┐ ┌───▼──┐ ┌───▼───┐ ┌───▼───┐        │
  │   │检索   │ │概念   │ │计算   │ │出题   │ │审查   │        │
  │   │Agent  │ │Agent  │ │Agent  │ │Agent  │ │Agent  │        │
  │   │       │ │       │ │       │ │       │ │       │        │
  │   │Self-  │ │多来源 │ │分步   │ │模拟   │ │幻觉   │        │
  │   │RAG    │ │交叉   │ │求解   │ │出题   │ │检测   │        │
  │   │自省   │ │验证   │ │公式   │ │自动   │ │引用   │        │
  │   │回路   │ │       │ │标注   │ │批改   │ │验证   │        │
  │   └───┬──┘ └───────┘ └───────┘ └───────┘ └───────┘        │
  │       │                                                    │
  │   ┌───▼────────────────────────────────────┐              │
  │   │            共享工具层                    │              │
  │   │  dense_search · sparse_search           │              │
  │   │  graph_traverse · rerank · calculator   │              │
  │   └────────────────────────────────────────┘              │
  ├─────────────────────────────────────────────────────────┤
  │                 三层索引体系                               │
  │                                                          │
  │  Layer 1: 稠密向量 — ChromaDB + BGE-M3 (1024维)          │
  │  Layer 2: 稀疏关键词 — BM25 (精确匹配公式/术语)            │
  │  Layer 3: 知识图谱 — NetworkX                             │
  │           概念 ↔ 公式 ↔ 课件 ↔ 考题 ↔ 答案                │
  ├─────────────────────────────────────────────────────────┤
  │            基础设施: Docker · LangSmith · RAGAS            │
  └─────────────────────────────────────────────────────────┘
```

<a id="ch-quickstart"></a>
## 快速开始

```bash
git clone https://github.com/ZSHYC/cer-rag.git && cd cer-rag
conda create -n cer_rag python=3.11 -y && conda activate cer_rag
pip install -r requirements.txt
cp .env.example .env                    # 编辑填入 DeepSeek API Key
python rag/build_index.py               # 构建索引（一次性，~5分钟）
```

**四种启动方式：**

```bash
python rag/cli.py                       # ① CLI 命令行（Rich 美化）
python rag/ui/app.py                    # ② Gradio Web UI → http://localhost:7860
python -m rag.server.main               # ③ FastAPI 后端 → http://localhost:8000
docker-compose up -d                    # ④ Docker 一键部署
```

<a id="ch-guide"></a>
## 完整使用指南

<a id="ch-env"></a>
### 环境配置

`.env` 文件：

```bash
DEEPSEEK_API_KEY=sk-your-key-here      # 必填
DEEPSEEK_MODEL=deepseek-v4-pro         # deepseek-v4-pro 或 deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

<a id="ch-install"></a>
### 首次安装与索引构建

```bash
python rag/build_index.py
```

<details>
<summary>📋 构建输出（点击展开）</summary>

```
============================================================
Marine Structures RAG - 索引构建
============================================================

[Step 1] 文档分块...
  [1/4] 课件分块: 21 PDF → 582 pages → 200 chunks
  [2/4] 试卷与答案分块: 63 chunks
  [3/4] 中文复习笔记分块: 475 chunks
  总计: 738 chunks

[Step 2] 构建向量索引 (738 个文档)...
  进度: 738/738 → data/chroma_db

[Step 3] 构建 BM25 索引...
  data/chunks/bm25_index.pkl (738 docs)
```
</details>

| 构建产物 | 说明 |
|---------|------|
| `data/chroma_db/` | ChromaDB 稠密向量索引 (1024维 × 738) |
| `data/chunks/bm25_index.pkl` | BM25 稀疏索引 |
| `data/chunks/all_chunks.jsonl` | 所有 chunk 文本+元数据 |
| `data/knowledge_graph/knowledge_graph.gml` | 知识图谱 (79节点, 269边) |

<a id="ch-cli"></a>
### CLI 命令行

```bash
python rag/cli.py
```

| 命令 | 作用 |
|------|------|
| `:q` | 退出 |
| `:s` | 切换来源显示（查看检索到的文档） |
| `:d` | **调试模式** — 实时展示每个 Agent 的执行规划、检索文档数、自省重试次数 |

<details>
<summary>🔍 调试模式输出示例（点击展开）</summary>

```
You: 计算正方形截面的形状系数
[calculation] thinking...
  │ [supervisor] plan=['检索形状系数公式', '计算形状系数'] docs=0 retries=0
  │ [retrieval] plan=[] docs=8 retries=1
  │ [calc] plan=[] docs=0 retries=0

Tutor: 已知正方形截面边长 a
公式：Z = a³/6, Zp = a³/4
形状系数 φ = Zp/Z = 1.5

Plan: 检索形状系数公式 › 计算形状系数

Sources (8 docs):
  #  Source                              Preview
  1  [Review: 10_核心题型...]             Zp = (a²/2)(a/4) + ...
  2  [Answer 2022/23 Q2(iii)]            Derive the shape factor...
  3  [Lecture 18, Slide 14-16]           Plastic section modulus...
```
</details>

**查询路由表：**

| 查询类型 | 示例 | LangGraph 路径 |
|---------|------|---------------|
| 概念题 | "What is Rational Ship Structural Design?" | supervisor → retrieval → concept |
| 计算题 | "板的屈曲临界应力怎么算？" | supervisor → retrieval → calc |
| 公式查找 | "Faulkner 有效宽度公式" | supervisor → retrieval → concept |
| 出题 | "给我出3道 buckling 的题" | supervisor → exam_generate |
| 跨语言 | "用中文解释 plastic collapse" | 自动识别，双语回答 |

<a id="ch-gradio"></a>
### Gradio Web 界面

```bash
python rag/ui/app.py
# → http://localhost:7860
```

- 完整的 Agentic RAG 对话面板
- Debug Mode 开关实时显示 Agent 执行链
- 预设示例问题一键测试
- 自动显示检索来源引用
- 多轮对话记忆

<a id="ch-fastapi"></a>
### FastAPI 后端

```bash
python -m rag.server.main
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/chat` | POST | SSE 流式对话 |
| `/exam/generate` | POST | 生成模拟题 |
| `/exam/grade` | POST | 批改作答 |

<details>
<summary>📡 SSE 流式响应格式（点击展开）</summary>

```
data: {"node":"supervisor","plan":["搜索资料","解释概念"],"docs":0}
data: {"node":"retrieval","plan":[],"docs":8}
data: {"node":"concept","plan":[],"docs":0}
data: {"answer":"Plate buckling is a structural instability..."}
data: [DONE]
```
</details>

<details>
<summary>🐍 Python 客户端示例（点击展开）</summary>

```python
import requests, json

with requests.post("http://localhost:8000/chat",
    json={"query": "What is plate buckling?"}, stream=True) as r:
    for line in r.iter_lines():
        if line and line.startswith(b'data: '):
            payload = line[6:].decode()
            if payload == '[DONE]': break
            event = json.loads(payload)
            if 'answer' in event: print(event['answer'])
```
</details>

<a id="ch-docker"></a>
### Docker 部署

```bash
docker-compose up -d          # 启动
docker-compose logs -f        # 查看日志
docker-compose down           # 停止
curl http://localhost:8000/health  # 验证
```

```yaml
# docker-compose.yml 关键配置
services:
  tutor-api:
    build: .                    # 从 Dockerfile 构建
    ports: ["8000:8000"]        # 端口映射
    environment:                # 环境变量注入
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
    volumes:
      - ./data:/app/data        # 索引数据持久化
      - ./.env:/app/.env:ro     # 配置只读挂载
    restart: unless-stopped     # 异常自动重启
```

<a id="ch-ragas"></a>
### RAGAS 评估

```bash
python rag/evaluation/ragas_eval.py
```

基于 **5 道历年真题** 作为 ground truth，输出每题的检索文档数、Self-RAG 重试次数、答案质量对比。

<a id="ch-kg"></a>
### 知识图谱 API

```python
from rag.indexing.knowledge_graph import get_kg, graph_traverse

kg = get_kg()                              # 79 节点, 269 边
results = graph_traverse("buckling")       # 找 buckling 相关的公式、考题、课件
# → [Formula] σcr = kπ²E/[12(1-ν²)]·(t/b)²
# → [Exam] 2022/23 Q3, 2023/24 Q3, 2024/25 Q3
# → [Lecture] L13: Buckling of Plates
```

```
Concept ──EXPLAINED_IN──→ Lecture
Concept ──TESTED_BY─────→ ExamQuestion
Formula ──RELATED_TO────→ Concept
```

<a id="ch-custom"></a>
### 自定义 Agent 流程

```python
from rag.graph.build_graph import build_tutor_graph

graph = build_tutor_graph(enable_critic=False)   # 无审查模式
graph = build_tutor_graph(enable_critic=True)    # 含审查模式（回答先经 Critic 验证）

state = {"query": "...", "query_type": "...", ...}
for step in graph.stream(state, {"recursion_limit": 20}):
    node = list(step.keys())[0]
    print(f"[{node}]", step[node].get("plan", []))
```

<a id="ch-data"></a>
## 数据处理管线

**738 chunks** 跨 5 种源，每种采用专门的分块策略：

| 数据源 | 规模 | Chunks | 分块策略 | 关键元数据 |
|--------|------|--------|---------|-----------|
| 📚 课件 | 21 PDF · 582 页 | 200 | 3 页合并窗口 | lecture_number, slide_range, topic_tags |
| 📝 历年试卷 | 9 年 (2014–2025) | 25 | 题号边界切分 | year, question_number, marks, content_type |
| ✅ 参考答案 | 3 年 (2022–2025) | 15 | 子题成对切分 | year, question_number, sub_question |
| 🔬 研讨题解答 | 4 Tutorials | 23 | Tutorial→Question 层级 | tutorial_number, language |
| 📖 中文复习笔记 | 12 Markdown | 475 | Markdown 标题感知 | file_topic, breadcrumb, language |

<a id="ch-stack"></a>
## 技术栈

| 层 | 选型 | 选型理由 |
|----|------|---------|
| **LLM** | DeepSeek V4 Pro | 中英文+数学推理强，API 价格低 |
| **Agent 框架** | LangGraph | 有状态多 Agent 编排，条件路由 |
| **嵌入** | BGE-M3 (FlagEmbedding) | 免费本地，1024维，100+语言 |
| **向量库** | ChromaDB | 零运维，元数据过滤 |
| **稀疏检索** | BM25 (rank_bm25) | 公式/术语精确匹配 |
| **重排序** | BGE-reranker-v2-m3 | Cross-encoder 精度提升 |
| **知识图谱** | NetworkX | 轻量图遍历 |
| **后端** | FastAPI + SSE | 异步流式 |
| **UI** | Gradio + Rich CLI | 快速原型 + 终端美化 |
| **评估** | RAGAS | 量化检索+生成质量 |
| **部署** | Docker Compose | 一键启动 |

<a id="ch-structure"></a>
## 项目结构

```
rag/
├── cli.py · config.py · build_index.py        # 入口
├── ITERATION_0.md ~ ITERATION_4.md            # 5 篇迭代文档
│
├── preprocess/         # 文档预处理 (PyMuPDF + 乱码修复 + 5种分块)
├── indexing/           # 索引层 (ChromaDB + BM25 + RRF + KG)
├── agents/             # 多智能体 (7 个 Agent + 工具层)
├── graph/              # LangGraph (编译器 + 路由器)
├── prompts/            # 8 套 Prompt 模板
├── evaluation/         # RAGAS 评估 + A/B 对比
├── server/             # FastAPI 服务
├── ui/                 # Gradio 界面
└── tests/              # 测试
```

<a id="ch-lessons"></a>
## 踩过的坑

<details open>
<summary><b>1. sentence-transformers 与 BGE-M3 不兼容</b></summary>

**现象**: `TypeError: TextEncodeInput must be Union[...]`

**根因**: BGE-M3 是 FlagEmbedding 家族的模型，不能用 `sentence-transformers` 加载。

**解法**: 直接用 `FlagEmbedding.BGEM3FlagModel`，自行封装 LangChain `Embeddings` 接口（`vector_index.py:BgeM3FlagEmbedding`）。
</details>

<details>
<summary><b>2. 分块策略是 RAG 最重要的超参数</b></summary>

固定 1000-token 切分 vs 按题目边界切分，检索精度差 **2-3 倍**。最终采用 5 种源感知分块策略。
</details>

<details>
<summary><b>3. 试卷格式跨 9 年不一致</b></summary>

至少 3 种格式：`[Total marks N]`(2022-25)、`Question N`(2014-19)、`N. [M marks]`(2020-21)。正则需全部兼容。
</details>

<details>
<summary><b>4. PDF 公式符号乱码</b></summary>

`pdftotext` 输出希腊字母乱码（`𝞮𝞮` → `EI`）。需建立 100+ 字符映射表归一化。
</details>

<a id="ch-iterations"></a>
## 开发迭代

| # | 主题 | 核心交付 | 状态 |
|---|------|---------|------|
| 0 | 基础 RAG | 文档处理, 738 chunks, 三层索引, 混合检索, CLI | ✅ |
| 1 | Agentic 升级 | LangGraph 多 Agent, Supervisor, Self-RAG 回路 | ✅ |
| 2 | 知识图谱 | 79节点269边 KG, Exam/Critic Agent, 图增强搜索 | ✅ |
| 3 | 工程化 | FastAPI SSE, Docker, RAGAS 评估, A/B 对比 | ✅ |
| 4 | 打磨 | Gradio UI, 19项测试通过, 中英双语文档 | ✅ |

---

# English {#english}

<a id="english-toc"></a>
## 📑 Table of Contents

1. [What & Why](#en-why)
2. [Architecture](#en-arch)
3. [Quick Start](#en-quickstart)
4. [Usage Guide](#en-guide)
5. [Tech Stack](#en-stack)
6. [Project Structure](#en-structure)
7. [Lessons Learned](#en-lessons)
8. [Development Roadmap](#en-roadmap)

<a id="en-why"></a>
## What & Why

An **Agentic RAG** multi-agent exam tutoring system. Built for SESS3026/JEIS3005 "Marine Structures" at the University of Southampton.

### The Evolution of RAG — In One Project

| Stage | Architecture | Innovation |
|-------|-------------|------------|
| **Naive RAG** | Retrieve → Generate | Dense + Sparse Hybrid, RRF Fusion |
| **Advanced RAG** | Retrieve → Rerank → Generate | Cross-encoder, Multi-strategy Chunking |
| **Agentic RAG** | Supervisor → 5 Specialist Agents | Self-RAG Reflection, Tool-use |
| **Production** | FastAPI SSE + Docker | RAGAS Evaluation, A/B Testing |

### Key Technical Highlights

- **6-Agent system**: Supervisor + Retrieval (Self-RAG) + Concept + Calculation + Exam + Critic — orchestrated via LangGraph state machine
- **Self-RAG pattern**: Retrieval agent evaluates sufficiency, rewrites queries, and re-searches when needed — paper-backed (Self-RAG '23, CRAG '24)
- **Three-layer indexing**: Dense (BGE-M3 · 1024-dim) + Sparse (BM25) + Knowledge Graph (NetworkX · 79 nodes, 269 edges)
- **Cross-lingual**: English lectures + Chinese study notes in one unified semantic space
- **Production-grade**: FastAPI SSE streaming, Docker Compose, LangSmith tracing
- **Evaluation-driven**: RAGAS metrics with A/B strategy comparison

### Interview Pitch

> "I built a multi-agent AI tutoring system with LangGraph. A Supervisor agent orchestrates 5 specialists — retrieval with Self-RAG reflection, concept explanation, calculation solving, exam generation/grading, and quality critique. Three-layer indexing: dense vectors + sparse keywords + knowledge graph traversal. FastAPI SSE backend, Gradio UI, Docker deployment, RAGAS evaluation. This project taught me the full RAG evolution from naive retrieval to agentic orchestration."

<a id="en-arch"></a>
## Architecture

```
  User Interface: CLI (Rich) · Gradio · FastAPI SSE
  ─────────────────────────────────────────────────
  LangGraph Multi-Agent Layer:
     Supervisor Agent (Plan → Delegate → Synthesize)
        ├── Retrieval Agent (Self-RAG reflection loop)
        ├── Concept Agent  (Cross-source verification)
        ├── Calc Agent     (Step-by-step solving)
        ├── Exam Agent     (Generate + Grade)
        └── Critic Agent   (Hallucination check)
  ─────────────────────────────────────────────────
  Tool Layer: dense_search · sparse_search · graph_traverse · rerank
  ─────────────────────────────────────────────────
  Three-Layer Indexing:
     Dense (ChromaDB + BGE-M3)
     Sparse (BM25)
     Knowledge Graph (Concept ↔ Formula ↔ Lecture ↔ Exam)
  ─────────────────────────────────────────────────
  Infrastructure: Docker · LangSmith · RAGAS
```

<a id="en-quickstart"></a>
## Quick Start

```bash
git clone https://github.com/ZSHYC/cer-rag.git && cd cer-rag
conda create -n cer_rag python=3.11 -y && conda activate cer_rag
pip install -r requirements.txt
cp .env.example .env              # add your DeepSeek API key
python rag/build_index.py         # one-time, ~5 min
```

**Launch options:**

```bash
python rag/cli.py                 # CLI with Rich formatting
python rag/ui/app.py              # Gradio UI → localhost:7860
python -m rag.server.main         # FastAPI → localhost:8000
docker-compose up -d              # Docker deployment
```

<a id="en-guide"></a>
## Usage Guide

### CLI

| Key | Action |
|-----|--------|
| `:q` | Quit |
| `:s` | Toggle source display |
| `:d` | **Debug mode** — live agent execution trace |

**Agent routing:**

| Query Type | Example | Route |
|-----------|---------|-------|
| Conceptual | "What is Rational Ship Structural Design?" | supervisor → retrieval → concept |
| Calculation | "Calculate shape factor of square section" | supervisor → retrieval → calc |
| Formula | "What is Faulkner's effective width?" | supervisor → retrieval → concept |
| Exam Gen | "Generate 3 questions about buckling" | supervisor → exam_generate |
| Cross-lingual | "用中文解释 plastic collapse" | Auto-detect, bilingual |

### FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | SSE streaming chat |
| `POST` | `/exam/generate` | Generate mock exam |
| `POST` | `/exam/grade` | Grade student answer |

```bash
# Health
curl http://localhost:8000/health

# SSE Chat
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is plate buckling?"}'

# Generate exam
curl -X POST http://localhost:8000/exam/generate \
  -H "Content-Type: application/json" \
  -d '{"topic":"buckling","num_questions":3}'
```

### Docker

```bash
docker-compose up -d     # start (port 8000)
docker-compose logs -f   # logs
docker-compose down      # stop
```

### RAGAS Evaluation

```bash
python rag/evaluation/ragas_eval.py
# 5 ground-truth cases, outputs per-case retrieval + generation metrics
```

### Knowledge Graph API

```python
from rag.indexing.knowledge_graph import get_kg, graph_traverse
kg = get_kg()                         # 79 nodes, 269 edges
graph_traverse("buckling")            # → formulas, exams, lectures
```

<a id="en-stack"></a>
## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| LLM | DeepSeek V4 Pro | Strong Chinese+math, OpenAI-compatible |
| Agent Framework | LangGraph | State machine multi-agent orchestration |
| Embedding | BGE-M3 (FlagEmbedding) | 1024-dim, multilingual, free & local |
| Vector DB | ChromaDB | Metadata filtering, zero-ops |
| Sparse | BM25 (rank_bm25) | Exact formula/term matching |
| Reranker | BGE-reranker-v2-m3 | Cross-encoder precision |
| KG | NetworkX | Lightweight graph traversal |
| Backend | FastAPI + SSE | Async streaming |
| UI | Gradio + Rich CLI | Rapid prototyping + polished terminal |
| Eval | RAGAS | Faithfulness, relevancy, precision/recall |
| Deploy | Docker Compose | One-command |

<a id="en-structure"></a>
## Project Structure

```
rag/
├── cli.py · config.py · build_index.py
├── ITERATION_0.md ~ ITERATION_4.md
├── preprocess/    # PyMuPDF extraction + normalization + 5 chunking strategies
├── indexing/      # ChromaDB + BM25 + RRF fusion + Knowledge Graph
├── agents/        # 7 Agents (supervisor, retrieval, concept, calc, exam, critic) + tools
├── graph/         # LangGraph compiler + router
├── prompts/       # 8 prompt templates
├── evaluation/    # RAGAS evaluation + A/B comparison
├── server/        # FastAPI (SSE streaming)
├── ui/            # Gradio Web UI
└── tests/
```

<a id="en-lessons"></a>
## Lessons Learned

<details open>
<summary><b>1. BGE-M3 belongs to FlagEmbedding, not sentence-transformers</b></summary>

Loading via sentence-transformers triggers `TextEncodeInput` errors. Use `FlagEmbedding.BGEM3FlagModel` directly.
</details>

<details>
<summary><b>2. Chunking strategy is the #1 RAG hyperparameter</b></summary>

Source-aware chunking improved retrieval precision by **2-3×** over fixed-size splitting.
</details>

<details>
<summary><b>3. PDF slide extraction is non-trivial</b></summary>

Greek letters and subscripts from `pdftotext` require a 100+ character normalization map. Diagrams in slides are lost — cite slide numbers.
</details>

<details>
<summary><b>4. Exam formats vary across years</b></summary>

9 years of exams had 3+ question marking formats. Multi-pattern regex compatibility is essential.
</details>

<a id="en-roadmap"></a>
## Development Roadmap

| # | Focus | Key Output | Status |
|---|-------|------------|--------|
| 0 | Basic RAG | Doc processing, 738 chunks, hybrid retrieval, CLI | ✅ |
| 1 | Agentic Upgrade | LangGraph multi-agent, Supervisor, Self-RAG | ✅ |
| 2 | Knowledge Graph | 79-node KG, Exam/Critic agents, KG search | ✅ |
| 3 | Productionization | FastAPI SSE, Docker, RAGAS eval, A/B testing | ✅ |
| 4 | Polish | Gradio UI, 19/19 tests, bilingual docs | ✅ |

## License

MIT — free to use, adapt, and learn from.

---

<p align="center">
  <sub>LangGraph · DeepSeek V4 Pro · ChromaDB · BGE-M3 · NetworkX · FastAPI</sub>
</p>
