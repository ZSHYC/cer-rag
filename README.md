# Marine Structures AI Tutor · 船舶结构物 AI 导师

<p align="center">
  <b><a href="#chinese">中文版</a></b> · <a href="#english">English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-Agentic%20RAG-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/LLM-DeepSeek%20V4%20Pro-purple.svg" alt="DeepSeek">
  <img src="https://img.shields.io/badge/Embedding-BGE--M3-orange.svg" alt="BGE-M3">
  <img src="https://img.shields.io/badge/Vector%20DB-ChromaDB-red.svg" alt="ChromaDB">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

---

# 中文版 {#chinese}

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
│     │         ↑ 检索不充分时自省重试 (Self-RAG)        │
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

---

## 完整使用指南

### 零、环境配置

```bash
# .env 文件
DEEPSEEK_API_KEY=sk-your-key-here          # 必填
DEEPSEEK_MODEL=deepseek-v4-pro             # deepseek-v4-pro 或 deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

### 一、首次安装与索引构建

```bash
# 1. 克隆仓库
git clone https://github.com/ZSHYC/cer-rag.git
cd cer-rag

# 2. 创建 conda 环境
conda create -n cer_rag python=3.11 -y
conda activate cer_rag
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key

# 4. 构建索引（首次运行，约5分钟）
python rag/build_index.py
```

`build_index.py` 的输出：
```
============================================================
Marine Structures RAG - 索引构建
============================================================

[Step 1] 文档分块...
  [1/4] 课件分块: 21 PDF → 582 pages → 200 chunks
  [2/4] 试卷与答案分块: 63 chunks
  [3/4] 中文复习笔记分块: 475 chunks
  [4/4] 元数据构建
分块完成! 总计: 738 chunks
已保存至: data/chunks/all_chunks.jsonl

[Step 2] 构建向量索引 (738 个文档)...
  进度: 16/738 → 32/738 → ... → 738/738
向量索引构建完成，存储于: data/chroma_db

[Step 3] 构建 BM25 索引...
BM25 index saved to data/chunks/bm25_index.pkl (738 docs)
```

构建完成后，以下数据目录被创建：

| 文件/目录 | 说明 |
|-----------|------|
| `data/chroma_db/` | ChromaDB 稠密向量索引 |
| `data/chunks/bm25_index.pkl` | BM25 稀疏索引（pickle） |
| `data/chunks/all_chunks.jsonl` | 所有 chunk 文本+元数据（可人工查看） |
| `data/knowledge_graph/knowledge_graph.gml` | 知识图谱（79节点, 269边） |
| `data/metadata.json` | 课件主题映射 + 考题知识点映射 |

### 二、CLI 命令行（Rich 美化终端）

```bash
conda activate cer_rag
cd /home/zshyc/cer
python rag/cli.py
```

**启动画面：**
```
╭──────────────── Welcome ─────────────────╮
│ Marine Structures AI Tutor               │
│ SESS3026/JEIS3005 — Agentic RAG 考试导师 │
│                                          │
│ :q 退出  :s 切换来源显示  :d 调试模式     │
╰──────────────────────────────────────────╯
加载向量索引...
  ChromaDB: 738 条
加载 BM25 索引...
  BM25: 738 篇
加载 LLM...
  Model: deepseek-v4-pro
加载 Reranker...
  Reranker: OK

Ready!
```

**CLI 命令：**

| 命令 | 功能 |
|------|------|
| `:q` | 退出程序 |
| `:s` | 切换来源显示（回答后显示/隐藏检索到的文档来源表格） |
| `:d` | 切换调试模式（实时显示每个 Agent 节点的执行计划、文档数、重试次数） |

**调试模式（`:d` 开启）输出示例：**

```
You: 计算正方形截面的形状系数
[calculation] thinking...
  │ [supervisor] plan=['检索形状系数公式和相关信息', '计算形状系数'] docs=0 retries=0
  │ [retrieval] plan=[] docs=8 retries=1
  │ [calc] plan=[] docs=0 retries=0

Tutor: 已知正方形截面边长 a
公式：Z = a³/6, Zp = a³/4
形状系数 φ = Zp/Z = 1.5
...

Plan: 检索形状系数公式和相关信息 › 计算形状系数

Sources (8 docs):
  #  Source                              Preview
  1  [Review: 10_核心题型...]             Zp = (a²/2)(a/4) + (a²/2)(a/4) = a³/4...
  2  [Answer 2022/23 Q2(iii)]            Derive the shape factor for a square...
  3  [Lecture 18, Slide 14-16]           Plastic section modulus and shape factor...
```

**支持的查询类型与路由逻辑：**

| 查询类型 | 示例 | Agent 路由 |
|---------|------|-----------|
| 概念题 | "What is Rational Ship Structural Design?" | Supervisor → Retrieval → Concept Agent |
| 计算题 | "板的屈曲临界应力怎么算？" | Supervisor → Retrieval → Calc Agent |
| 公式查找 | "Faulkner 有效宽度公式是什么？" | Supervisor → Retrieval → Concept Agent |
| 出题 | "给我出3道 buckling 的题" | Supervisor → Exam Agent |
| 参考查询 | "2023年考过 buckling 吗？" | Supervisor → Retrieval → Concept Agent |
| 中英混合 | "Explain shape factor in Chinese" | 自动识别，双语回答 |

**对话示例：**

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

来源：
  [Review: 03_公式与计算题套路]
  [Lecture 13, Slide 18-20]
  [Exam 2023/24 Q3]
```

```
You: 给我出3道 buckling 相关的题目

Tutor: 以下是根据历年考题改编的 3 道模拟题：

### Q1. [Total 25 marks]
(i) Explain the phenomenon of plate buckling...
(ii) A deck plate of 10mm thickness...

### Q2. [Total 25 marks]
...
```

### 三、Gradio Web 界面

```bash
conda activate cer_rag
cd /home/zshyc/cer
python rag/ui/app.py
# 打开浏览器访问 http://localhost:7860
```

界面功能：
- **对话面板**：和 CLI 一样使用完整的 Agentic RAG 流程（Supervisor → Retrieval → Specialist Agent）
- **调试开关**：勾选 "Debug Mode" 后实时显示每个 Agent 节点的执行状态
- **示例问题**：右侧预设了 4 个典型问题，点击可快速测试
- **来源引用**：每条回答末尾自动附带检索到的文档来源列表
- **对话历史**：保留完整的多轮对话上下文，支持追问

### 四、FastAPI 后端服务

```bash
conda activate cer_rag
cd /home/zshyc/cer
python -m rag.server.main

# 服务运行在 http://localhost:8000
# 交互式 API 文档: http://localhost:8000/docs
# ReDoc 文档: http://localhost:8000/redoc
```

#### `GET /health` — 健康检查

```bash
curl http://localhost:8000/health
# 响应: {"status":"ok"}
```

#### `POST /chat` — 对话接口（SSE 流式返回）

这是核心对话接口。使用 Server-Sent Events (SSE) 流式推送每个 Agent 节点的执行状态，最后推送完整答案。

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is plate buckling?",
    "chat_history": []
  }'
```

**SSE 事件流格式（测试时能看到每个 Agent 的执行过程）：**

```
data: {"node": "supervisor", "plan": ["搜索plate buckling资料", "解释概念"], "docs": 0}
data: {"node": "retrieval", "plan": [], "docs": 8}
data: {"node": "concept", "plan": [], "docs": 0}
data: {"answer": "Plate buckling is a structural instability phenomenon that occurs when..."}
data: [DONE]
```

- `node`: 当前执行的 Agent 名称（`supervisor` / `retrieval` / `concept` / `calc`）
- `plan`: Supervisor 制定的执行计划
- `docs`: 检索到的文档数量
- `answer`: Concept Agent 或 Calc Agent 生成的最终答案
- `[DONE]`: 流结束信号

```python
# Python 客户端示例
import requests, json

url = "http://localhost:8000/chat"
data = {"query": "What is Rational Ship Structural Design?"}

with requests.post(url, json=data, stream=True) as r:
    for line in r.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                payload = line[6:]
                if payload == '[DONE]':
                    break
                event = json.loads(payload)
                if 'answer' in event:
                    print(event['answer'])
```

#### `POST /exam/generate` — 出题接口

根据指定知识点，检索历年相关考题，改编数值生成模拟题。

```bash
curl -X POST http://localhost:8000/exam/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "buckling",
    "num_questions": 3
  }'
```

响应：
```json
{
  "answer": "## 模拟试题 — Buckling\n\n### Q1. [Total 25 marks]\n(i) Explain the difference between local plate buckling and overall column buckling...\n\n### Q2. [Total 25 marks]\n..."
}
```

#### `POST /exam/grade` — 批改接口

将学生作答与标准答案对比，逐步骤评分。

```bash
curl -X POST http://localhost:8000/exam/grade \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Calculate the shape factor of a square cross-section of side a",
    "student_answer": "Z = a^3/6, Zp = a^3/4, φ = Zp/Z = 1.5"
  }'
```

响应包含：
- `correctness`: correct / partially correct / incorrect
- `step_scores`: 每步得分
- `errors`: 具体错误说明
- `review_topics`: 建议复习的知识点

### 五、Docker 部署

```bash
# 确保 .env 已配置
cp .env.example .env  # 编辑填入 API Key

# 构建镜像并启动（后台运行）
docker-compose up -d

# 查看运行日志
docker-compose logs -f

# 停止服务
docker-compose down

# 验证
curl http://localhost:8000/health
```

**docker-compose.yml 详解：**

```yaml
version: '3.8'
services:
  tutor-api:
    build: .                          # 从 Dockerfile 构建
    ports:
      - "8000:8000"                   # 宿主机:容器 端口映射
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}    # 从 .env 注入
      - DEEPSEEK_MODEL=${DEEPSEEK_MODEL}
      - DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL}
    volumes:
      - ./data:/app/data              # 索引数据持久化
      - ./.env:/app/.env:ro           # 只读挂载配置
    restart: unless-stopped           # 异常自动重启
```

**Dockerfile 详解：**

```dockerfile
FROM python:3.11-slim                 # 轻量基础镜像
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag/ ./rag/
EXPOSE 8000
CMD ["python", "-m", "rag.server.main"]
```

### 六、RAGAS 评估体系

```bash
conda activate cer_rag
cd /home/zshyc/cer
python rag/evaluation/ragas_eval.py
```

评估使用 5 道历年真题作为 ground truth：

```python
TEST_CASES = [
    {"query": "What is Rational Ship Structural Design?",
     "ground_truth": "A design process directly based on structural theory...",
     "query_type": "conceptual"},
    {"query": "Calculate the shape factor for a square cross-section",
     "ground_truth": "Z = a^3/6, Zp = a^3/4, φ = 1.5",
     "query_type": "calculation"},
    {"query": "板屈曲的临界应力怎么算？",
     "ground_truth": "σcr = (k·π²·E)/[12(1-ν²)]·(t/b)²",
     "query_type": "calculation"},
    # ... 共5道
]
```

**输出示例：**

```
RAG Strategy Comparison
============================================================
Test cases: 5
Avg retrieved docs: 7.2
Avg retrieval attempts (Self-RAG): 1.2

  [1] What is Rational Ship Structural Design?...
      Docs: 8 | Retries: 1
      Answer: Rational Ship Structural Design is a design process...

  [2] Calculate the shape factor for a square cross-section...
      Docs: 8 | Retries: 1
      Answer: Z = a^3/6, Zp = a^3/4, φ = Zp/Z = 1.5...
```

### 七、知识图谱（编程接口）

```python
from rag.indexing.knowledge_graph import get_kg, graph_traverse

# 获取图（首次调用自动构建，后续从 GML 文件加载）
kg = get_kg()
print(f"节点: {kg.number_of_nodes()}, 边: {kg.number_of_edges()}")
# 输出: 节点: 79, 边: 269

# 按概念名模糊匹配 → BFS 游走
results = graph_traverse("buckling", max_depth=2)
for r in results:
    print(f"  [{r['type']}] {r['label']} (距离={r['distance']})")

# 输出示例:
#   [Formula] σcr = kπ²E/[12(1-ν²)]·(t/b)² (距离=1)
#   [ExamQuestion] Exam 2022/23 Q3 (距离=1)
#   [ExamQuestion] Exam 2023/24 Q3 (距离=1)
#   [Lecture] L13: Buckling of Plates (距离=1)
#   [Lecture] L14: Stiffened Panels (距离=1)
```

知识图谱的实体和关系：

```
概念 (Concept) ──EXPLAINED_IN──→ 课件 (Lecture)
概念 (Concept) ──TESTED_BY─────→ 考题 (ExamQuestion)
公式 (Formula) ──RELATED_TO────→ 概念 (Concept)
```

### 八、多智能体流程自定义（编程接口）

```python
from rag.graph.build_graph import build_tutor_graph, classify_query

# 构建图（不含 Critic——回答直接输出）
graph = build_tutor_graph(enable_critic=False)

# 构建图（含 Critic——所有回答先经质量审查再输出）
graph = build_tutor_graph(enable_critic=True)

# 执行
state = {
    "query": "What is shape factor?",
    "chat_history": [],
    "query_type": classify_query("What is shape factor?"),
    "retrieval_attempts": 0,
    "retrieval_results": [],
    "retrieval_sufficient": False,
    "plan": [],
    "current_step": 0,
    "next_agent": "",
}

for step in graph.stream(state, {"recursion_limit": 20}):
    node_name = list(step.keys())[0]
    node_data = step[node_name]
    print(f"[{node_name}]", node_data.get("plan", []))

# 输出:
# [supervisor] ['retrieve shape factor definition and formula', 'explain concept']
# [retrieval] []
# [concept] []
```

---

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
├── ITERATION_0.md           # 迭代0文档
├── ITERATION_1.md           # 迭代1文档
├── ITERATION_2.md           # 迭代2文档
├── ITERATION_3.md           # 迭代3文档
├── ITERATION_4.md           # 迭代4文档
│
├── preprocess/              # 文档预处理
│   ├── extract_lectures.py  # PyMuPDF PDF 提取
│   ├── normalize_txt.py     # pdftotext 乱码修复
│   ├── chunk_lectures.py    # 课件分块（3页合并窗口）
│   ├── chunk_exams.py       # 试卷/答案分块（题号边界切分）
│   ├── chunk_markdown.py    # 复习笔记分块（Markdown标题感知）
│   ├── run_chunking.py      # 完整分块流水线入口
│   └── build_metadata.py    # 元数据映射（课件→主题、考题→考点）
│
├── indexing/                # 三层索引
│   ├── vector_index.py      # BGE-M3 FlagEmbedding + ChromaDB 稠密索引
│   ├── sparse_index.py      # BM25 稀疏关键词索引
│   ├── build_sparse.py      # BM25 预构建 + pickle 持久化
│   ├── fusion.py            # RRF 融合 + Cross-encoder 重排序
│   └── knowledge_graph.py   # 知识图谱构建 + BFS 游走 (NetworkX)
│
├── agents/                  # LangGraph 多智能体
│   ├── state.py             # TutorState TypedDict 共享状态
│   ├── supervisor.py        # Supervisor Agent（分析→规划→路由）
│   ├── retrieval_agent.py   # Retrieval Agent（混合搜索+Self-RAG自省重试）
│   ├── concept_agent.py     # Concept Agent（概念讲解+多来源交叉验证）
│   ├── calc_agent.py        # Calc Agent（分步计算+公式标注）
│   ├── exam_agent.py        # Exam Agent（出题+批改）
│   ├── critic_agent.py      # Critic Agent（幻觉检测+引用验证）
│   └── tools.py             # 共享工具层（懒加载+KG增强搜索）
│
├── graph/                   # LangGraph 编排
│   ├── build_graph.py       # 图编译器 + query分类器
│   └── router.py            # 条件路由（Supervisor→Retrieval→Specialist）
│
├── prompts/templates.py     # 8套Prompt模板（System/Agent专用）
├── evaluation/ragas_eval.py # RAGAS评估 + A/B策略对比（5道ground truth）
├── server/                  # FastAPI 服务
│   ├── main.py              # 应用入口（/chat SSE, /exam/generate, /health）
│   └── schemas.py           # Pydantic 数据模型
├── ui/app.py                # Gradio Web UI
└── tests/                   # 测试（占位）
```

## 踩过的坑（重要经验的沉淀）

### 1. sentence-transformers 与 BGE-M3 不兼容

**现象**: `TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]`

**原因**: BGE-M3 是 **FlagEmbedding 家族的模型**，不能用 `sentence-transformers` 加载。新版 tokenizer 接口不兼容。

**解法**: 直接使用 `FlagEmbedding.BGEM3FlagModel`，自己实现 LangChain `Embeddings` 接口（见 `vector_index.py:BgeM3FlagEmbedding`）。

### 2. 幻灯片 PDF 文本提取的格式问题

`pdftotext` 对希腊字母、上下标输出乱码（`𝞮𝞮` → 应该是 `EI`）。建立了 100+ 字符的符号映射表归一化。

### 3. 试卷格式跨年份不一致

9 年试卷，至少 3 种题目格式：`[Total marks N]`（2022-2025）、`Question N`（2014-2019）、`N. [M marks]`（2020-2021）。正则需兼容所有年份。

### 4. 分块策略是 RAG 系统最重要的超参数

同一个问题，用固定 1000-token 切分 vs 按题目边界切分，检索精度差 2-3 倍。最终采用了 5 种源感知分块策略。

## 开发迭代

| 迭代 | 内容 | 核心产出 | 状态 |
|------|------|---------|------|
| 0 | 基础 RAG | 文档处理 + 738chunk 三层索引 + 混合检索 + CLI | ✅ |
| 1 | Agentic 升级 | LangGraph 多智能体 + Supervisor + Self-RAG 自省回路 | ✅ |
| 2 | 知识图谱 | 79节点269边 KG + Exam/Critic Agent + 图增强搜索 | ✅ |
| 3 | 工程化 | FastAPI + SSE 流式 + Docker + RAGAS 评估 | ✅ |
| 4 | 打磨 | Gradio Web UI + 全面测试 + 中英文文档 | ✅ |

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
│     │     ↑ retry on insufficient (Self-RAG)       │
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
# Clone & setup
git clone https://github.com/ZSHYC/cer-rag.git && cd cer-rag
conda create -n cer_rag python=3.11 -y && conda activate cer_rag
pip install -r requirements.txt
cp .env.example .env  # edit with your DeepSeek API key

# Build index (one-time, ~5 min)
python rag/build_index.py

# Launch — pick your interface
python rag/cli.py                  # CLI with Rich formatting
python rag/ui/app.py               # Gradio Web UI (http://localhost:7860)
python -m rag.server.main          # FastAPI server (http://localhost:8000)
docker-compose up -d               # Docker deployment
```

## Complete Usage Guide

### CLI Commands

| Command | Action |
|---------|--------|
| `:q` | Quit |
| `:s` | Toggle source display (show/hide retrieved document table) |
| `:d` | Toggle debug mode (real-time agent execution trace) |

### Query Types & Agent Routing

| Type | Example | Route |
|------|---------|-------|
| Conceptual | "What is Rational Ship Structural Design?" | Supervisor → Retrieval → Concept Agent |
| Calculation | "Calculate the shape factor of a square section" | Supervisor → Retrieval → Calc Agent |
| Formula | "What is Faulkner's effective width equation?" | Supervisor → Retrieval → Concept Agent |
| Exam Gen | "Generate 3 questions about buckling" | Supervisor → Exam Agent |
| Cross-lingual | "用中文解释 plastic collapse" | Auto-detect language, bilingual answer |

### FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Chat with SSE streaming |
| `POST` | `/exam/generate` | Generate mock exam questions |
| `POST` | `/exam/grade` | Grade student answers |

```bash
# Health check
curl http://localhost:8000/health

# Chat (SSE streaming)
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
  -d '{"question": "Calculate shape factor of square section", "student_answer": "φ = 1.5"}'
```

### Docker

```bash
docker-compose up -d          # Start (port 8000)
docker-compose logs -f        # View logs
docker-compose down           # Stop
```

### RAGAS Evaluation

```bash
python rag/evaluation/ragas_eval.py
# Outputs per-test-case metrics: docs retrieved, Self-RAG retries, answer quality
```

### Knowledge Graph (Python API)

```python
from rag.indexing.knowledge_graph import get_kg, graph_traverse

kg = get_kg()  # 79 nodes, 269 edges
results = graph_traverse("buckling", max_depth=2)
# Returns related formulas, exam questions, lectures
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

1. **BGE-M3 belongs to FlagEmbedding, not sentence-transformers** — loading it via sentence-transformers triggers `TextEncodeInput` errors. Use `FlagEmbedding.BGEM3FlagModel` directly.
2. **Chunking is the most important hyperparameter in RAG** — switching from fixed-size to source-aware chunking improved retrieval precision by 2-3x.
3. **Self-reflection dramatically improves retrieval quality** — a simple "is this sufficient?" check with re-query significantly reduces "I don't know" answers.
4. **PDF slide extraction is non-trivial** — Greek letters and subscripts from `pdftotext` require normalization mappings. Diagrams in slides are lost — cite slide numbers instead.
5. **Exam paper formats vary across years** — 9 years of exams had 3+ different question marking formats requiring multi-pattern regex compatibility.

## Development Roadmap

| Iteration | Focus | Key Output | Status |
|-----------|-------|------------|--------|
| 0 | Basic RAG | Doc processing, 738 chunks, hybrid retrieval, CLI | ✅ |
| 1 | Agentic Upgrade | LangGraph multi-agent, Supervisor, Self-RAG loop | ✅ |
| 2 | Knowledge Graph | 79-node KG, Exam/Critic agents, KG-enhanced search | ✅ |
| 3 | Productionization | FastAPI SSE, Docker, RAGAS evaluation, A/B testing | ✅ |
| 4 | Polish | Gradio UI, comprehensive testing, bilingual docs | ✅ |

## License

MIT License — feel free to use and adapt.

---

<p align="center">
  <sub>Built with LangGraph · DeepSeek V4 Pro · ChromaDB · BGE-M3 · NetworkX · FastAPI</sub>
</p>
