# 迭代 0：基础 RAG 系统

**日期**: 2026-05-12
**状态**: 已完成
**目标**: 搭建完整的文档处理 → 索引 → 检索 → 生成的 RAG 管线，CLI 可交互

---

## 1. 完成内容

### 1.1 数据资产

| 来源 | 数量 | 格式 | 语言 | chunk 数 |
|------|------|------|------|----------|
| 课件 (Lecture 1-21) | 21 个 PDF, 582 页 | 幻灯片 | English | 200 |
| 历年试卷 (2014-2025) | 9 套 | PDF → TXT | English | 25 |
| 参考答案 (2022-2025) | 3 年 | PDF → TXT | English | 15 |
| 研讨题解答 (Tutorial 1-4) | 4 次 | PDF → TXT | English | 23 |
| 中文复习笔记 | 12 个 `.md` | Markdown | 中英双语 | 475 |
| **合计** | | | | **738** |

### 1.2 技术栈

| 组件 | 选型 | 原因 |
|------|------|------|
| 文档提取 | PyMuPDF (fitz) | 对幻灯片格式 PDF 的文本提取质量好 |
| 文本分块 | LangChain TextSplitters | RecursiveCharacterTextSplitter + MarkdownHeaderTextSplitter 覆盖所有场景 |
| 嵌入模型 | BGE-M3 via FlagEmbedding | 免费、本地、中英双语、1024 维 |
| 稠密向量库 | ChromaDB (langchain-chroma) | 零运维、元数据过滤、Python 原生 |
| 稀疏索引 | BM25 (rank_bm25) | 关键词精确匹配 |
| 检索融合 | Reciprocal Rank Fusion (k=60) | 稠密 + 稀疏混合检索 |
| 重排序 | BGE-reranker-v2-m3 | Cross-encoder 精排 top-8 |
| LLM | DeepSeek V4 Pro (API) | 中英文 + 数学推理强、兼容 OpenAI 格式 |
| CLI | Rich | 美化终端输出、来源表格显示 |

### 1.3 分块策略

不同来源使用不同分块策略，这是检索质量的关键：

| 来源 | 分块策略 | chunk_size | overlap | 说明 |
|------|---------|-----------|---------|------|
| 课件 | RecursiveCharacterTextSplitter → 3页合并 | 800 | 150 | 知识点常跨幻灯片 |
| 试卷 | 自定义切分（按题号边界） | 每题1 chunk | - | 兼容多种年份格式 |
| 答案 | 自定义切分（按子题边界） | 每个子题1 chunk | - | 题目+解答成对 |
| 复习笔记 | MarkdownHeaderTextSplitter | 800 | 150 | 按 `##`/`###` 层级 |
| 公式清单 | RecursiveCharacterTextSplitter | 300 | 50 | 小颗粒，精确匹配 |

### 1.4 元数据体系

每个 chunk 携带丰富的结构化元数据，支持按来源/年份/题型/知识点过滤：

```python
# 课件 chunk 元数据
{"source_type": "lecture", "lecture_number": 18, "slide_range": "14-16",
 "lecture_topic": "Effective Width and Faulkner's Equation", "language": "en"}

# 试卷 chunk 元数据
{"source_type": "exam", "exam_year": "2022/23", "question_number": 3,
 "total_marks": 25, "content_type": "conceptual", "language": "en"}

# 复习笔记 chunk 元数据
{"source_type": "review_md", "source_file": "03_公式与计算题套路.md",
 "breadcrumb": "公式与计算题套路 > 截面模数与形状系数", "language": "zh"}
```

---

## 2. 踩过的坑

### 2.1 sentence-transformers 与 BGE-M3 不兼容

**现象**: `TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]`

**原因**: `langchain_community.embeddings.HuggingFaceBgeEmbeddings` 底层用 `sentence-transformers` 加载 BGE-M3 模型。新版本 `sentence-transformers` 的 tokenizer 接口与 BGE-M3 不兼容——BGE-M3 的 tokenizer 要求特定的输入格式。

**解决**: 放弃 `HuggingFaceBgeEmbeddings`，自己实现 `BgeM3FlagEmbedding` 类，直接调用 `FlagEmbedding.BGEM3FlagModel`。BGE-M3 是 FlagEmbedding 家族的模型，用原生库最稳妥。

```python
# 错误做法
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
embedder = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")  # ❌ 会崩

# 正确做法
from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
vec = model.encode([text])['dense_vecs']  # ✅ 稳定
```

### 2.2 LangChain 版本 deprecation

`HuggingFaceBgeEmbeddings` 和 `Chroma` 在新版 LangChain 中已被标记为 deprecated，官方推荐迁移到：
- `langchain_huggingface.HuggingFaceEmbeddings`
- `langchain_chroma.Chroma`

已安装 `langchain-chroma` 并使用新路径。

### 2.3 试卷切分格式兼容

试卷合集跨越 9 年，题目格式不一致：
- 2022-2025：`N. [Total marks M]`
- 2014-2019：`Question N` + `[M marks]`
- 2020-2021：`N. <text>` + 无显式 marks 标记

用多模式正则匹配兼容所有格式。详见 `chunk_exams.py:split_exam_questions()`。

### 2.4 中文查询长度变量名

中文查询中 "求"、"算"、"推导" 等词会被英文关键字 "find"、"solve"、"derive" 等匹配到，用于 query classification。分类逻辑按优先级：reference > formula > calculation > conceptual。

---

## 3. 端到端测试结果

### 测试 1: 英文概念题

**Q**: "What is Rational Ship Structural Design?"

**检索**: 26 个文档，top-3 全部来自 `review_md`（中文复习笔记）+ `exam`

**生成**: 英文回答，包含完整定义和分步流程，引用格式 `[Review: 09_核心题型完整答题模板_理论论述题.md]`

### 测试 2: 中文计算题

**Q**: "板屈曲的临界应力怎么算？"

**检索**: 40 个文档，命中了 `03_公式与计算题套路.md`、`00_总复习路线图.md` 等

**生成**: 中文回答，给出完整公式 σ_cr = (k·π²·E)/[12(1-ν²)]·(t/b)²，分步讲解参数含义，标注讲义引用

### 测试 3: 跨语言计算题

**Q**: "How to calculate the shape factor of a square cross-section?"

**检索**: 32 个文档，top-1 来自 `10_核心题型完整答题模板_板屈曲与极限强度计算题.md`

**生成**: 英文分步解答，Z = a³/6 → Zp = a³/4 → φ = 1.5，与标准答案一致

---

## 4. 项目文件结构

```
/home/zshyc/cer/rag/
├── config.py                      # 全局配置
├── cli.py                         # CLI 入口
├── build_index.py                 # 索引构建入口
│
├── preprocess/
│   ├── normalize_txt.py           # pdftotext 乱码修复
│   ├── extract_lectures.py        # PyMuPDF 提取课件 PDF
│   ├── chunk_lectures.py          # 课件分块
│   ├── chunk_exams.py             # 试卷/答案/研讨题分块
│   ├── chunk_markdown.py          # 复习笔记分块
│   ├── run_chunking.py            # 分块流水线入口
│   └── build_metadata.py          # 元数据映射
│
├── indexing/
│   ├── vector_index.py            # ChromaDB 稠密索引（BgeM3FlagEmbedding）
│   ├── sparse_index.py            # BM25 稀疏索引
│   ├── build_sparse.py            # BM25 预构建+持久化
│   └── fusion.py                  # RRF 融合 + Cross-encoder 重排序
│
├── prompts/
│   └── templates.py               # 所有 Prompt 模板
│
├── agents/                        # (迭代1占位)
├── graph/                         # (迭代1占位)
├── evaluation/                    # (迭代3占位)
├── tracking/                      # (迭代2占位)
├── server/                        # (迭代3占位)
├── ui/                            # (迭代4占位)
└── tests/                         # (迭代4占位)
```

---

## 5. 使用方式

```bash
# 激活环境
source /home/zshyc/miniforge3/etc/profile.d/conda.sh && conda activate cer_rag

# 运行 CLI
cd /home/zshyc/cer && python rag/cli.py

# CLI 命令
#   :q - 退出
#   :s - 切换来源显示
```

---

## 6. 后续迭代规划

| 迭代 | 内容 | 状态 |
|------|------|------|
| **迭代 0** | 基础 RAG：文档处理 + 索引 + 检索 + 生成 + CLI | ✅ 完成 |
| 迭代 1 | Agentic 升级：LangGraph + Supervisor Agent + 检索自省 | 待开始 |
| 迭代 2 | 知识图谱 + 高级 Agent (Exam/Critic) | 待开始 |
| 迭代 3 | 工程化：FastAPI + Docker + RAGAS 评估 | 待开始 |
| 迭代 4 | 打磨：Gradio UI + 文档 + 测试 | 待开始 |
