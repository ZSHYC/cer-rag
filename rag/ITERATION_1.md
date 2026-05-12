# 迭代 1：Agentic 升级 — LangGraph 多智能体系统

**日期**: 2026-05-12
**状态**: 已完成
**目标**: 将基础 RAG 管线升级为 LangGraph 多智能体协作系统，实现 Supervisor 调度 + 检索自省回路

---

## 1. 新增组件

### 1.1 多智能体架构

```
supervisor → retrieval → (concept | calc) → END
                ↑              │
                └── retry ─────┘  (Self-RAG 自省回路)
```

| Agent | 职责 | 核心能力 |
|-------|------|---------|
| **Supervisor Agent** | 分析问题、制定计划、分派任务 | LLM 推理，输出 JSON 执行计划 |
| **Retrieval Agent** | 混合检索 + 自省重试 | Self-RAG：检索 → 评估充分性 → 不足则改写查询重新检索 |
| **Concept Agent** | 概念讲解和理论阐述 | 多来源交叉验证，以考试得分点为导向组织回答 |
| **Calc Agent** | 分步求解计算题 | 公式 → 代入 → 中间结果 → 最终答案，每步标注来源 |

### 1.2 LangGraph 状态图

```python
class TutorState(TypedDict):
    query: str                    # 用户问题
    chat_history: List[dict]      # 对话历史
    plan: List[str]               # Supervisor 的执行计划
    next_agent: str               # 路由目标
    retrieval_results: List[Document]  # 检索到的文档
    retrieval_sufficient: bool    # 检索是否充分
    retrieval_attempts: int       # 重试次数（最多2次）
    concept_answer: str           # Concept Agent 输出
    calc_result: dict             # Calc Agent 输出
    final_answer: str             # 最终回答
    query_type: str               # conceptual | calculation | formula | reference
```

### 1.3 路由逻辑

```
Supervisor → 分类：
  - "done" → END (打招呼等)
  - 其他 → retrieval_agent

Retrieval → 条件分支：
  - insufficient && retries < 2 → 重新检索 (self-loop)
  - query_type = "calculation" → calc_agent
  - 其他 → concept_agent
```

### 1.4 Self-RAG 自省回路

这是 Agentic RAG 区别于普通 RAG 的关键：

```
1. Retrieval Agent 执行混合检索
2. 取 top-5 文档发 LLM 反思：
   "这些文档能充分回答用户问题吗？如果不能，缺少什么？建议用什么关键词重新搜索？"
3. LLM 回复 [SUFFICIENT] 或 [INSUFFICIENT: 缺少XXX，建议搜索YYY]
4. INSUFFICIENT → 用建议的新查询重新检索 → 合并结果
5. SUFFICIENT → 进入 specialist agent
```

最多重试 2 次，避免死循环。

---

## 2. 测试结果

### 测试 1: 概念题

**Q**: "What is Rational Ship Structural Design?"

```
[supervisor] plan=['Retrieve course materials on Rational Ship Structural Design', 
                    'Explain the concept based on retrieved content']
[retrieval] docs=8 retries=1
[concept] → 完整定义 + 关键要点 + 考试导向
```

### 测试 2: 计算题（中文）

**Q**: "计算正方形截面的形状系数"

```
[supervisor] plan=['Retrieve formula and relevant material for shape factor', 
                    'Calculate shape factor step-by-step']
[retrieval] docs=8 retries=1
[calc] → 已知参数 → 公式 → 代入 → φ = Zp/Z = 1.5
```

---

## 3. 与迭代0的对比

| 维度 | 迭代0 (Basic RAG) | 迭代1 (Agentic RAG) |
|------|-------------------|---------------------|
| 架构 | 单体 pipeline | 多智能体协作 (LangGraph) |
| 检索 | 一次检索 → 生成 | 检索 → 自省 → 不足则重试 |
| 路由 | 简单关键词分类 | LLM 推理 + 条件边 |
| 计算题 | 同样的 prompt 模板 | 专门的 Calc Agent + 分步模板 |
| 概念题 | 同样的 prompt 模板 | 专门的 Concept Agent + 考试导向 |
| 状态管理 | 无 | TypedDict 全流程状态追踪 |
| 调试 | 无 | :d debug 模式，每个节点执行追踪 |

---

## 4. 新增文件

```
rag/agents/
├── state.py               # LangGraph 共享状态 TypedDict
├── supervisor.py           # Supervisor Agent (规划+路由)
├── retrieval_agent.py      # Retrieval Agent (检索+自省)
├── concept_agent.py        # Concept Agent (概念讲解)
├── calc_agent.py           # Calc Agent (计算求解)
└── tools.py                # 共享工具层 (懒加载)

rag/graph/
├── build_graph.py          # LangGraph 编译器 + query classifier
└── router.py               # 条件路由函数

rag/cli.py                  # 重写：使用 graph.stream() 执行
test_agentic.py             # Agentic RAG 集成测试
```

---

## 5. 后续

迭代 2 将添加:
- Knowledge Graph 构建和游走
- Exam Agent (出题+批改)
- Critic Agent (质量审查)
