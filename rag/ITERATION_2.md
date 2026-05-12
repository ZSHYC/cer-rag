# 迭代 2：知识图谱 + 高级 Agent

**日期**: 2026-05-12
**状态**: 已完成

---

## 1. 新增组件

### 1.1 知识图谱 (Knowledge Graph)

**规模**: 80 节点, 290 边

| 实体类型 | 数量 | 示例 |
|---------|------|------|
| Concept | 27 | Rational Design, Buckling, Plastic Collapse... |
| Formula | 8 | σcr, Faulkner, Euler, Shape Factor, Grillage... |
| Lecture | 21 | L1-L21, 含主题名称 |
| ExamQuestion | 24 | 2014/15 Q1 - 2024/25 Q4 |

**关系类型**:
- `EXPLAINED_IN`: Concept → Lecture (概念在哪节课讲)
- `TESTED_BY`: Concept → ExamQuestion (概念在哪些考题中出现)
- `RELATED_TO`: Formula ↔ Concept (公式与概念关联)

**图游走**:
```python
traverse("buckling") → 49 个关联项
  - 所有涉及 buckling 的考题 (1-hop)
  - 相关课件 (1-hop)
  - 相关公式 (1-hop)
```

### 1.2 Exam Agent

- **出题**: 根据 topic 从知识图谱找到相关考题 → 改编数值生成模拟题
- **批改**: 对比学生答案 vs 标准答案 → 分步评分 + 错误分析 + 复习建议

### 1.3 Critic Agent

质量审查: 引用有效性、计算合理性、幻觉检测、与标准答案一致性

### 1.4 KG-Enhanced Search

常规混合搜索 + 知识图谱游走补充: 从 query 提取概念关键词 → 图游走找关联考题 → 额外补充搜索

---

## 2. 工程化组件 (迭代3起步)

- FastAPI `/chat` SSE 流式、`/exam/generate`、`/health`
- RAGAS 评估: 5 道真题作为 ground truth, 对比检索策略
- Docker + docker-compose 一键部署

---

## 3. 新增文件

```
rag/indexing/knowledge_graph.py   # 知识图谱构建+游走 (networkx)
rag/agents/exam_agent.py           # Exam Agent
rag/agents/critic_agent.py         # Critic Agent
rag/server/main.py                 # FastAPI 后端
rag/server/schemas.py              # Pydantic 数据模型
rag/evaluation/ragas_eval.py       # RAGAS 评估 + A/B 对比
Dockerfile                         # 容器化
docker-compose.yml                 # 一键部署
```
