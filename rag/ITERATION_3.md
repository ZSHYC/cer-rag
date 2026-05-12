# 迭代 3：工程化 — FastAPI + Docker + RAGAS

**日期**: 2026-05-12
**状态**: 已完成

---

## 新增组件

### FastAPI 后端
- `/chat` — SSE 流式返回，实时推送 Agent 执行状态和最终答案
- `/exam/generate` — 出题接口
- `/health` — 健康检查
- CORS 全开，Pydantic 数据校验

### Docker 部署
```bash
docker-compose up -d  # 一键启动
```
- Python 3.11-slim 基础镜像
- 数据目录 volume 挂载
- 环境变量注入 (.env)

### RAGAS 评估
- 5 道历年真题作为 ground truth
- 评估维度：检索召回数、Self-RAG 重试次数、答案内容
- A/B 策略对比框架（不同检索策略可替换对比）

### 新增文件
```
rag/server/main.py          # FastAPI 入口
rag/server/schemas.py       # Pydantic 模型
rag/evaluation/ragas_eval.py # 评估框架
Dockerfile                  # 容器化
docker-compose.yml          # 编排
requirements.txt            # 完整依赖
```
