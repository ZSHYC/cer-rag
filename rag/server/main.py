"""FastAPI 后端入口"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

from rag.server.schemas import ChatRequest, ExamGenerateRequest, ExamGradeRequest
from rag.graph.build_graph import build_tutor_graph, classify_query

app = FastAPI(
    title="Marine Structures AI Tutor",
    description="Agentic RAG Multi-Agent Exam Tutoring System",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 全局图实例（懒加载）
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_tutor_graph(enable_critic=False)
    return _graph


@app.post("/chat")
async def chat(request: ChatRequest):
    """对话接口：流式返回"""
    graph = get_graph()
    qtype = classify_query(request.query)
    state = {
        "query": request.query, "chat_history": request.chat_history or [],
        "query_type": qtype, "retrieval_attempts": 0, "retrieval_results": [],
        "retrieval_sufficient": False,
        "plan": [], "current_step": 0, "next_agent": "",
    }

    async def generate():
        for step in graph.stream(state, {"recursion_limit": 20}):
            node = list(step.keys())[0]
            data = step[node]
            yield f"data: {json.dumps({'node': node, 'plan': data.get('plan', []), 'docs': len(data.get('retrieval_results', []))}, ensure_ascii=False)}\n\n"

            if node in ("concept", "calc"):
                answer = data.get("final_answer", "")
                yield f"data: {json.dumps({'answer': answer}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/exam/generate")
async def exam_generate(request: ExamGenerateRequest):
    graph = get_graph()
    state = {
        "query": f"Generate {request.num_questions} questions about {request.topic}",
        "chat_history": [], "query_type": "exam", "retrieval_attempts": 0,
        "retrieval_results": [], "retrieval_sufficient": False,
        "plan": [], "current_step": 0, "next_agent": "exam_agent",
    }
    final = None
    for step in graph.stream(state, {"recursion_limit": 20}):
        final = step
    answer = final.get("exam_generate", {}).get("final_answer", "") if final else ""
    return {"answer": answer}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
