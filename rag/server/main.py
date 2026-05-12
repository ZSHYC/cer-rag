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

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_tutor_graph(enable_critic=False)
    return _graph


@app.post("/chat")
async def chat(request: ChatRequest):
    """对话接口：SSE 流式返回每个 Agent 执行状态 + 最终答案"""
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

            if node in ("concept", "calc", "exam_generate", "exam_grade"):
                answer = data.get("final_answer", "")
                if answer:
                    yield f"data: {json.dumps({'answer': answer}, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/exam/generate")
async def exam_generate(request: ExamGenerateRequest):
    """出题接口"""
    graph = get_graph()
    query = f"Generate {request.num_questions} practice exam questions about {request.topic}."

    state = {
        "query": query, "chat_history": [],
        "query_type": "exam", "retrieval_attempts": 0,
        "retrieval_results": [], "retrieval_sufficient": False,
        "plan": [], "current_step": 0, "next_agent": "",
    }

    final = None
    for step in graph.stream(state, {"recursion_limit": 20}):
        final = step

    if final:
        for node_name, node_data in final.items():
            answer = node_data.get("final_answer", "")
            if answer:
                return {"answer": answer, "topic": request.topic}

    return {"answer": "", "error": "No response generated"}


@app.post("/exam/grade")
async def exam_grade(request: ExamGradeRequest):
    """批改接口：对比学生答案和标准答案"""
    graph = get_graph()

    # 先检索标准答案
    search_state = {
        "query": request.question, "chat_history": [],
        "query_type": "calculation", "retrieval_attempts": 0,
        "retrieval_results": [], "retrieval_sufficient": True,
        "plan": ["search for reference answer"], "current_step": 0,
        "next_agent": "retrieval_agent",
    }

    final = None
    for step in graph.stream(search_state, {"recursion_limit": 10}):
        final = step

    reference_docs = []
    if final:
        for data in final.values():
            reference_docs = data.get("retrieval_results", [])

    # 用检索结果 + 学生答案生成批改
    from rag.agents.tools import docs_to_context, get_llm
    from rag.prompts.templates import EXAM_GRADE_PROMPT
    from langchain_core.messages import SystemMessage, HumanMessage

    ctx = docs_to_context([(doc, 1.0) for doc in reference_docs], max_tokens=6000)
    llm = get_llm()
    prompt = EXAM_GRADE_PROMPT.format(
        reference_answer=ctx,
        student_answer=request.student_answer,
    )
    messages = [SystemMessage(content=prompt), HumanMessage(content=request.student_answer)]
    response = llm.invoke(messages)

    return {"feedback": response.content}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
