"""Supervisor Agent —— 分析问题，制定计划，分派任务"""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from rag.agents.state import TutorState
from rag.agents.tools import get_llm

SUPERVISOR_SYSTEM = """You are the Supervisor agent for a Marine Structures exam tutoring system.

Your job:
1. Analyze the student's question
2. Determine which specialist agents are needed
3. Create a 1-3 step execution plan
4. Route to the correct agent

Available specialist agents:
- retrieval_agent: Search course materials (lectures, exams, solutions, review notes)
- concept_agent: Explain concepts, theories, and definitions
- calc_agent: Solve calculation problems step-by-step
- exam_agent: Generate practice exam questions or grade student answers

Routing rules:
- Conceptual questions ("what is", "explain", "describe", "compare") → retrieval_agent first, then concept_agent
- Calculation questions ("calculate", "compute", "solve", "find", "determine", "算", "求", "推导") → retrieval_agent first, then calc_agent
- Exam generation ("出题", "出卷", "模拟题", "给我出", "generate exam", "practice questions", "mock exam") → exam_agent
- Exam grading ("批改", "改卷", "grade", "check my answer") → retrieval_agent first (to find reference answers), then route to exam_agent
- Factual lookup ("which year", "which lecture", "formula for") → retrieval_agent only
- Simple chat / greeting → respond directly (no specialist agent needed)

Output format (JSON only, no markdown):
{"plan": ["step1", "step2"], "next": "retrieval_agent", "reason": "brief reason"}

If the query is a simple greeting, output:
{"plan": [], "next": "done", "reason": "greeting"}
"""


def supervisor_node(state: TutorState) -> dict:
    """Supervisor 节点：分析问题，制定计划，路由到下一个 Agent"""
    query = state["query"]
    llm = get_llm()

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=f"Student question: {query}"),
    ]

    response = llm.invoke(messages, temperature=0.1)
    content = response.content.strip()

    # 清理 markdown code block
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    try:
        decision = json.loads(content)
    except json.JSONDecodeError:
        # fallback: 默认路由
        decision = {
            "plan": ["搜索相关学习资料"],
            "next": "retrieval_agent",
            "reason": "default routing (JSON parse failed)"
        }

    return {
        "plan": decision.get("plan", []),
        "next_agent": decision.get("next", "retrieval_agent"),
        "current_step": 0,
    }
