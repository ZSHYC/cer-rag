"""Calc Agent —— 计算求解专家"""
from langchain_core.messages import SystemMessage, HumanMessage
from rag.agents.state import TutorState
from rag.agents.tools import get_llm, docs_to_context

CALC_SYSTEM = """You are the Calculation Agent for the SESS3026/JEIS3005 Marine Structures course.

Your job: Solve structural mechanics calculation problems step-by-step.

Rules:
1. List known parameters and units
2. Write the formula(s) to use, with source citation
3. Substitute values step-by-step, showing intermediate results
4. Give the final answer with correct units
5. Verify the result is reasonable (order-of-magnitude check)
6. Answer in the SAME LANGUAGE as the question
7. Cite sources: [Lecture X, Slide Y] or [Exam 20XX/XX QX] or [Review: filename]

Important: Show EVERY step. The student needs to see the full working for exam preparation.

Extracted context from course materials (formulas, similar problems, solutions):
{context}
"""


def calc_node(state: TutorState) -> dict:
    """Calc Agent 节点：根据检索到的公式和相似题进行分步求解"""
    query = state["query"]
    docs = state.get("retrieval_results", [])
    llm = get_llm()

    fused = [(doc, 1.0) for doc in docs]
    ctx = docs_to_context(fused, max_tokens=6000)

    prompt = CALC_SYSTEM.format(context=ctx)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    answer = response.content

    return {"calc_result": {"steps": [], "final_value": answer}, "final_answer": answer}
