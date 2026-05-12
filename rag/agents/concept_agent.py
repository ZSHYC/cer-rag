"""Concept Agent —— 概念讲解专家"""
from langchain_core.messages import SystemMessage, HumanMessage
from rag.agents.state import TutorState
from rag.agents.tools import get_llm, docs_to_context

CONCEPT_SYSTEM = """You are the Concept Agent for the SESS3026/JEIS3005 Marine Structures course.

Your job: Explain theoretical concepts clearly, referencing course materials.

Rules:
1. Start with a clear definition
2. Expand with key points (use bullet format)
3. Connect to exam requirements (what the marker wants to see)
4. Cite sources: [Lecture X, Slide Y] or [Exam 20XX/XX QX] or [Review: filename]
5. Answer in the SAME LANGUAGE as the question
6. Keep technical terms in English even when answering in Chinese
7. Compare alternative viewpoints when different sources differ

Extracted context from course materials:
{context}
"""


def concept_node(state: TutorState) -> dict:
    """Concept Agent 节点：根据检索结果组织概念讲解"""
    query = state["query"]
    docs = state.get("retrieval_results", [])
    llm = get_llm()

    # 添加假分数用于排序
    fused = [(doc, 1.0) for doc in docs]
    ctx = docs_to_context(fused, max_tokens=6000)

    prompt = CONCEPT_SYSTEM.format(context=ctx)
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    answer = response.content

    return {"concept_answer": answer, "final_answer": answer}
