"""Critic Agent —— 质量审查"""
from langchain_core.messages import SystemMessage, HumanMessage
from rag.agents.state import TutorState
from rag.agents.tools import get_llm

CRITIC_SYSTEM = """You are the Critic Agent for a Marine Structures exam tutoring system.

Your job: Review the AI-generated answer and check for quality issues.

Check:
1. Are citations valid? (Do the referenced lectures/exams actually exist in the context?)
2. Is the calculation numerically reasonable?
3. Is there any hallucination (information NOT in the provided context)?
4. If there is a known reference answer in the context, does the answer match?

Output JSON:
{
    "citations_valid": true/false,
    "calculation_reasonable": true/false,
    "hallucination_detected": true/false,
    "issues": ["specific issue 1", "specific issue 2"],
    "verdict": "pass" or "needs_revision"
}

Context used for generation:
{context}

Answer to review:
{answer}
"""


def critic_node(state: TutorState) -> dict:
    """Critic Agent 节点：审查生成的回答"""
    answer = state.get("final_answer", "")
    if not answer:
        answer = state.get("concept_answer", "") or state.get("calc_result", {}).get("final_value", "")
    if not answer or len(answer) < 50:
        return {"error": "No answer to review"}

    docs = state.get("retrieval_results", [])
    from rag.agents.tools import docs_to_context
    ctx = docs_to_context([(doc, 1.0) for doc in docs], max_tokens=4000)

    llm = get_llm()
    prompt = CRITIC_SYSTEM.format(context=ctx, answer=answer[:3000])
    messages = [SystemMessage(content=prompt), HumanMessage(content="Review this answer")]

    import json
    response = llm.invoke(messages)
    content = response.content.strip()
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    try:
        feedback = json.loads(content.strip())
    except json.JSONDecodeError:
        feedback = {"verdict": "pass", "issues": []}

    return {"final_answer": answer, "needs_revision": feedback.get("verdict") != "pass"}
