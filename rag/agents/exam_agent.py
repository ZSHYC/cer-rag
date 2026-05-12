"""Exam Agent —— 出题 + 批改"""
from langchain_core.messages import SystemMessage, HumanMessage
from rag.agents.state import TutorState
from rag.agents.tools import get_llm, docs_to_context, do_hybrid_search
from rag.prompts.templates import EXAM_GENERATE_PROMPT, EXAM_GRADE_PROMPT


EXAM_GENERATE_SYSTEM = """You are the Exam Agent for SESS3026 Marine Structures.

Generate practice exam questions based on past exam patterns. Make sure:
1. Question structure matches recent exam formats
2. You can modify numbers while keeping the same question type
3. Include mark breakdown
4. Cite the original exam question you based it on

{context}

Generate {num_questions} exam questions about: {topic}
"""


EXAM_GRADE_SYSTEM = """You are the Exam Grader for SESS3026 Marine Structures.

Compare the student's answer against the reference solution. Provide:
1. Overall correctness (correct / partially correct / incorrect)
2. Step-by-step scoring (which steps got marks, which didn't)
3. Specific error explanations
4. Recommended topics to review

Be fair but strict — the real exam expects precise working.

Reference answer:
{reference}

Student answer:
{student_answer}
"""


def exam_generate_state(state: TutorState) -> dict:
    """Exam Agent 节点：生成模拟题"""
    # 这是一个可选的独立入口，通过 state 中的特殊标记触发
    query = state["query"]
    # 可以从 query 中提取 topic 和数量
    topic = state.get("query_type", "general")
    num = 2

    # 检索相关考题
    fused = do_hybrid_search(f"{topic} exam questions", source_type="exam")
    docs = [doc for doc, _ in fused]
    ctx = docs_to_context(fused[:5], max_tokens=3000)

    llm = get_llm()
    prompt = EXAM_GENERATE_SYSTEM.format(context=ctx, num_questions=num, topic=topic)
    messages = [SystemMessage(content=prompt), HumanMessage(content=query)]
    response = llm.invoke(messages)

    return {"final_answer": response.content, "retrieval_results": docs}


def exam_grade_node(state: TutorState) -> dict:
    """Exam Agent 节点：批改答案"""
    # 从 state 获取用户答案
    student_answer = state["query"]
    docs = state.get("retrieval_results", [])

    # 组装参考答案
    ctx = docs_to_context([(doc, 1.0) for doc in docs], max_tokens=6000)

    llm = get_llm()
    prompt = EXAM_GRADE_SYSTEM.format(reference=ctx, student_answer=student_answer)
    messages = [SystemMessage(content=prompt), HumanMessage(content=student_answer)]
    response = llm.invoke(messages)

    return {"final_answer": response.content}
