"""LangGraph 状态图编译器（含知识图谱 + Exam/Critic Agents）"""
from langgraph.graph import StateGraph, END
from rag.agents.state import TutorState
from rag.agents.supervisor import supervisor_node
from rag.agents.retrieval_agent import retrieval_node
from rag.agents.concept_agent import concept_node
from rag.agents.calc_agent import calc_node
from rag.agents.exam_agent import exam_generate_state, exam_grade_node
from rag.agents.critic_agent import critic_node
from rag.graph.router import (
    route_after_supervisor, route_after_retrieval,
    route_after_specialist,
)


def build_tutor_graph(enable_critic: bool = False) -> StateGraph:
    """构建完整多 Agent 状态图

    Graph: supervisor → retrieval → (concept|calc) → (critic?) → END
    Special modes: generate_exam → retrieval → END
    """
    workflow = StateGraph(TutorState)

    # 核心节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("concept", concept_node)
    workflow.add_node("calc", calc_node)

    # 高级节点
    workflow.add_node("exam_generate", exam_generate_state)
    workflow.add_node("exam_grade", exam_grade_node)
    if enable_critic:
        workflow.add_node("critic", critic_node)

    workflow.set_entry_point("supervisor")

    # Supervisor → retrieval (or done or exam_generate)
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "retrieval_agent": "retrieval",
            "exam_agent": "exam_generate",
            "done": END,
        }
    )

    # Retrieval → concept | calc | retry
    workflow.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "retrieval_agent": "retrieval",
            "concept_agent": "concept",
            "calc_agent": "calc",
        }
    )

    # Specialist → critic or END
    if enable_critic:
        workflow.add_conditional_edges(
            "concept",
            route_after_specialist,
            {"done": "critic", "critic": "critic"}
        )
        workflow.add_conditional_edges(
            "calc",
            route_after_specialist,
            {"done": "critic", "critic": "critic"}
        )
        workflow.add_edge("critic", END)
    else:
        workflow.add_edge("concept", END)
        workflow.add_edge("calc", END)

    workflow.add_edge("exam_generate", END)
    workflow.add_edge("exam_grade", END)

    return workflow.compile()


def classify_query(query: str) -> str:
    q = query.lower()
    calc_kw = ['calculate', 'compute', 'solve', 'find', 'determine', 'derive',
               'shape factor', 'modulus', 'buckling stress', 'stiffness',
               '计算', '求', '算', '求解', '推导']
    exam_kw = ['出题', '模拟题', '批改', 'grade', 'generate exam', 'mock exam',
               '出卷', '批卷', '给我出']
    formula_kw = ['formula', 'equation', 'expression', '公式', '表达式']
    ref_kw = ['which year', 'which exam', 'which lecture', '哪年', '哪个课件', '哪道题']

    if any(k in q for k in exam_kw):
        return "exam"
    if any(k in q for k in ref_kw):
        return "reference"
    if any(k in q for k in formula_kw):
        return "formula"
    if any(k in q for k in calc_kw):
        return "calculation"
    return "conceptual"
