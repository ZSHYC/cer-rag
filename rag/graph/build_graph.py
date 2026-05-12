"""LangGraph 状态图编译器"""
from langgraph.graph import StateGraph, END
from rag.agents.state import TutorState
from rag.agents.supervisor import supervisor_node
from rag.agents.retrieval_agent import retrieval_node
from rag.agents.concept_agent import concept_node
from rag.agents.calc_agent import calc_node
from rag.graph.router import route_after_supervisor, route_after_retrieval, route_after_specialist


def build_tutor_graph() -> StateGraph:
    """构建多 Agent 状态图

    Graph structure:
        supervisor → retrieval → (concept | calc) → END
                        ↑              │
                        └── retry ─────┘
    """
    workflow = StateGraph(TutorState)

    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("concept", concept_node)
    workflow.add_node("calc", calc_node)

    # 入口
    workflow.set_entry_point("supervisor")

    # Supervisor → retrieval (or done)
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "retrieval_agent": "retrieval",
            "done": END,
        }
    )

    # Retrieval → concept or calc (or self-retry)
    workflow.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "retrieval_agent": "retrieval",  # self-loop for retry
            "concept_agent": "concept",
            "calc_agent": "calc",
        }
    )

    # Specialist agents → END
    workflow.add_edge("concept", END)
    workflow.add_edge("calc", END)

    return workflow.compile()


# ---- Query classification helper (used by CLI before graph invocation) ----

def classify_query(query: str) -> str:
    q = query.lower()
    calc_kw = ['calculate', 'compute', 'solve', 'find', 'determine', 'derive',
               'shape factor', 'modulus', 'buckling stress', 'stiffness',
               '计算', '求', '算', '求解', '推导']
    formula_kw = ['formula', 'equation', 'expression', '公式', '表达式']
    ref_kw = ['which year', 'which exam', 'which lecture', '哪年', '哪个课件', '哪道题']

    if any(k in q for k in ref_kw):
        return "reference"
    if any(k in q for k in formula_kw):
        return "formula"
    if any(k in q for k in calc_kw):
        return "calculation"
    return "conceptual"
