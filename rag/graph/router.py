"""LangGraph 条件路由 —— 决定 Agent 之间的跳转"""
from rag.agents.state import TutorState


def route_after_supervisor(state: TutorState) -> str:
    """Supervisor 之后的路由：决定下一个 Agent"""
    next_agent = state.get("next_agent", "retrieval_agent")

    if next_agent == "done":
        return "done"
    if next_agent == "retrieval_agent":
        return "retrieval_agent"
    if next_agent in ("concept_agent", "calc_agent"):
        # 如果 supervisor 直接路由到 specialist，先做检索
        return "retrieval_agent"

    return "retrieval_agent"


def route_after_retrieval(state: TutorState) -> str:
    """检索之后的路由：根据查询类型决定用哪个 specialist agent"""
    # 如果检索不充分且还有重试次数，可以再次检索
    if not state.get("retrieval_sufficient", True) and state.get("retrieval_attempts", 0) < 2:
        return "retrieval_agent"  # 重新检索（已改写查询）

    qtype = state.get("query_type", "conceptual")

    if qtype == "calculation":
        return "calc_agent"
    else:
        return "concept_agent"


def route_after_specialist(state: TutorState) -> str:
    """Specialist agent 之后：直接结束"""
    return "done"
