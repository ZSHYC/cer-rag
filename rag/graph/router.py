"""LangGraph 条件路由"""
from rag.agents.state import TutorState


def route_after_supervisor(state: TutorState) -> str:
    next_agent = state.get("next_agent", "retrieval_agent")

    if next_agent == "done":
        return "done"
    if next_agent == "exam_agent":
        return "exam_agent"
    if next_agent == "retrieval_agent":
        return "retrieval_agent"

    return "retrieval_agent"


def route_after_retrieval(state: TutorState) -> str:
    if not state.get("retrieval_sufficient", True) and state.get("retrieval_attempts", 0) < 2:
        return "retrieval_agent"

    qtype = state.get("query_type", "conceptual")
    if qtype == "calculation":
        return "calc_agent"
    return "concept_agent"


def route_after_specialist(state: TutorState) -> str:
    return "done"
