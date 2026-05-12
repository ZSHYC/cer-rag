"""LangGraph 共享状态"""
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class TutorState(TypedDict):
    # 用户输入
    query: str
    chat_history: List[Dict[str, str]]

    # Supervisor 规划
    plan: List[str]
    current_step: int
    next_agent: str

    # 检索结果
    retrieval_results: List[Document]
    retrieval_sufficient: bool
    retrieval_attempts: int  # 重试计数

    # 各 Agent 输出
    concept_answer: str
    calc_result: Dict[str, Any]  # {"steps": [...], "final_value": ..., "formula_used": [...]}

    # 最终输出
    final_answer: str
    citations: List[str]

    # 元数据
    query_type: str  # conceptual | calculation | formula | reference
    error: Optional[str]
