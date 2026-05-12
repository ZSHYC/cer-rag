"""FastAPI 数据模型"""
from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []


class ChatResponse(BaseModel):
    answer: str
    query_type: str
    sources: List[dict]
    plan: List[str]
    retrieval_attempts: int


class ExamGenerateRequest(BaseModel):
    topic: str = "buckling"
    num_questions: int = 2


class ExamGradeRequest(BaseModel):
    question: str
    student_answer: str


class CoverageResponse(BaseModel):
    total_topics: int
    covered: List[str]
    uncovered: List[str]
    coverage_pct: float
