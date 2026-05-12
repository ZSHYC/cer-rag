"""共享工具 —— 检索、生成等"""
import pickle
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from rag.config import (
    CHROMA_DIR, CHUNKS_DIR, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    DENSE_K, SPARSE_K, RERANK_TOP_K,
)
from rag.indexing.vector_index import load_vector_index, dense_search
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.fusion import hybrid_search, rerank_with_cross_encoder

# --- 全局懒加载 ---
_vectorstore = None
_sparse_index = None
_reranker = None
_llm = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vector_index()
    return _vectorstore


def get_sparse_index():
    global _sparse_index
    if _sparse_index is None:
        pickle_path = CHUNKS_DIR / "bm25_index.pkl"
        with open(pickle_path, 'rb') as f:
            docs = pickle.load(f)
        _sparse_index = SparseIndex(docs)
    return _sparse_index


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
        except Exception:
            _reranker = False  # sentinel: failed
    return _reranker if _reranker is not False else None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
            temperature=0.3, streaming=True,
        )
    return _llm


# --- 检索工具 ---

def do_dense_search(query: str, k: int = DENSE_K, source_type: str = None) -> List[Document]:
    filter_dict = None
    if source_type:
        filter_dict = {"source_type": source_type}
    results = dense_search(get_vectorstore(), query, k=k, filter_dict=filter_dict)
    return [doc for doc, _ in results]


def do_sparse_search(query: str, k: int = SPARSE_K) -> List[Document]:
    results = get_sparse_index().search(query, k=k)
    return results


def do_hybrid_search(
    query: str,
    dense_k: int = DENSE_K,
    sparse_k: int = SPARSE_K,
    source_type: str = None,
) -> List[tuple]:
    vs = get_vectorstore()
    si = get_sparse_index()

    def _dense(q, k, filter_dict=None):
        r = dense_search(vs, q, k=k, filter_dict=filter_dict)
        return [(doc, score) for doc, score in r]

    def _sparse(q, k):
        return si.search_with_scores(q, k=k)

    filter_dict = {"source_type": source_type} if source_type else None
    fused = hybrid_search(_dense, _sparse, query, dense_k=dense_k, sparse_k=sparse_k)

    reranker = get_reranker()
    if reranker and len(fused) > RERANK_TOP_K:
        fused = rerank_with_cross_encoder(query, fused, reranker, RERANK_TOP_K)

    return fused


def format_source(doc: Document) -> str:
    meta = doc.metadata
    st = meta.get("source_type", "")
    if st == "lecture":
        lec = meta.get("lecture_number", "?")
        sr = meta.get("slide_range") or meta.get("slide_num", "?")
        return f"[Lecture {lec}, Slide {sr}]"
    if st == "exam":
        return f"[Exam {meta.get('exam_year','?')} Q{meta.get('question_number','?')}]"
    if st == "solution":
        return f"[Answer {meta.get('exam_year','?')} Q{meta.get('question_number','?')}{meta.get('sub_question','')}]"
    if st == "seminar":
        return f"[Tutorial {meta.get('tutorial_number','?')}]"
    if st == "review_md":
        return f"[Review: {meta.get('source_file','?')}]"
    return f"[{meta.get('source_file','?')}]"


def docs_to_context(docs_with_scores: List[tuple], max_tokens: int = 8000) -> str:
    parts = []
    est = 0
    for doc, score in docs_with_scores:
        block = f"### {format_source(doc)}\n{doc.page_content}\n"
        e = len(block) // 3
        if est + e > max_tokens:
            break
        parts.append(block)
        est += e
    return "\n---\n".join(parts)


def do_graph_traverse(entity: str, max_depth: int = 2) -> str:
    """知识图谱游走：从概念出发找关联的公式、考题、课件"""
    from rag.indexing.knowledge_graph import graph_traverse
    results = graph_traverse(entity, max_depth=max_depth)

    if not results:
        return f"No graph results for '{entity}'"

    lines = [f"Knowledge Graph results for '{entity}':"]
    for r in results:
        line = f"  [{r['type']}] {r['label']} (distance={r['distance']}"
        if r.get('relation'):
            line += f", relation={r['relation']}"
        line += ")"
        if r.get('expression'):
            line += f"\n    Expression: {r['expression']}"
        if r.get('year'):
            line += f"\n    Year: {r['year']}"
        lines.append(line)

    return "\n".join(lines)


def do_kg_enhanced_search(query: str) -> List[tuple]:
    """知识图谱增强检索：常规混合搜索 + 图游走补充"""
    # 常规搜索
    fused = do_hybrid_search(query)
    docs = [doc for doc, _ in fused]

    # 图游走：从 query 提取可能的概念
    concepts = ["shape factor", "buckling", "stiffness matrix", "rational design",
                "plastic design", "plate buckling", "hull girder", "stiffened panel",
                "column buckling", "effective width", "faulkner", "grillage",
                "ultimate strength", "vibration"]

    from rag.indexing.knowledge_graph import graph_traverse
    for concept in concepts:
        if concept in query.lower():
            kg_results = graph_traverse(concept, max_depth=2)
            # 从图结果中找到对应的考题和课件，额外搜索
            for r in kg_results:
                if r['type'] == 'ExamQuestion' and r.get('year'):
                    extra_query = f"{r['year']} {r['label']}"
                    extra_fused = do_hybrid_search(extra_query, source_type="exam")
                    for doc, score in extra_fused[:2]:
                        docs.append(doc)
            break  # 只游走第一个匹配概念

    return [(doc, 1.0) for doc in docs]
