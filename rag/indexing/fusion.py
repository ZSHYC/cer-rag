"""检索融合 —— RRF + Cross-Encoder 重排序"""
from typing import List, Tuple
from langchain_core.documents import Document
from rag.config import RRF_K, RERANK_TOP_K


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    sparse_results: List[Tuple[Document, float]],
    k: int = RRF_K,
    dense_weight: float = 1.0,
    sparse_weight: float = 0.5,
) -> List[Tuple[Document, float]]:
    """RRF 融合稠密和稀疏搜索结果

    RRF score = sum(weight / (k + rank)) for each result list
    """
    scores = {}

    for rank, (doc, _) in enumerate(dense_results):
        doc_id = doc.page_content[:100]  # 用前100字符做近似去重
        rrf_score = dense_weight / (k + rank + 1)
        scores[doc_id] = (doc, scores.get(doc_id, (None, 0))[1] + rrf_score)

    for rank, (doc, _) in enumerate(sparse_results):
        doc_id = doc.page_content[:100]
        rrf_score = sparse_weight / (k + rank + 1)
        if doc_id in scores:
            scores[doc_id] = (scores[doc_id][0], scores[doc_id][1] + rrf_score)
        else:
            scores[doc_id] = (doc, rrf_score)

    # 按融合分降序排列
    fused = sorted(scores.values(), key=lambda x: x[1], reverse=True)
    return fused


def hybrid_search(
    dense_search_fn,
    sparse_search_fn,
    query: str,
    dense_k: int = 20,
    sparse_k: int = 20,
    source_filter: dict = None,
) -> List[Tuple[Document, float]]:
    """混合检索：稠密 + 稀疏 → RRF 融合"""
    dense_results = dense_search_fn(query, k=dense_k, filter_dict=source_filter)
    sparse_results = sparse_search_fn(query, k=sparse_k)

    # 确保格式一致
    if dense_results and not isinstance(dense_results[0], tuple):
        dense_results = [(d, 0.5) for d in dense_results]
    if sparse_results and not isinstance(sparse_results[0], tuple):
        sparse_results = [(d, 0.5) for d in sparse_results]

    return reciprocal_rank_fusion(dense_results, sparse_results)


def rerank_with_cross_encoder(
    query: str,
    candidates: List[Tuple[Document, float]],
    reranker_model=None,
    top_k: int = RERANK_TOP_K,
) -> List[Tuple[Document, float]]:
    """用 Cross-Encoder 重排序"""
    if len(candidates) <= top_k:
        return candidates

    from sentence_transformers import CrossEncoder

    if reranker_model is None:
        reranker_model = CrossEncoder('BAAI/bge-reranker-v2-m3')

    pairs = [(query, doc.page_content[:1000]) for doc, _ in candidates]
    scores = reranker_model.predict(pairs)

    # 按新分数排序
    scored = list(zip([doc for doc, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]
