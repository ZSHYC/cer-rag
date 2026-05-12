"""稀疏关键词索引 —— BM25"""
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


class SparseIndex:
    """BM25 稀疏索引封装"""

    def __init__(self, documents: List[Document] = None):
        self.documents = documents or []
        self.retriever = None
        if self.documents:
            self._build()

    def _build(self):
        self.retriever = BM25Retriever.from_documents(self.documents)
        self.retriever.k = 20

    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)
        self._build()

    def search(self, query: str, k: int = 20) -> List[Document]:
        if self.retriever is None:
            return []
        self.retriever.k = k
        return self.retriever.invoke(query)

    def search_with_scores(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """搜索并返回 (Document, score)"""
        docs = self.search(query, k)
        # BM25Retriever 不直接返回分数，这里给降序近似分
        results = []
        for i, doc in enumerate(docs):
            score = 1.0 / (i + 1)  # 顺序衰减作为近似分数
            results.append((doc, score))
        return results
