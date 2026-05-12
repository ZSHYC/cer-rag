"""稠密向量索引 —— ChromaDB + BGE-M3 (FlagEmbedding)"""
from typing import List, Optional
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rag.config import CHROMA_DIR, EMBEDDING_MODEL


class BgeM3FlagEmbedding(Embeddings):
    """BGE-M3 embedding via FlagEmbedding (not sentence-transformers)"""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(model_name, use_fp16=(device != "cpu"))
        self.device = device

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.model.encode(
            texts, batch_size=8, max_length=8192,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        return result['dense_vecs'].tolist()

    def embed_query(self, text: str) -> List[float]:
        result = self.model.encode(
            [text], batch_size=1, max_length=8192,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        vec = result['dense_vecs']
        if isinstance(vec, np.ndarray):
            vec = vec.tolist()
        return vec[0]


def create_embedder() -> Embeddings:
    return BgeM3FlagEmbedding()


def build_vector_index(
    documents: List[Document],
    collection_name: str = "marine_structures_rag",
    persist_dir: Optional[str] = None,
):
    """构建 ChromaDB 向量索引"""
    if persist_dir is None:
        persist_dir = str(CHROMA_DIR)

    from langchain_chroma import Chroma

    embedder = create_embedder()
    batch_size = 16
    total = len(documents)

    print(f"开始嵌入 {total} 个文档...")
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        if i == 0:
            vectorstore = Chroma.from_documents(
                documents=batch, embedding=embedder,
                persist_directory=persist_dir, collection_name=collection_name,
            )
        else:
            vectorstore.add_documents(batch)
        print(f"  进度: {min(i + batch_size, total)}/{total}")

    print(f"向量索引构建完成，存储于: {persist_dir}")
    return vectorstore


def load_vector_index(
    collection_name: str = "marine_structures_rag",
    persist_dir: Optional[str] = None,
):
    """加载已有的 ChromaDB 索引"""
    if persist_dir is None:
        persist_dir = str(CHROMA_DIR)

    from langchain_chroma import Chroma

    embedder = create_embedder()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedder,
        collection_name=collection_name,
    )


def dense_search(
    vectorstore,
    query: str,
    k: int = 20,
    filter_dict: Optional[dict] = None,
) -> List[tuple]:
    """稠密语义搜索，返回 (Document, score) 列表"""
    if filter_dict:
        results = vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)
    else:
        results = vectorstore.similarity_search_with_score(query, k=k)
    return results
