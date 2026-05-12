"""预构建 BM25 索引并保存"""
import json
import pickle
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from rag.config import CHUNKS_DIR


def build_and_save_sparse(chunks_path: str = None, output_path: str = None):
    """从 chunks JSONL 构建 BM25 索引并保存"""
    if chunks_path is None:
        chunks_path = str(CHUNKS_DIR / "all_chunks.jsonl")
    if output_path is None:
        output_path = str(CHUNKS_DIR / "bm25_index.pkl")

    docs = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            docs.append(Document(
                page_content=data["content"],
                metadata=data["metadata"],
            ))

    print(f"Loading {len(docs)} documents for BM25...")
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 20

    # BM25Retriever 不支持直接 pickle，保存文档列表
    with open(output_path, 'wb') as f:
        pickle.dump(docs, f)

    print(f"BM25 index saved to {output_path} ({len(docs)} docs)")
    return docs


def load_sparse_index(path: str = None):
    """加载预构建的 BM25 文档列表"""
    if path is None:
        path = str(CHUNKS_DIR / "bm25_index.pkl")

    with open(path, 'rb') as f:
        docs = pickle.load(f)

    from rag.indexing.sparse_index import SparseIndex
    return SparseIndex(docs)


if __name__ == "__main__":
    build_and_save_sparse()
