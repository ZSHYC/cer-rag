#!/usr/bin/env python3
"""运行完整索引构建流水线：分块 -> 嵌入 -> ChromaDB -> BM25"""
import sys, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.preprocess.run_chunking import run_full_chunking
from rag.indexing.vector_index import build_vector_index
from rag.indexing.build_sparse import build_and_save_sparse
from rag.config import CHROMA_DIR


def main():
    print("=" * 60)
    print("Marine Structures RAG - 索引构建")
    print("=" * 60)

    print("\n[Step 1] 文档分块...")
    all_docs = run_full_chunking()

    print(f"\n[Step 2] 构建向量索引 ({len(all_docs)} 个文档)...")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vs = build_vector_index(all_docs)
    print(f"向量索引: {vs._collection.count()} 条")

    print("\n[Step 3] 构建 BM25 索引...")
    build_and_save_sparse()

    print("\n" + "=" * 60)
    print("索引构建完成！运行: python rag/cli.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
