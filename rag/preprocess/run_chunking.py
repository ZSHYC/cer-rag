"""运行完整的分块流水线，生成所有 Document 对象"""
from typing import List
from langchain_core.documents import Document
from rag.preprocess.chunk_lectures import chunk_lectures
from rag.preprocess.chunk_exams import chunk_all_exams_and_answers
from rag.preprocess.chunk_markdown import chunk_review_markdowns
from rag.preprocess.build_metadata import build_all_metadata
from rag.config import DATA_DIR, CHUNKS_DIR


def run_full_chunking() -> List[Document]:
    """运行完整分块流水线"""
    print("=" * 60)
    print("开始完整分块流水线...")
    print("=" * 60)

    # 1. 课件分块
    print("\n[1/4] 课件分块...")
    lecture_docs = chunk_lectures()

    # 2. 试卷+答案+研讨题分块
    print("\n[2/4] 试卷与答案分块...")
    exam_docs = chunk_all_exams_and_answers()

    # 3. 复习笔记分块
    print("\n[3/4] 中文复习笔记分块...")
    review_docs = chunk_review_markdowns()

    # 4. 构建元数据
    print("\n[4/4] 构建元数据...")
    build_all_metadata()

    all_docs = lecture_docs + exam_docs + review_docs

    # 保存为 JSONL
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    output_path = CHUNKS_DIR / "all_chunks.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in all_docs:
            f.write(json.dumps({
                "content": doc.page_content,
                "metadata": doc.metadata,
            }, ensure_ascii=False) + '\n')

    print(f"\n{'=' * 60}")
    print(f"分块完成! 总计: {len(all_docs)} 个 chunk")
    print(f"课件: {len(lecture_docs)} | 试卷类: {len(exam_docs)} | 复习笔记: {len(review_docs)}")
    print(f"已保存至: {output_path}")
    print(f"{'=' * 60}")

    return all_docs


if __name__ == "__main__":
    run_full_chunking()
