"""课件幻灯片分块 + Parent-Child Chunking"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.config import LECTURE_CHUNK_SIZE, LECTURE_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, PARENT_CHUNK_SIZE
from rag.preprocess.extract_lectures import extract_all_lectures


def build_slide_documents(slides: list[dict]) -> List[Document]:
    """将幻灯片列表转为 LangChain Document（每页一个doc，保留元数据）"""
    docs = []
    for i, s in enumerate(slides):
        # 合并前后页作为上下文（前一页 + 当前 + 后一页）
        prev_text = slides[i - 1]['text'] if i > 0 else ""
        next_text = slides[i + 1]['text'] if i < len(slides) - 1 else ""
        full_context = f"{prev_text}\n---\n{s['text']}\n---\n{next_text}"

        docs.append(Document(
            page_content=s['text'],
            metadata={
                "source_type": "lecture",
                "source_file": s["source_file"],
                "lecture_number": s["lecture_num"],
                "lecture_topic": s["lecture_topic"],
                "slide_num": s["slide_num"],
                "title": s["title"],
                "language": "en",
                "context": full_context,  # 保留完整上下文
            }
        ))
    return docs


def chunk_lectures(slides: list[dict] = None) -> List[Document]:
    """对课件幻灯片执行分块策略

    策略：将 2-3 张连续幻灯片合并为一个 chunk（因为知识点常跨幻灯片）
    """
    if slides is None:
        slides = extract_all_lectures()

    docs = build_slide_documents(slides)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=LECTURE_CHUNK_SIZE,
        chunk_overlap=LECTURE_CHUNK_OVERLAP,
        separators=["\n---\n", "\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # 合并相邻幻灯片为一个 chunk
    merged_chunks = _merge_adjacent_slides(chunks)

    print(f"课件分块: {len(docs)} 页幻灯片 -> {len(merged_chunks)} 个 chunk")
    return merged_chunks


def _merge_adjacent_slides(chunks: List[Document], group_size: int = 3) -> List[Document]:
    """将相邻的幻灯片 chunk 按 group_size 合并"""
    merged = []
    for i in range(0, len(chunks), group_size):
        group = chunks[i:i + group_size]
        if not group:
            continue

        combined_text = "\n\n".join(d.page_content for d in group)
        first, last = group[0], group[-1]
        slide_range = f"{first.metadata['slide_num']}-{last.metadata['slide_num']}"

        merged.append(Document(
            page_content=combined_text,
            metadata={
                **first.metadata,
                "slide_range": slide_range,
            }
        ))
    return merged


# ---- Parent-Child Chunking ----

def create_parent_child_chunks(slides: list[dict] = None):
    """创建父子分块结构

    - 子 chunk (检索用): ~300 token，精准匹配
    - 父 chunk (上下文用): ~1500 token，完整上下文
    """
    if slides is None:
        slides = extract_all_lectures()

    docs = build_slide_documents(slides)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=50,
        separators=["\n---\n", "\n\n", "\n", " ", ""],
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=200,
        separators=["\n---\n", "\n\n", "\n", " ", ""],
    )

    child_chunks = child_splitter.split_documents(docs)
    parent_chunks = parent_splitter.split_documents(docs)

    print(f"Parent-Child 分块: {len(parent_chunks)} 父chunk, {len(child_chunks)} 子chunk")
    return parent_chunks, child_chunks
