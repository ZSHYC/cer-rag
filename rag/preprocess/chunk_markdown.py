"""中文复习笔记分块——按 Markdown 标题层级切分"""
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rag.config import REVIEW_MD_DIR, FORMULA_CHUNK_SIZE, FORMULA_CHUNK_OVERLAP, LECTURE_CHUNK_SIZE, LECTURE_CHUNK_OVERLAP

# 复习笔记的文件名前缀映射
FILE_TOPIC_MAP = {
    "00": "总复习路线图",
    "01": "历年试卷题型索引",
    "02": "高频知识点与答题模板",
    "03": "公式与计算题套路",
    "04": "近三年答案与数值索引",
    "05": "研讨题专题整理",
    "06": "期末冲刺清单",
    "07": "常见失分点与检查表",
    "08": "核心题型完整答题模板索引",
    "09": "核心题型答题模板_理论论述题",
    "10": "核心题型答题模板_板屈曲与极限强度计算题",
    "11": "核心题型答题模板_矩阵梁格夹层综合题",
}


def _is_formula_file(filename: str) -> bool:
    """判断是否是公式密集型文件（需要更小的 chunk）"""
    formula_files = {"03", "04", "10", "11"}
    prefix = filename[:2]
    return prefix in formula_files


def chunk_review_markdowns() -> List[Document]:
    """对所有中文复习笔记按 Markdown 标题层级分块"""
    md_files = sorted(REVIEW_MD_DIR.glob("*.md"))
    all_docs = []

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for md_path in md_files:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        prefix = md_path.stem[:2]
        topic = FILE_TOPIC_MAP.get(prefix, md_path.stem)

        # 用 Markdown 标题切分
        sections = md_splitter.split_text(content)

        # 对公式文件二次细切
        if _is_formula_file(md_path.stem):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=FORMULA_CHUNK_SIZE,
                chunk_overlap=FORMULA_CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            )
            sections = text_splitter.split_documents(sections)

        # 对于其他文件，如果 section 太长，也切一下
        else:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=LECTURE_CHUNK_SIZE,
                chunk_overlap=LECTURE_CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""],
            )
            sections = text_splitter.split_documents(sections)

        # 丰富元数据
        for doc in sections:
            h1 = doc.metadata.get("h1", "")
            h2 = doc.metadata.get("h2", "")
            h3 = doc.metadata.get("h3", "")
            breadcrumb = " > ".join(filter(None, [h1, h2, h3]))

            doc.metadata.update({
                "source_type": "review_md",
                "source_file": md_path.name,
                "file_topic": topic,
                "breadcrumb": breadcrumb,
                "language": "zh",
                "is_formula_file": _is_formula_file(md_path.stem),
            })

        all_docs.extend(sections)
        print(f"  {md_path.name}: {len(sections)} 个chunk")

    print(f"复习笔记总计: {len(all_docs)} 个 chunk")
    return all_docs
