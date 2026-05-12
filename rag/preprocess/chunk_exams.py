"""试卷与答案分块——按题目/子题边界切分"""
import re
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from rag.config import EXTRACTED_DIR
from rag.preprocess.normalize_txt import normalize_file


def parse_exam_year(text: str) -> str:
    """从试卷文本中提取学年"""
    # 2022/23, 2023/24 等格式
    match = re.search(r'SEMESTER \d+ (?:FINAL ASSESSMENT|EXAMINATIONS?)\s+(\d{4}[/-]\d{2,4})', text)
    if match:
        return match.group(1)
    # 2014/15, 2016/17 等格式
    match = re.search(r'(?:FINAL ASSESSMENTS?|EXAMINATIONS?)\s+(\d{4}[/-]\d{2,4})', text)
    if match:
        return match.group(1)
    match = re.search(r'SEMESTER \d+ (?:EXAMINATIONS?|FINAL)\s+(\d{4}[/-]\d{2,4})', text)
    if match:
        return match.group(1)
    # 匹配下划线包围的年份
    match = re.search(r'_{3,}.*?(\d{4}[/-]\d{2,4}).*?_{3,}', text)
    if match:
        return match.group(1)
    return "unknown"


def split_exam_questions(text: str) -> List[dict]:
    """按题目边界切分试卷文本，兼容多种格式"""
    year = parse_exam_year(text)

    # 尝试多种切分模式
    # 模式1: N. [Total marks M] (2022-2025 格式)
    parts = re.split(r'(?=\d+\.\s+\[Total marks \d+\])', text)
    # 模式2: Question N (2014-2019 格式)
    if len(parts) <= 1:
        parts = re.split(r'(?=Question \d+)', text)
    # 模式3: N. (text)... [M marks] 提取
    if len(parts) <= 1:
        parts = re.split(r'(?=\d+\.\s+)', text)

    questions = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 50:
            continue

        # 提取题号（兼容多种格式）
        q_match = re.match(r'(\d+)\.\s+\[Total marks (\d+)\]', part)
        if q_match:
            q_num = int(q_match.group(1))
            marks = int(q_match.group(2))
        elif re.match(r'Question (\d+)', part):
            q_num = int(re.match(r'Question (\d+)', part).group(1))
            # 尝试从内容中提取分值
            marks_match = re.findall(r'\[(\d+)\s*marks?\]', part)
            marks = sum(int(m) for m in marks_match) if marks_match else 0
        elif re.match(r'(\d+)\.\s+', part):
            q_num = int(re.match(r'(\d+)\.\s+', part).group(1))
            marks_match = re.findall(r'\[(\d+)\s*marks?\]', part)
            marks = sum(int(m) for m in marks_match) if marks_match else 0
        else:
            continue  # 不是题目

        # 判断题型
        part_lower = part.lower()
        if any(kw in part_lower for kw in ['describe', 'explain', 'discuss', 'define', 'list', 'essay']):
            q_type = "conceptual"
        elif any(kw in part_lower for kw in ['calculate', 'determine', 'derive', 'compute', 'solve', 'find']):
            q_type = "calculation"
        else:
            q_type = "mixed"

        questions.append({
            "year": year,
            "question_number": q_num,
            "total_marks": marks,
            "content_type": q_type,
            "text": part,
        })

    return questions


def split_solution_sections(text: str) -> List[dict]:
    """按子题边界切分答案文本"""
    year = parse_exam_year(text)

    # 按 "Q1", "Q2" 或 "(i)", "(ii)" 切分
    parts = re.split(r'\n(?=Q\d+|\(\w+\))', text)

    sections = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 50:
            continue

        q_match = re.match(r'Q(\d+)', part)
        sub_match = re.match(r'\((\w+)\)', part)

        sections.append({
            "year": year,
            "question_number": int(q_match.group(1)) if q_match else 0,
            "sub_question": sub_match.group(1) if sub_match else "",
            "text": part,
        })

    return sections


def chunk_exam_papers(normalized_exam_path: str = None) -> List[Document]:
    """分块试卷"""
    if normalized_exam_path is None:
        normalized_exam_path = str(EXTRACTED_DIR / "normalized" / "exams_normalized.txt")

    # 确保已归一化
    src = Path("/home/zshyc/cer/extracted/exams.txt")
    dst = Path(normalized_exam_path)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = normalize_file(str(src), str(dst))
    else:
        with open(dst, 'r', encoding='utf-8') as f:
            text = f.read()

    questions = split_exam_questions(text)

    # 试卷合集里有多套卷，按年份切分
    year_blocks = re.split(r'(?=UNIVERSITY OF SOUTHAMPTON)', text)
    all_questions = []
    for block in year_blocks:
        block = block.strip()
        if not block or 'SEMESTER' not in block or len(block) < 100:
            continue
        qs = split_exam_questions(block)
        all_questions.extend(qs)

    if not all_questions:
        all_questions = questions

    docs = []
    for q in all_questions:
        docs.append(Document(
            page_content=q["text"],
            metadata={
                "source_type": "exam",
                "source_file": "试卷合集.pdf",
                "exam_year": q["year"],
                "question_number": q["question_number"],
                "total_marks": q["total_marks"],
                "content_type": q["content_type"],
                "language": "en",
            }
        ))

    print(f"试卷分块: {len(all_questions)} 道题目 -> {len(docs)} 个 chunk")
    return docs


def chunk_answer_keys(normalized_answer_path: str = None) -> List[Document]:
    """分块答案"""
    if normalized_answer_path is None:
        normalized_answer_path = str(EXTRACTED_DIR / "normalized" / "answers_normalized.txt")

    src = Path("/home/zshyc/cer/extracted/answers_22_25.txt")
    dst = Path(normalized_answer_path)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = normalize_file(str(src), str(dst))
    else:
        with open(dst, 'r', encoding='utf-8') as f:
            text = f.read()

    # 按年份和题目切分
    sections = split_solution_sections(text)
    docs = []
    for s in sections:
        docs.append(Document(
            page_content=s["text"],
            metadata={
                "source_type": "solution",
                "source_file": "答案合集（22-25）.pdf",
                "exam_year": s["year"],
                "question_number": s["question_number"],
                "sub_question": s["sub_question"],
                "language": "en",
            }
        ))

    print(f"答案分块: {len(sections)} 个 section -> {len(docs)} 个 chunk")
    return docs


def chunk_seminar_answers(normalized_seminar_path: str = None) -> List[Document]:
    """分块研讨题解答"""
    if normalized_seminar_path is None:
        normalized_seminar_path = str(EXTRACTED_DIR / "normalized" / "seminar_normalized.txt")

    src = Path("/home/zshyc/cer/extracted/seminar_answers.txt")
    dst = Path(normalized_seminar_path)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = normalize_file(str(src), str(dst))
    else:
        with open(dst, 'r', encoding='utf-8') as f:
            text = f.read()

    # 按 Tutorial 标题切分
    parts = re.split(r'(?=Solutions to JEIS3005 Tutorial|^Tutorial\s*\d)', text, flags=re.IGNORECASE | re.MULTILINE)
    docs = []
    for part in parts:
        part = part.strip()
        if not part or len(part) < 100:
            continue

        # 提取 Tutorial 编号
        t_match = re.search(r'Tutorial\s*(\d+)', part, re.IGNORECASE)
        tutorial_num = int(t_match.group(1)) if t_match else 0

        # 按 Question N: 切分
        sub_parts = re.split(r'(?=Question\s*\d+)', part)
        for sub in sub_parts:
            sub = sub.strip()
            if not sub or len(sub) < 50:
                continue
            docs.append(Document(
                page_content=sub,
                metadata={
                    "source_type": "seminar",
                    "source_file": "研讨题答案合集.pdf",
                    "tutorial_number": tutorial_num,
                    "language": "en",
                }
            ))

    print(f"研讨题分块: {len(docs)} 个 chunk")
    return docs


def chunk_all_exams_and_answers() -> List[Document]:
    """分块所有试卷、答案、研讨题"""
    all_docs = []
    all_docs.extend(chunk_exam_papers())
    all_docs.extend(chunk_answer_keys())
    all_docs.extend(chunk_seminar_answers())
    print(f"试卷类总计: {len(all_docs)} 个 chunk")
    return all_docs
