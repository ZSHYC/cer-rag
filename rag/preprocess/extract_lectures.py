"""从课件 PDF 提取文本，保留幻灯片边界和元数据"""
import fitz  # PyMuPDF
import re
from pathlib import Path
from rag.config import LECTURE_DIR, EXTRACTED_DIR, LECTURE_TOPICS


def extract_lecture_number(filename: str) -> int:
    """从文件名提取课件编号: JEIS3005_Lecture_1.pdf -> 1"""
    match = re.search(r'Lecture_(\d+)', filename)
    return int(match.group(1)) if match else 0


def extract_slide_title(page_text: str) -> str:
    """尝试从幻灯片文本中提取标题（通常是第一行非空文本）"""
    lines = [l.strip() for l in page_text.split('\n') if l.strip()]
    if lines:
        # 过滤掉明显的页眉/页脚
        first = lines[0]
        if len(first) < 120 and not first.startswith(('JEIS', 'SESS', 'Copyright')):
            return first
    return ""


def extract_lecture(pdf_path: Path) -> list[dict]:
    """提取单个课件PDF的所有幻灯片

    返回: [{"lecture_num": 1, "slide_num": 1, "title": "...", "text": "..."}, ...]
    """
    lecture_num = extract_lecture_number(pdf_path.name)
    topic = LECTURE_TOPICS.get(lecture_num, "Unknown")
    doc = fitz.open(str(pdf_path))

    slides = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        title = extract_slide_title(text)

        slides.append({
            "lecture_num": lecture_num,
            "lecture_topic": topic,
            "slide_num": page_num + 1,
            "title": title,
            "text": text.strip(),
            "source_file": pdf_path.name,
        })

    doc.close()
    return slides


def extract_all_lectures(output_dir: Path = None) -> list[dict]:
    """提取所有课件 PDF"""
    if output_dir is None:
        output_dir = EXTRACTED_DIR / "lectures"

    output_dir.mkdir(parents=True, exist_ok=True)
    all_slides = []
    pdf_files = sorted(LECTURE_DIR.glob("*.pdf"))

    print(f"发现 {len(pdf_files)} 个课件 PDF")
    for pdf_path in pdf_files:
        slides = extract_lecture(pdf_path)
        all_slides.extend(slides)
        print(f"  Lecture {slides[0]['lecture_num'] if slides else '?'}: "
              f"{len(slides)} 页 - {slides[0]['lecture_topic'] if slides else ''}")

    print(f"总计: {len(all_slides)} 页")
    return all_slides


if __name__ == "__main__":
    slides = extract_all_lectures()
    # 打印第一页和最后一页作为样例
    if slides:
        print("\n--- 样例: 第一页 ---")
        print(f"Lecture {slides[0]['lecture_num']} Slide {slides[0]['slide_num']}")
        print(slides[0]['text'][:500])
        print("\n--- 样例: 最后一页 ---")
        print(f"Lecture {slides[-1]['lecture_num']} Slide {slides[-1]['slide_num']}")
        print(slides[-1]['text'][:500])
