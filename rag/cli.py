#!/usr/bin/env python3
"""Marine Structures AI Tutor — 命令行交互入口"""
import sys
import os
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from rag.config import (
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
    DENSE_K, SPARSE_K, RERANK_TOP_K, CHUNKS_DIR, MAX_CONTEXT_TOKENS,
)
from rag.prompts.templates import SYSTEM_PROMPT
from rag.indexing.vector_index import load_vector_index, dense_search
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.fusion import hybrid_search, rerank_with_cross_encoder

console = Console()


def load_llm():
    api_key = LLM_API_KEY or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        console.print("[red]请设置 DEEPSEEK_API_KEY 环境变量[/red]")
        sys.exit(1)
    return ChatOpenAI(
        model=LLM_MODEL, api_key=api_key, base_url=LLM_BASE_URL,
        temperature=0.3, streaming=True,
    )


def load_sparse_index():
    """从预构建的 pickle 文件加载 BM25"""
    pickle_path = CHUNKS_DIR / "bm25_index.pkl"
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            docs = pickle.load(f)
        return SparseIndex(docs)
    return None


def format_source(doc) -> str:
    meta = doc.metadata
    st = meta.get("source_type", "")
    if st == "lecture":
        lec = meta.get("lecture_number", "?")
        sr = meta.get("slide_range") or meta.get("slide_num", "?")
        return f"[Lecture {lec}, Slide {sr}]"
    if st == "exam":
        return f"[Exam {meta.get('exam_year','?')} Q{meta.get('question_number','?')}]"
    if st == "solution":
        return f"[Answer {meta.get('exam_year','?')} Q{meta.get('question_number','?')}{meta.get('sub_question','')}]"
    if st == "seminar":
        return f"[Tutorial {meta.get('tutorial_number','?')}]"
    if st == "review_md":
        return f"[Review: {meta.get('source_file','?')}]"
    return f"[{meta.get('source_file','?')}]"


def build_context(docs_with_scores, max_tokens=MAX_CONTEXT_TOKENS):
    parts = []
    est = 0
    for doc, score in docs_with_scores:
        block = f"### {format_source(doc)}\n{doc.page_content}\n"
        e = len(block) // 3
        if est + e > max_tokens:
            break
        parts.append(block)
        est += e
    return "\n---\n".join(parts)


def classify_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ['calculate', 'compute', 'solve', 'find', 'determine', 'derive',
                              'shape factor', 'modulus', 'buckling stress', 'stiffness',
                              '计算', '求', '算', '求解', '推导']):
        return "calculation"
    if any(k in q for k in ['what is', 'describe', 'explain', 'compare', 'why', 'define',
                              'discuss', '是什么', '解释', '说明', '区别', '为什么']):
        return "conceptual"
    if any(k in q for k in ['formula', 'equation', 'expression', '公式', '表达式']):
        return "formula"
    if any(k in q for k in ['which year', 'which exam', 'which lecture',
                              '哪年', '哪个课件', '哪道题']):
        return "reference"
    return "conceptual"


def run():
    console.print(Panel.fit(
        "[bold cyan]Marine Structures AI Tutor[/bold cyan]\n"
        "[dim]SESS3026/JEIS3005 — Agentic RAG 考试导师[/dim]\n\n"
        "[yellow]:q[/yellow] 退出  [yellow]:s[/yellow] 切换来源显示",
        title="Welcome"
    ))

    # --- Load ---
    console.print("[dim]加载向量索引...[/dim]")
    vs = load_vector_index()
    console.print(f"  ChromaDB: {vs._collection.count()} 条")

    console.print("[dim]加载 BM25 索引...[/dim]")
    si = load_sparse_index()
    if si is None:
        console.print("[yellow]BM25 索引未找到，仅用稠密检索[/yellow]")
    else:
        console.print(f"  BM25: {len(si.documents)} 篇")

    console.print("[dim]加载 LLM...[/dim]")
    llm = load_llm()
    console.print(f"  Model: {LLM_MODEL}")

    console.print("[dim]加载 Reranker...[/dim]")
    reranker = None
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
        console.print("  Reranker: OK")
    except Exception:
        console.print("  [yellow]跳过 (加载失败)[/yellow]")

    console.print("\n[green]Ready![/green]\n")

    history = []
    show_src = True

    while True:
        try:
            q = console.input("[bold green]You: [/bold green]")
        except (EOFError, KeyboardInterrupt):
            break
        q = q.strip()
        if not q:
            continue
        if q == ":q":
            break
        if q == ":s":
            show_src = not show_src
            console.print(f"[dim]Show sources: {show_src}[/dim]")
            continue

        qtype = classify_query(q)
        console.print(f"[dim]{qtype} · searching...[/dim]")

        # Search
        def _dense(qq, k, filter_dict=None):
            return dense_search(vs, qq, k=k, filter_dict=filter_dict)
        def _sparse(qq, k):
            return si.search_with_scores(qq, k=k) if si else []

        fused = hybrid_search(_dense, _sparse, q, dense_k=DENSE_K, sparse_k=SPARSE_K)

        if reranker and len(fused) > RERANK_TOP_K:
            fused = rerank_with_cross_encoder(q, fused, reranker, RERANK_TOP_K)

        ctx = build_context(fused)
        if not ctx:
            console.print("[yellow]No relevant documents found.[/yellow]")
            continue

        # Generate
        prompt = SYSTEM_PROMPT.format(context=ctx)
        msgs = [SystemMessage(content=prompt)]
        for h in history[-10:]:
            msgs.append(HumanMessage(content=h["q"]))
            msgs.append(AIMessage(content=h["a"]))
        msgs.append(HumanMessage(content=q))

        console.print("[bold blue]Tutor: [/bold blue]", end="")
        full = ""
        for chunk in llm.stream(msgs):
            c = chunk.content if hasattr(chunk, 'content') else str(chunk)
            console.print(c, end="")
            full += c
        console.print()

        # Sources
        if show_src:
            t = Table(title="Sources", show_header=True, header_style="dim",
                      title_style="dim")
            t.add_column("#", style="dim", width=3)
            t.add_column("Source", style="cyan")
            t.add_column("Preview", style="dim", max_width=80)
            for i, (doc, _) in enumerate(fused[:5], 1):
                pv = doc.page_content[:80].replace('\n', ' ') + "..."
                t.add_row(str(i), format_source(doc), pv)
            console.print(t)

        history.append({"q": q, "a": full})

    console.print("\n[dim]Goodbye & good luck![/dim]")


if __name__ == "__main__":
    run()
