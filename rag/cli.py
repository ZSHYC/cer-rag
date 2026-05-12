#!/usr/bin/env python3
"""Marine Structures AI Tutor — Agentic RAG CLI"""
import sys
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

from rag.config import LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, CHUNKS_DIR
from rag.graph.build_graph import build_tutor_graph, classify_query

console = Console()


def load_llm():
    return ChatOpenAI(
        model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
        temperature=0.3, streaming=True,
    )


def format_source(doc) -> str:
    meta = doc.metadata
    st = meta.get("source_type", "")
    if st == "lecture":
        return f"[Lecture {meta.get('lecture_number','?')}, Slide {meta.get('slide_range') or meta.get('slide_num','?')}]"
    if st == "exam":
        return f"[Exam {meta.get('exam_year','?')} Q{meta.get('question_number','?')}]"
    if st == "solution":
        return f"[Answer {meta.get('exam_year','?')} Q{meta.get('question_number','?')}{meta.get('sub_question','')}]"
    if st == "seminar":
        return f"[Tutorial {meta.get('tutorial_number','?')}]"
    if st == "review_md":
        return f"[Review: {meta.get('source_file','?')}]"
    return f"[{meta.get('source_file','?')}]"


def run():
    console.print(Panel.fit(
        "[bold cyan]Marine Structures AI Tutor[/bold cyan]\n"
        "[dim]SESS3026/JEIS3005 — Agentic RAG Multi-Agent System[/dim]\n\n"
        "[yellow]:q[/yellow] Quit   [yellow]:s[/yellow] Toggle sources   [yellow]:d[/yellow] Debug mode",
        title="Welcome"
    ))

    # Load graph
    console.print("[dim]Compiling LangGraph agent graph...[/dim]")
    graph = build_tutor_graph()
    console.print(f"  Nodes: {list(graph.nodes.keys())}")

    console.print("[dim]Loading LLM...[/dim]")
    console.print(f"  Model: {LLM_MODEL}")

    console.print("\n[green]Ready![/green]\n")

    history = []
    show_src = True
    debug = False

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
        if q == ":d":
            debug = not debug
            console.print(f"[dim]Debug mode: {debug}[/dim]")
            continue

        qtype = classify_query(q)
        console.print(f"[dim][{qtype}][/dim] ", end="")

        # 初始化状态
        initial_state = {
            "query": q,
            "chat_history": history[-10:],
            "query_type": qtype,
            "retrieval_attempts": 0,
            "retrieval_results": [],
            "retrieval_sufficient": False,
            "plan": [],
            "current_step": 0,
            "next_agent": "",
        }

        # 流式执行 LangGraph
        console.print("[dim]thinking...[/dim]")
        final_state = None
        try:
            for step_output in graph.stream(initial_state, {"recursion_limit": 20}):
                node_name = list(step_output.keys())[0]
                node_state = step_output[node_name]

                if debug:
                    plan = node_state.get("plan", [])
                    attempts = node_state.get("retrieval_attempts", 0)
                    ndocs = len(node_state.get("retrieval_results", []))
                    console.print(f"  [dim]│ [{node_name}] plan={plan[:2]} "
                                  f"docs={ndocs} retries={attempts}[/dim]")

                final_state = node_state
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

        if final_state is None:
            console.print("[yellow]No response generated.[/yellow]")
            continue

        answer = final_state.get("final_answer", "")
        if not answer:
            answer = final_state.get("concept_answer", "") or final_state.get("calc_result", {}).get("final_value", "")
        if not answer:
            console.print("[yellow]No answer generated. Try a different question.[/yellow]")
            continue

        console.print(f"\n[bold blue]Tutor: [/bold blue]")
        console.print(answer)

        # 显示来源和 Agent 执行信息
        if show_src:
            docs = final_state.get("retrieval_results", [])
            plan = final_state.get("plan", [])
            attempts = final_state.get("retrieval_attempts", 0)

            console.print()
            if plan:
                console.print(f"[dim]Plan: {' › '.join(plan)}[/dim]")
            if attempts > 1:
                console.print(f"[dim]Retrieval retries: {attempts-1}[/dim]")

            if docs:
                t = Table(title=f"Sources ({len(docs)} docs)", show_header=True,
                          header_style="dim", title_style="dim")
                t.add_column("#", style="dim", width=3)
                t.add_column("Source", style="cyan")
                t.add_column("Preview", style="dim", max_width=80)
                for i, doc in enumerate(docs[:5], 1):
                    pv = doc.page_content[:80].replace('\n', ' ') + "..."
                    t.add_row(str(i), format_source(doc), pv)
                console.print(t)

        console.print()
        history.append({"query": q, "answer": answer})

    console.print("\n[dim]Goodbye & good luck with your exam![/dim]")


if __name__ == "__main__":
    run()
