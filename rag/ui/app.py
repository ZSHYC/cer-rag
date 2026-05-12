"""Gradio Web UI — compatible with Gradio 6.0 message format"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import gradio as gr
from rag.graph.build_graph import build_tutor_graph, classify_query


def chat_fn(query: str, history: list, show_debug: bool):
    """Handle chat turn — Gradio 6.0 message format"""
    if not query.strip():
        yield history, ""
        return

    graph = build_tutor_graph(enable_critic=False)
    qtype = classify_query(query)

    # Gradio 6.0 history is list of {"role": "user"/"assistant", "content": "..."}
    chat_history = []
    for msg in history[-20:]:  # last 10 turns = 20 messages
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            chat_history.append({"query": content, "answer": ""})
        elif role == "assistant" and chat_history:
            chat_history[-1]["answer"] = content

    state = {
        "query": query, "chat_history": chat_history,
        "query_type": qtype, "retrieval_attempts": 0,
        "retrieval_results": [], "retrieval_sufficient": False,
        "plan": [], "current_step": 0, "next_agent": "",
    }

    debug_info = []
    final_state = None
    try:
        for step in graph.stream(state, {"recursion_limit": 20}):
            node = list(step.keys())[0]
            data = step[node]
            plan = data.get("plan", [])
            ndocs = len(data.get("retrieval_results", []))
            retries = data.get("retrieval_attempts", 0)
            debug_info.append(f"[{node}] plan={plan[:1]} docs={ndocs} retries={retries}")
            final_state = step
    except Exception as e:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        yield history, ""
        return

    if final_state is None:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": "No response generated."})
        yield history, ""
        return

    node_name = list(final_state.keys())[0]
    node_data = final_state[node_name]
    answer = node_data.get("final_answer", "")
    if not answer:
        answer = node_data.get("concept_answer", "") or node_data.get("calc_result", {}).get("final_value", "")

    # Build source citations
    docs = node_data.get("retrieval_results", [])
    src_lines = []
    for i, doc in enumerate(docs[:5], 1):
        from rag.agents.tools import format_source
        src_lines.append(f"{i}. {format_source(doc)}")

    full_answer = answer
    if src_lines:
        full_answer += "\n\n---\n**Sources:**\n" + "\n".join(src_lines)
    if show_debug:
        full_answer += "\n\n---\n**Debug:**\n" + "\n".join(debug_info)

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": full_answer})
    yield history, ""


def build_ui():
    with gr.Blocks(title="Marine Structures AI Tutor") as demo:
        gr.Markdown("""
        # Marine Structures AI Tutor
        **SESS3026/JEIS3005 — Agentic RAG Multi-Agent Exam Tutoring System**
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Conversation")
                with gr.Row():
                    query_box = gr.Textbox(
                        placeholder="Ask anything about Marine Structures... (English or Chinese)",
                        scale=4, container=False, show_label=False)
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            with gr.Column(scale=1):
                debug_toggle = gr.Checkbox(label="Debug Mode", value=False)
                gr.Markdown("""
                **Example Questions:**
                - What is Rational Ship Structural Design?
                - 板屈曲的临界应力怎么算？
                - Calculate shape factor of square section
                - 这次考试的考点是啥？
                """)

        send_btn.click(
            fn=chat_fn, inputs=[query_box, chatbot, debug_toggle],
            outputs=[chatbot, query_box],
        )
        query_box.submit(
            fn=chat_fn, inputs=[query_box, chatbot, debug_toggle],
            outputs=[chatbot, query_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
