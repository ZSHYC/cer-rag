"""测试 Agentic RAG —— LangGraph 多智能体"""
import sys
sys.path.insert(0, '/home/zshyc/cer')
from dotenv import load_dotenv; load_dotenv('/home/zshyc/cer/.env')

from rag.graph.build_graph import build_tutor_graph, classify_query

print("Compiling graph...")
graph = build_tutor_graph()
print(f"Nodes: {list(graph.nodes.keys())}")

queries = [
    "What is Rational Ship Structural Design?",
    "计算正方形截面的形状系数",
]

for q in queries:
    qtype = classify_query(q)
    print(f"\n{'='*60}")
    print(f"Q [{qtype}]: {q}")
    print("="*60)

    state = {
        "query": q, "chat_history": [], "query_type": qtype,
        "retrieval_attempts": 0, "retrieval_results": [],
        "retrieval_sufficient": False,
        "plan": [], "current_step": 0, "next_agent": "",
    }

    for step in graph.stream(state, {"recursion_limit": 20}):
        node = list(step.keys())[0]
        data = step[node]
        plan = data.get("plan", [])
        attempts = data.get("retrieval_attempts", 0)
        ndocs = len(data.get("retrieval_results", []))
        print(f"  [{node}] plan={plan[:2]} docs={ndocs} retries={attempts}")
        if node in ("concept", "calc"):
            answer = data.get("final_answer", "")[:300]
            print(f"  Answer preview: {answer}...")

    print()
