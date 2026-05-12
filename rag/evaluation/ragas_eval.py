"""RAGAS 评估 + 检索策略 A/B 对比"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from rag.graph.build_graph import build_tutor_graph, classify_query


# 测试用例：历年真题作为 ground truth
TEST_CASES = [
    {
        "query": "What is Rational Ship Structural Design?",
        "ground_truth": "A design process directly based on structural theory, computer-based structural analysis and optimization, aiming to achieve the optimum structure according to a designer-selected measure of merit.",
        "query_type": "conceptual",
    },
    {
        "query": "Calculate the shape factor for a square cross-section of side a",
        "ground_truth": "Z = a^3/6, Zp = a^3/4, shape factor φ = Zp/Z = 1.5",
        "query_type": "calculation",
    },
    {
        "query": "板屈曲的临界应力怎么算？",
        "ground_truth": "σcr = (k·π²·E)/[12(1-ν²)]·(t/b)²",
        "query_type": "calculation",
    },
    {
        "query": "List three main disadvantages of the traditional prescriptive Rulebook approach",
        "ground_truth": "1. Cannot handle novel designs outside rule scope 2. Over-conservative weight estimation 3. Does not optimize for specific performance criteria",
        "query_type": "conceptual",
    },
    {
        "query": "What is Faulkner's effective width equation and how is it used?",
        "ground_truth": "be = b·(σcr/σy) or be/b = 2/β - 1/β² where β = √(σy/σcr). Used to account for post-buckling strength in plates.",
        "query_type": "conceptual",
    },
]


def run_evaluation(graph=None) -> list:
    """运行 RAGAS 评估，返回每个测试用例的结果"""
    if graph is None:
        graph = build_tutor_graph(enable_critic=False)

    results = []
    for tc in TEST_CASES:
        state = {
            "query": tc["query"], "chat_history": [],
            "query_type": tc["query_type"], "retrieval_attempts": 0,
            "retrieval_results": [], "retrieval_sufficient": False,
            "plan": [], "current_step": 0, "next_agent": "",
        }

        final_state = None
        for step in graph.stream(state, {"recursion_limit": 20}):
            final_state = step

        if final_state is None:
            results.append({**tc, "answer": "", "retrieved_docs": 0, "error": "No output"})
            continue

        node_name = list(final_state.keys())[0]
        node_data = final_state[node_name]
        answer = node_data.get("final_answer", "") or node_data.get("concept_answer", "")
        docs = node_data.get("retrieval_results", [])

        results.append({
            "query": tc["query"],
            "ground_truth": tc["ground_truth"],
            "answer": answer[:500],
            "retrieved_docs": len(docs),
            "plan": node_data.get("plan", []),
            "retrieval_attempts": node_data.get("retrieval_attempts", 0),
        })

    return results


def compare_strategies():
    """A/B 对比：不同的检索策略"""
    print("RAG Strategy Comparison")
    print("=" * 60)

    results = run_evaluation()

    total_attempts = sum(r["retrieval_attempts"] for r in results)
    avg_docs = sum(r["retrieved_docs"] for r in results) / len(results) if results else 0

    print(f"Test cases: {len(results)}")
    print(f"Avg retrieved docs: {avg_docs:.1f}")
    print(f"Avg retrieval attempts (Self-RAG): {total_attempts/len(results):.1f}")
    print()

    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['query'][:60]}...")
        print(f"      Docs: {r['retrieved_docs']} | Retries: {r['retrieval_attempts']}")
        print(f"      Answer: {r['answer'][:120]}...")
        print()

    return results


if __name__ == "__main__":
    compare_strategies()
