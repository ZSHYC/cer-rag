"""Retrieval Agent —— 自主检索 + 自省重试 (Self-RAG)"""
from langchain_core.messages import SystemMessage, HumanMessage
from rag.agents.state import TutorState
from rag.agents.tools import do_hybrid_search, docs_to_context, get_llm
from rag.config import MAX_RETRY

RETRIEVAL_SYSTEM = """You are the Retrieval Agent. You search across all course materials to find the most relevant information for the student's question.

Available sources:
- Lecture slides (Lectures 1-21, English)
- Past exam papers (2014-2025, English)
- Answer keys (2022-2025, English)
- Tutorial solutions (Tutorials 1-4, English)
- Chinese review notes (12 curated markdown files, bilingual)

After performing the search, evaluate whether the retrieved documents are sufficient to answer the question.

Output format (JSON only):
{
  "sufficient": true/false,
  "missing_info": "what's missing if not sufficient",
  "suggested_requery": "alternative search query if needed"
}
"""


def retrieval_node(state: TutorState) -> dict:
    """Retrieval Agent 节点：执行检索 + 自省 + 必要时重试"""
    query = state["query"]
    attempts = state.get("retrieval_attempts", 0)
    all_docs = state.get("retrieval_results", [])
    llm = get_llm()

    # 按查询类型偏好来源
    qtype = state.get("query_type", "conceptual")
    source_pref = None
    if qtype == "calculation":
        source_pref = None  # 搜索全部，因为答案和复习笔记都有
    elif qtype == "reference":
        source_pref = None

    # 执行混合检索
    fused = do_hybrid_search(query, source_type=source_pref)
    docs = [doc for doc, _ in fused]

    if not docs:
        return {"retrieval_sufficient": False, "retrieval_attempts": attempts + 1,
                "error": "No documents found"}

    # 去重合并
    seen = set()
    unique_docs = []
    for doc in docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    all_docs.extend(unique_docs)

    # 检索自省
    if attempts < MAX_RETRY:
        ctx_summary = docs_to_context(fused[:5], max_tokens=2000)

        reflection_msg = f"""Query: {query}

Retrieved {len(unique_docs)} documents. Here are summaries of the top results:

{ctx_summary[:1500]}

Are these documents sufficient to fully answer the question?
If not, what is missing and how should we re-search?"""

        messages = [
            SystemMessage(content=RETRIEVAL_SYSTEM),
            HumanMessage(content=reflection_msg),
        ]
        response = llm.invoke(messages, temperature=0.1)
        content = response.content.strip()

        import json
        try:
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            reflection = json.loads(content.strip())
        except json.JSONDecodeError:
            reflection = {"sufficient": True, "missing_info": "", "suggested_requery": ""}

        if not reflection.get("sufficient") and reflection.get("suggested_requery"):
            # 用改写后的查询再搜一次
            requery = reflection["suggested_requery"]
            more_fused = do_hybrid_search(requery)
            more_docs = [doc for doc, _ in more_fused]
            for doc in more_docs:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)
    else:
        reflection = {"sufficient": True, "missing_info": ""}

    return {
        "retrieval_results": all_docs,
        "retrieval_sufficient": reflection.get("sufficient", True),
        "retrieval_attempts": attempts + 1,
    }
