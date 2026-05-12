"""端到端 CLI 测试（非交互）"""
import sys, pickle
sys.path.insert(0, '/home/zshyc/cer')
from dotenv import load_dotenv; load_dotenv('/home/zshyc/cer/.env')
from langchain_core.messages import SystemMessage, HumanMessage

from rag.config import DENSE_K, SPARSE_K, RERANK_TOP_K, CHUNKS_DIR
from rag.prompts.templates import SYSTEM_PROMPT
from rag.indexing.vector_index import load_vector_index, dense_search
from rag.indexing.sparse_index import SparseIndex
from rag.indexing.fusion import hybrid_search, rerank_with_cross_encoder
from langchain_openai import ChatOpenAI

# Load
print("Loading...")
vs = load_vector_index()
with open(CHUNKS_DIR / "bm25_index.pkl", 'rb') as f:
    si = SparseIndex(pickle.load(f))
llm = ChatOpenAI(model="deepseek-v4-pro", api_key="sk-fff8dfb329b24b7dbb204bdf4c1e41c6",
                  base_url="https://api.deepseek.com/v1", temperature=0.3)

# Test queries
queries = [
    "What is Rational Ship Structural Design?",
    "板屈曲的临界应力怎么算？",
    "How to calculate the shape factor of a square cross-section?",
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Q: {q}")

    # Search
    def _d(qq, k, filter_dict=None): return dense_search(vs, qq, k=k, filter_dict=filter_dict)
    def _s(qq, k): return si.search_with_scores(qq, k=k)
    fused = hybrid_search(_d, _s, q, dense_k=DENSE_K, sparse_k=SPARSE_K)

    print(f"Retrieved {len(fused)} docs:")
    for i, (doc, sc) in enumerate(fused[:3]):
        src = doc.metadata.get('source_type', '?')
        fn = doc.metadata.get('source_file', '?')
        print(f"  {i+1}. [{src}] {fn} ({sc:.3f}): {doc.page_content[:80]}...")

    # Generate
    ctx_parts = []
    for doc, _ in fused[:5]:
        ctx_parts.append(f"### [{doc.metadata.get('source_type','?')}] {doc.metadata.get('source_file','?')}\n{doc.page_content}")
    ctx = "\n---\n".join(ctx_parts)[:6000]

    prompt = SYSTEM_PROMPT.format(context=ctx)
    msgs = [SystemMessage(content=prompt), HumanMessage(content=q)]

    resp = llm.invoke(msgs)
    print(f"\nA: {resp.content[:500]}...")
