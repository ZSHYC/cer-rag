"""快速测试检索"""
import sys, time
sys.path.insert(0, '/home/zshyc/cer')
from dotenv import load_dotenv
load_dotenv('/home/zshyc/cer/.env')

from rag.indexing.vector_index import load_vector_index, dense_search

print('Loading index...')
t0 = time.time()
vs = load_vector_index()
print(f'Loaded {vs._collection.count()} docs in {time.time()-t0:.1f}s')

print('Searching...')
t0 = time.time()
results = dense_search(vs, 'What is Rational Ship Structural Design?', k=5)
print(f'Search took {time.time()-t0:.1f}s')

for i, (doc, score) in enumerate(results):
    src = doc.metadata.get('source_type', '?')
    fn = doc.metadata.get('source_file', '?')
    print(f'  {i+1}. [{src}] {fn} | score={score:.3f}')
    print(f'     {doc.page_content[:120]}...')
