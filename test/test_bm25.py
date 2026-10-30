
from services.bm25_service import BM25Service

# Replace this path with the actual JSON file produced
# by your existing chunking node.
CHUNK_FILE = "chunks/rules_of_ml_chunks.json"

bm25 = BM25Service()

result = bm25.build_index(CHUNK_FILE)
print("Index result:", result)
query="explain rule 5 of machine learning"
print("query:",query)
results = bm25.search(
    query=query,
    top_k=5,
)

for i, chunk in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Paper:", chunk.get("paper_title"))
    print("Section:", chunk.get("section"))
    print("Chunk ID:", chunk.get("chunk_id"))
    print("BM25 score:", chunk["bm25_score"])
    print("Text:", chunk["text"][:500])
