
from services.hybrid_search_service import HybridSearchService

retriever = HybridSearchService()

query = (
    "What efficiency benefits do 1-bit LLMs offer compared with full-precision models?"
)

results = retriever.search(
    query=query,
    top_k=5,
    candidate_k=20,
)
print("query:",query)
for result in results:
    print("=" * 80)
    print("Rank:", result["rank"])
    print("RRF score:", result["score"])
    print("rerank score:", result["rerank_score"])
    print("Paper:", result["paper_title"])
    print("Section:", result["section"])
    print("Chunk ID:", result["chunk_id"])
    print("Content:", result["page_content"][:500])
