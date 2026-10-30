
from services.hybrid_search_service import HybridSearchService


# Initialize hybrid search service
hybrid_search = HybridSearchService()


def retriever_agent(state):
    """
    Retrieve relevant research-paper chunks using
    hybrid search (Dense + Sparse + RRF).
    """

    print("\n" + "=" * 60)
    print("RETRIEVER AGENT")
    print("=" * 60)

    # -------------------------------------------------
    # 1. GET USER QUERY
    # -------------------------------------------------

    query = state.get("user_query", "").strip()

    if not query:
        return {
            "retrieved_chunks": [],
            "retrieval_status": "failed",
            "retrieval_error": "User query is empty."
        }

    print(f"\nUser Query: {query}")

    # -------------------------------------------------
    # 2. HYBRID SEARCH CONFIGURATION
    # -------------------------------------------------

    TOP_K = 5
    CANDIDATE_K = 20

    print("\nSearch Configuration:")
    print(f"Top K: {TOP_K}")
    print(f"Candidate K: {CANDIDATE_K}")
    print("Search Mode: Hybrid (Dense + Sparse + RRF)")

    # -------------------------------------------------
    # 3. RETRIEVE RELEVANT CHUNKS
    # -------------------------------------------------

    try:

        chunks = hybrid_search.search(
            query=query,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
        )

    except Exception as e:

        print(f"\nRetrieval Error: {str(e)}")

        return {
            "retrieved_chunks": [],
            "retrieval_status": "failed",
            "retrieval_error": str(e)
        }

    # -------------------------------------------------
    # 4. CHECK RETRIEVAL RESULTS
    # -------------------------------------------------

    if not chunks:

        print("\nNo relevant chunks found.")

        return {
            "retrieved_chunks": [],
            "retrieval_status": "empty",
            "retrieval_error": None
        }

    # -------------------------------------------------
    # 5. DISPLAY RETRIEVED CHUNKS
    # -------------------------------------------------

    print(f"\nRetrieved {len(chunks)} chunks successfully.")

    for chunk in chunks:

        print("\n" + "-" * 60)

        print(f"Rank: {chunk.get('rank')}")
        print(f"Score: {chunk.get('score')}")
        print(f"Paper: {chunk.get('paper_title')}")
        print(f"Section: {chunk.get('section')}")
        print(f"Chunk ID: {chunk.get('chunk_id')}")

        print("\nContent:")
        print(chunk.get("page_content", "")[:300])

    # -------------------------------------------------
    # 6. RETURN RESULTS TO LANGGRAPH STATE
    # -------------------------------------------------

    return {
        "retrieved_chunks": chunks,
        "retrieval_status": "success",
        "retrieval_error": None
    }









