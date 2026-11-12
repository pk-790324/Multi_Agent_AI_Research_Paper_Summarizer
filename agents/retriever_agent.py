
from services.hybrid_search_service import HybridSearchService


# Initialize hybrid search service
hybrid_search = HybridSearchService()


def retriever_agent(state):
    """
    Retrieve relevant research-paper chunks belonging
    only to the authenticated user.

    Hybrid search:
    Dense + Sparse + RRF + Cross-Encoder.
    """

    print("\n" + "=" * 60)
    print("RETRIEVER AGENT")
    print("=" * 60)

    # -------------------------------------------------
    # 1. GET USER QUERY
    # -------------------------------------------------

    query = (state.get("user_query") or "").strip()

    if not query:
        return {
            "retrieved_chunks": [],
            "retrieval_status": "failed",
            "retrieval_error": "User query is empty."
        }

    # -------------------------------------------------
    # 2. GET AUTHENTICATED USER ID
    # -------------------------------------------------

    user_id = state.get("user_id")

    if not isinstance(user_id, str) or not user_id.strip():

        return {
            "retrieved_chunks": [],
            "retrieval_status": "failed",
            "retrieval_error": "Authenticated user_id is missing."
        }

    user_id = user_id.strip()

    print(f"\nUser ID: {user_id}")
    print(f"User Query: {query}")

    # -------------------------------------------------
    # 3. HYBRID SEARCH CONFIGURATION
    # -------------------------------------------------

    TOP_K = 5
    CANDIDATE_K = 20

    print("\nSearch Configuration:")
    print(f"Top K: {TOP_K}")
    print(f"Candidate K: {CANDIDATE_K}")
    print("Search Mode: Hybrid (Dense + Sparse + RRF)")
    print("Document Filter: Authenticated User ID")

    # -------------------------------------------------
    # 4. RETRIEVE USER-SPECIFIC CHUNKS
    # -------------------------------------------------

    try:

        chunks = hybrid_search.search(
            query=query,
            user_id=user_id,
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
    # 5. CHECK RETRIEVAL RESULTS
    # -------------------------------------------------

    if not chunks:

        print("\nNo relevant chunks found for this user.")

        return {
            "retrieved_chunks": [],
            "retrieval_status": "empty",
            "retrieval_error": None
        }

    # -------------------------------------------------
    # 6. DISPLAY RETRIEVED CHUNKS
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
    # 7. RETURN RESULTS TO LANGGRAPH STATE
    # -------------------------------------------------

    return {
        "retrieved_chunks": chunks,
        "retrieval_status": "success",
        "retrieval_error": None
    }






