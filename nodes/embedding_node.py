from state.state import ResearchPaperState
from services.embedding_service import EmbeddingService



def embedding_node(
    state: ResearchPaperState,
) -> ResearchPaperState:
    """
    Generate embeddings and store them in Qdrant.
    """

    print("=" * 60)
    print("EMBEDDING NODE")
    print("=" * 60)

    artifacts = state.get("artifacts", {})
    chunk_file = artifacts["chunk_file"]

    result = EmbeddingService.index_chunks(
        chunk_file=chunk_file,
        collection_name=state.get(
            "collection_name",
            "research_papers",
        ),
    )

    print(
        f"Indexed {result['indexed_chunks']} chunks."
    )

    # Store embedding results inside artifacts.
    updated_artifacts = {
        **artifacts,
        "collection_name": result["collection_name"],
        "embedding": {
            "embedding_model": result["embedding_model"],
            "vector_dimension": result["vector_dimension"],
            "indexed_chunks": result["indexed_chunks"],
        },
    }

    return {
        **state,
        "artifacts": updated_artifacts,
    }
