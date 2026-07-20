from state.state import ResearchPaperState
from services.chunking_service import ChunkingService


def chunking_node(
    state: ResearchPaperState,
) -> ResearchPaperState:
    """
    Generate section-aware chunks from the parsed markdown
    and update the graph state.
    """

    print("=" * 60)
    print("CHUNKING NODE")
    print("=" * 60)

    markdown_path = state["artifacts"]["markdown_path"]

    paper_title = state["paper"][0]["title"]

    chunks, chunk_file = (
        ChunkingService.section_aware_chunking(
            markdown_path=markdown_path,
            paper_title=paper_title,
        )
    )

    print(f"Generated {len(chunks)} chunks")

    return {
        **state,
        "chunks": chunks,
        "artifacts": {
            **state["artifacts"],
            "chunk_file": chunk_file,
        },
    }