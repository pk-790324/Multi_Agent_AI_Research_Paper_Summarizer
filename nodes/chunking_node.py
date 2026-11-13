
from pathlib import Path

from state.state import ResearchPaperState
from services.chunking_service import ChunkingService


CHUNK_SIZE=1800
OVERLAP=200

def chunking_node(
    state: ResearchPaperState
) -> ResearchPaperState:

    print("=" * 60)
    print("CHUNKING NODE")
    print("=" * 60)

    artifacts = state.get("artifacts", {})

    try:

        # 1. Get Markdown path
        markdown_path = artifacts.get("markdown_path")

        if not markdown_path:
            raise ValueError(
                "markdown_path is missing from state artifacts."
            )

        # 2. Get paper title
        paper_title = artifacts.get("paper_title")

        if not paper_title:
            paper_title = Path(
                markdown_path
            ).stem.replace("_", " ")

        # 3. Generate chunks with page metadata
        chunks, chunk_file = ChunkingService.section_aware_chunking(
    markdown_path=markdown_path,
    paper_title=paper_title,
    user_id=state["user_id"],
    chunk_size=1800,
    overlap=200
)

        print(f"Paper title: {paper_title}")
        print(f"Generated chunks: {len(chunks)}")
        print(f"Chunks saved to: {chunk_file}")

        # 4. Update state
        return {
            **state,

            "artifacts": {
                **artifacts,

                "paper_title": paper_title,

                "chunks": chunks,

                "chunk_file": chunk_file,
                "chunk_size":CHUNK_SIZE,
                "chunk_overlap":OVERLAP
            },

            "error": "",
        }

    except Exception as exc:

        print(f"Chunking failed: {exc}")

        return {
            **state,
            "error": f"Chunking failed: {exc}",
        }
