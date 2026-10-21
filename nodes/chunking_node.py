from state.state import ResearchPaperState
from services.chunking_service import ChunkingService
from pathlib import Path




def chunking_node(state: ResearchPaperState) -> ResearchPaperState:
    print("=" * 60)
    print("CHUNKING NODE")
    print("=" * 60)

    artifacts = state.get("artifacts", {})

    try:
        # Get the Markdown file created by the parser node
        markdown_path = artifacts.get("markdown_path")

        if not markdown_path:
            raise ValueError(
                "markdown_path is missing from state artifacts."
            )

        # Use the uploaded document's filename as the title
        paper_title = artifacts.get("paper_title")

        if not paper_title:
            paper_title = Path(markdown_path).stem.replace("_", " ")

        # Create section-based chunks
        chunks, chunk_file = ChunkingService.section_aware_chunking(
            markdown_path=markdown_path,
            paper_title=paper_title,
        )

        print(f"Paper title: {paper_title}")
        print(f"Generated chunks: {len(chunks)}")
        print(f"Chunks saved to: {chunk_file}")

        return {
            **state,
            "artifacts": {
                **artifacts,
                "paper_title": paper_title,
                "chunks": chunks,
                "chunk_file": chunk_file,
            },
            "error": "",
        }

    except Exception as exc:
        print(f"Chunking failed: {exc}")

        return {
            **state,
            "error": f"Chunking failed: {exc}",
        }
