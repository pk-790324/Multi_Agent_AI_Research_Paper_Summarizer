
from state.state import ResearchPaperState

from services.paper_metadata_service import (
    PaperMetadataService
)


def metadata_node(state: ResearchPaperState) -> ResearchPaperState:

    print("=" * 60)
    print("PAPER METADATA NODE")
    print("=" * 60)

    artifacts = state.get("artifacts", {})

    try:

        markdown_path = artifacts.get("markdown_path")

        metadata = PaperMetadataService.extract(
            markdown_path
        )

        print(f"Paper title: {metadata['paper_title']}")
        print(f"DOI: {metadata['doi']}")

        return {
            **state,
            "artifacts": {
                **artifacts,
                **metadata,
            },
            "error": "",
        }

    except Exception as exc:

        print(f"Metadata extraction failed: {exc}")

        return {
            **state,
            "error": f"Metadata extraction failed: {exc}",
        }
