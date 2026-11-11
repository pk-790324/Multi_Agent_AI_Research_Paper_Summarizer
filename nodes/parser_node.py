from state.state import ResearchPaperState
from services.parser_service import ParserService



def parser_node(state: ResearchPaperState) -> ResearchPaperState:

    print("=" * 60)
    print("PARSER NODE")
    print("=" * 60)

    artifacts = state.get("artifacts", {})
    file_path = artifacts.get("file_path")

    if not file_path:
        return {
            **state,
            "error": "Uploaded file path is missing."
        }

    try:
        ingestion_id = artifacts.get("document_id")

        markdown_path = ParserService.parse_pdf(
            pdf_path=file_path,
            ingestion_id=artifacts["ingestion_id"]
        )

        print(f"Markdown saved: {markdown_path}")

        return {
            **state,
            "artifacts": {
                **artifacts,
                "markdown_path": markdown_path,
                "ingestion_id": ingestion_id,
            },
            "error": "",
        }

    except Exception as exc:
        print(f"Parsing failed: {exc}")

        return {
            **state,
            "error": f"Parsing failed: {exc}",
        }
