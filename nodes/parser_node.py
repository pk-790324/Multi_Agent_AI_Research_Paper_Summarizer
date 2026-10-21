from state.state import ResearchPaperState
from services.parser_service import ParserService


def parser_node(state: ResearchPaperState) -> ResearchPaperState:
    """
    Parse the saved document and update the graph state.
    """

    print("=" * 60)
    print("PARSER NODE")
    print("=" * 60)

    file_path = state["artifacts"]["file_path"]

    print(f"Parsing document: {file_path}")

    try:
        markdown_path = ParserService.parse_pdf(file_path)

        print(f"Markdown saved to: {markdown_path}")

        return {
            **state,
            "artifacts": {
                **state["artifacts"],
                "markdown_path": markdown_path,
            },
            "error": "",
        }

    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"Parsing failed: {exc}")

        return {
            **state,
            "error": f"Parsing failed: {exc}",
        }
