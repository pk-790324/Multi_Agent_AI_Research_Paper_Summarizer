from state.state import ResearchPaperState
from services.parser_service import ParserService


def parser_node(state: ResearchPaperState) -> ResearchPaperState:
    """
    Parse the downloaded PDF and update the graph state.
    """

    print("=" * 60)
    print("PARSER NODE")
    print("=" * 60)

    pdf_path = state["artifacts"]["pdf_path"]

    print(f"Parsing: {pdf_path}")

    markdown_path = ParserService.parse_pdf(pdf_path)

    print(f"Markdown saved to: {markdown_path}")

    return {
        **state,
        "artifacts": {
            **state["artifacts"],
            "markdown_path": markdown_path,
        },
    }