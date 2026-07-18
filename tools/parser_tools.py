
from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_core.tools import tool
from state.state import ResearchPaperState

from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_core.tools import tool

from state.state import ResearchPaperState


@tool
def parse_pdf(state: ResearchPaperState) -> ResearchPaperState:
    """
    Parse the downloaded PDF using Docling and save it as Markdown.
    """

    pdf_path = Path(state["artifacts"]["pdf_path"])

    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} does not exist.")

    converter = DocumentConverter()

    result = converter.convert(str(pdf_path))

    markdown = result.document.export_to_markdown()

    output_dir = Path("parsed_papers")
    output_dir.mkdir(exist_ok=True)

    markdown_path = output_dir / f"{pdf_path.stem}.md"

    markdown_path.write_text(
        markdown,
        encoding="utf-8",
    )

    return {
        **state,
        "artifacts": {
            **state["artifacts"],
            "markdown_path": str(markdown_path),
        },
    }