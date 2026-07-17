
from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_core.tools import tool


@tool
def parse_latest_pdf() -> str:
    """
    Parse the latest PDF in the papers directory using Docling.
    Save the parsed document as Markdown and return its path.
    """

    papers_dir = Path("papers")

    if not papers_dir.exists():
        return "Error: papers directory does not exist."

    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        return "Error: No PDF files found."

    latest_pdf = max(pdf_files, key=lambda f: f.stat().st_mtime)

    converter = DocumentConverter()
    result = converter.convert(str(latest_pdf))

    markdown = result.document.export_to_markdown()

    output_dir = Path("parsed_papers")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{latest_pdf.stem}.md"

    output_file.write_text(markdown, encoding="utf-8")

    return str(output_file)