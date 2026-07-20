
from pathlib import Path

from docling.document_converter import DocumentConverter


class ParserService:

    @staticmethod
    def parse_pdf(pdf_path: str) -> str:
        """
        Parse a PDF using Docling and save it as Markdown.

        Args:
            pdf_path: Path to the downloaded PDF.

        Returns:
            Path to the generated Markdown file.
        """

        pdf_path = Path(pdf_path)

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

        return str(markdown_path)