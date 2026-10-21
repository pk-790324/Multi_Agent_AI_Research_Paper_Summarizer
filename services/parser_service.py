
import os

# Must be set before importing Docling or PyTorch.
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch

# Disable Dynamo compilation for this process.
torch._dynamo.disable()

# Workaround for Docling versions that explicitly invoke torch.compile.
if hasattr(torch, "compile"):
    torch.compile = lambda model, *args, **kwargs: model

from pathlib import Path
from docling.document_converter import DocumentConverter


class ParserService:

    @staticmethod
    def parse_pdf(pdf_path: str) -> str:
        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"Document not found: {pdf_path}"
            )

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()

        output_dir = Path("parsed_papers")
        output_dir.mkdir(parents=True, exist_ok=True)

        markdown_path = output_dir / f"{pdf_path.stem}.md"
        markdown_path.write_text(markdown, encoding="utf-8")

        return str(markdown_path)
