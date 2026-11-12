
import re
from pathlib import Path


class PaperMetadataService:

    @staticmethod
    def extract(markdown_path: str) -> dict:

        path = Path(markdown_path)

        if not path.is_file():
            raise FileNotFoundError(path)

        content = path.read_text(encoding="utf-8")

        # Extract DOI.
        doi_pattern = (
            r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+'
        )

        doi_match = re.search(
            doi_pattern,
            content,
            re.IGNORECASE
        )

        doi = None

        if doi_match:
            doi = doi_match.group(0).rstrip(".,;:)]}")

        # Extract paper title from any Markdown heading (H1-H6).
        title_match = re.search(
            r"^\s{0,3}#{1,6}\s+(.+?)\s*$",
            content,
            re.MULTILINE
        )

        paper_title = (
            re.sub(
                r"\s+",
                " ",
                title_match.group(1).replace("\u00a0", " ")
            ).strip()
            if title_match
            else None
        )

        return {
            "paper_title": paper_title,
            "doi": doi,
        }
