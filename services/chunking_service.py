from pathlib import Path
import json
import re
from datetime import datetime


MAX_CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


class ChunkingService:

    # ---------------------------------------------------------
    # SECTION NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def normalize_section(heading: str) -> str:
        """
        Normalize section headings.

        Examples:

        1 Introduction
        -> Introduction

        2. Methodology
        -> Methodology

        4.1 Experimental Design
        -> 4.1 Experimental design

        Abstract
        -> Abstract
        """

        heading = heading.strip()

        # Remove markdown heading symbols
        heading = re.sub(
            r"^#{1,6}\s*",
            "",
            heading
        ).strip()

        # Remove only leading number for standard sections
        # BUT preserve subsection numbers such as 4.1
        #
        # 1 Introduction -> Introduction
        # 2. Methodology -> Methodology
        #
        # 4.1 Experimental Design -> 4.1 Experimental Design
        heading = re.sub(
            r"^\d+\.?\s+(?=[A-Za-z])",
            "",
            heading
        )

        heading = re.sub(
            r"\s+",
            " ",
            heading
        ).strip()

        normalized = heading.lower()

        mapping = {

            "abstract": "Abstract",

            "introduction": "Introduction",

            "background": "Background",

            "related work": "Related Work",

            "literature review": "Literature Review",

            "method": "Methodology",

            "methods": "Methodology",

            "methodology": "Methodology",

            "experimental setup": "Experiments",

            "experiment": "Experiments",

            "experiments": "Experiments",

            "evaluation": "Evaluation",

            "result": "Results",

            "results": "Results",

            "discussion": "Discussion",

            "conclusion": "Conclusion",

            "conclusions": "Conclusion",

            "limitations": "Limitations",

            "future work": "Future Work",

            "references": "References",

            "acknowledgements": "Acknowledgements",

            "acknowledgments": "Acknowledgements",
        }

        return mapping.get(
            normalized,
            heading
        )

    # ---------------------------------------------------------
    # MARKDOWN TABLE CLEANER
    # ---------------------------------------------------------

    @staticmethod
    def clean_markdown_table(text: str) -> str:
        """
        Convert Markdown tables into readable plain text.

        Example:

        | Method | Result |
        |--------|--------|
        | A      | 90%    |

        becomes:

        Method: A
        Result: 90%
        """

        lines = text.splitlines()

        table_lines = []

        for line in lines:

            stripped = line.strip()

            if not stripped:
                continue

            if "|" not in stripped:
                continue

            # Skip markdown separator rows
            if re.match(
                r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$",
                stripped
            ):
                continue

            table_lines.append(stripped)

        if not table_lines:
            return text

        cleaned = []

        headers = None

        for index, line in enumerate(table_lines):

            cells = [
                cell.strip()
                for cell in line.strip("|").split("|")
            ]

            if index == 0:

                headers = cells

                continue

            if headers:

                row_parts = []

                for i, value in enumerate(cells):

                    if i < len(headers):

                        header = headers[i]

                        if value:

                            row_parts.append(
                                f"{header}: {value}"
                            )

                if row_parts:

                    cleaned.append(
                        " ".join(row_parts)
                    )

            else:

                cleaned.append(
                    " ".join(cells)
                )

        if cleaned:

            return "\n".join(cleaned)

        return text

    # ---------------------------------------------------------
    # GENERAL TEXT CLEANING
    # ---------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean PDF/parser artifacts before embedding.
        """

        # Remove HTML comments
        text = re.sub(
            r"<!--.*?-->",
            "",
            text,
            flags=re.DOTALL
        )

        # Remove excessive whitespace
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # Normalize excessive blank lines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        # Fix spaces around punctuation
        text = re.sub(
            r"\s+([,.!?;:])",
            r"\1",
            text
        )

        return text.strip()

    # ---------------------------------------------------------
    # SPLIT LARGE TEXT
    # ---------------------------------------------------------

    @staticmethod
    def split_large_text(
        text: str,
        chunk_size: int = MAX_CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ):

        text = text.strip()

        if len(text) <= chunk_size:

            return [text]

        chunks = []

        start = 0

        while start < len(text):

            end = start + chunk_size

            # Try to break at a sentence
            if end < len(text):

                sentence_break = text.rfind(
                    ". ",
                    start,
                    end
                )

                if sentence_break > start:

                    end = sentence_break + 1

            chunk = text[start:end].strip()

            if chunk:

                chunks.append(chunk)

            start = max(
                end - overlap,
                start + 1
            )

        return chunks

    # ---------------------------------------------------------
    # SECTION-AWARE CHUNKING
    # ---------------------------------------------------------

    @staticmethod
    def section_aware_chunking(
        markdown_path: str,
        paper_title: str,
    ) -> tuple[list, str]:

        markdown_path = Path(markdown_path)

        if not markdown_path.exists():

            raise FileNotFoundError(
                markdown_path
            )

        markdown = markdown_path.read_text(
            encoding="utf-8"
        )

        # Split by markdown headings
        pattern = r"(?=^#{1,6}\s+)"

        sections = re.split(
            pattern,
            markdown,
            flags=re.MULTILINE
        )

        chunks = []

        chunk_id = 1

        for section in sections:

            section = section.strip()

            if not section:

                continue

            lines = section.splitlines()

            # -------------------------------------------------
            # GET HEADING
            # -------------------------------------------------

            heading = "Unknown"

            if lines and lines[0].startswith("#"):

                heading = ChunkingService.normalize_section(
                    lines[0]
                )

                body = "\n".join(
                    lines[1:]
                ).strip()

            else:

                body = section

            if not body:

                continue

            # -------------------------------------------------
            # CLEAN TABLES
            # -------------------------------------------------

            body = ChunkingService.clean_markdown_table(
                body
            )

            # -------------------------------------------------
            # CLEAN GENERAL TEXT
            # -------------------------------------------------

            body = ChunkingService.clean_text(
                body
            )

            if not body:

                continue

            # -------------------------------------------------
            # SPLIT INTO CHUNKS
            # -------------------------------------------------

            pieces = ChunkingService.split_large_text(
                body
            )

            for piece in pieces:

                chunks.append(
                    {
                        "chunk_id": chunk_id,

                        "paper_title": paper_title,

                        "section": heading,

                        "source_file": str(
                            markdown_path
                        ),

                        "text": piece,

                        "char_count": len(piece),

                        "created_at":
                            datetime.utcnow().isoformat(),
                    }
                )

                chunk_id += 1

        # -----------------------------------------------------
        # SAVE CHUNKS
        # -----------------------------------------------------

        output_dir = Path("chunks")

        output_dir.mkdir(
            exist_ok=True
        )

        chunk_file = (
            output_dir /
            f"{markdown_path.stem}_chunks.json"
        )

        with open(
            chunk_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                chunks,
                f,
                indent=2,
                ensure_ascii=False
            )

        return chunks, str(chunk_file)