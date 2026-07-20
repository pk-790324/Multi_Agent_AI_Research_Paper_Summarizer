from pathlib import Path
import json
import re
from datetime import datetime

MAX_CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


class ChunkingService:

    @staticmethod
    def split_large_text(
        text: str,
        chunk_size: int = MAX_CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ):
        """
        Split long text into overlapping chunks.
        """

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):

            end = start + chunk_size

            chunks.append(text[start:end])

            start = end - overlap

        return chunks

    @staticmethod
    def section_aware_chunking(
        markdown_path: str,
        paper_title: str,
    ) -> tuple[list, str]:
        """
        Chunk a parsed markdown document into section-aware chunks.

        Returns:
            (chunks, chunk_file_path)
        """

        markdown_path = Path(markdown_path)

        if not markdown_path.exists():
            raise FileNotFoundError(markdown_path)

        markdown = markdown_path.read_text(
            encoding="utf-8"
        )

        pattern = r"(?=^#{1,6}\s+)"

        sections = re.split(
            pattern,
            markdown,
            flags=re.MULTILINE,
        )

        chunks = []

        chunk_id = 1

        for section in sections:

            section = section.strip()

            if not section:
                continue

            lines = section.splitlines()

            heading = "Unknown"

            if lines and lines[0].startswith("#"):
                heading = lines[0].replace("#", "").strip()

            body = "\n".join(lines[1:]).strip()

            if not body:
                continue

            for piece in ChunkingService.split_large_text(body):

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "paper_title": paper_title,
                        "section": heading,
                        "source_file": str(markdown_path),
                        "text": piece,
                        "char_count": len(piece),
                        "created_at": datetime.utcnow().isoformat(),
                    }
                )

                chunk_id += 1

        output_dir = Path("chunks")
        output_dir.mkdir(exist_ok=True)

        chunk_file = (
            output_dir /
            f"{markdown_path.stem}_chunks.json"
        )

        with open(
            chunk_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                chunks,
                f,
                indent=2,
                ensure_ascii=False,
            )

        return chunks, str(chunk_file)