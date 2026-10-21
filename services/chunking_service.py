
from pathlib import Path
import json
import re
from datetime import datetime


class ChunkingService:

    @staticmethod
    def section_aware_chunking(
        markdown_path: str,
        paper_title: str,
        chunk_size: int = 1800,
        overlap: int = 200,
    ) -> tuple[list[dict], str]:

        markdown_path = Path(markdown_path)

        if not markdown_path.is_file():
            raise FileNotFoundError(
                f"Markdown file not found: {markdown_path}"
            )

        markdown = markdown_path.read_text(
            encoding="utf-8"
        )

        # Split at Markdown headings: #, ##, ###, etc.
        sections = re.split(
            r"(?=^#{1,6}\s+)",
            markdown,
            flags=re.MULTILINE,
        )

        chunks = []
        chunk_id = 1

        for section_text in sections:
            section_text = section_text.strip()

            if not section_text:
                continue

            lines = section_text.splitlines()

            # Extract the heading and section body
            if lines[0].startswith("#"):
                heading = re.sub(
                    r"^#{1,6}\s+",
                    "",
                    lines[0],
                ).strip()

                body = "\n".join(lines[1:]).strip()
            else:
                heading = "Document"
                body = section_text

            # Clean up unnecessary whitespace
            body = re.sub(r"\s+", " ", body).strip()

            if not body:
                continue

            # Split long sections; preserve section metadata
            start = 0

            while start < len(body):
                end = min(start + chunk_size, len(body))

                # Prefer a sentence boundary when possible
                if end < len(body):
                    sentence_end = body.rfind(
                        ". ", start, end
                    )

                    if sentence_end > start:
                        end = sentence_end + 1

                piece = body[start:end].strip()

                if piece:
                    chunks.append({
                        "chunk_id": chunk_id,
                        "paper_title": paper_title,
                        "section": heading,
                        "source_file": str(markdown_path),
                        "text": piece,
                        "char_count": len(piece),
                        "created_at": datetime.now().isoformat(),
                    })

                    chunk_id += 1

                if end >= len(body):
                    break

                # Move forward while retaining overlap
                start = max(end - overlap, start + 1)

        # Save chunks as JSON
        output_dir = Path("chunks")
        output_dir.mkdir(parents=True, exist_ok=True)

        chunk_file = (
            output_dir / f"{markdown_path.stem}_chunks.json"
        )

        chunk_file.write_text(
            json.dumps(
                chunks,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return chunks, str(chunk_file)
