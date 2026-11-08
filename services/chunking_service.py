
from pathlib import Path
import json
import re
from datetime import datetime



class ChunkingService:
    @staticmethod
    def section_aware_chunking(
        markdown_path: str,
        paper_title: str,
        chunk_size:int,
        overlap: int
    ) -> tuple[list[dict], str]:

        markdown_path = Path(markdown_path)

        if not markdown_path.is_file():
            raise FileNotFoundError(
                f"Markdown file not found: {markdown_path}"
            )

        markdown = markdown_path.read_text(
            encoding="utf-8"
        )

        # --------------------------------------------------
        # 1. SPLIT MARKDOWN INTO PDF PAGES
        # --------------------------------------------------

        page_marker = "<!-- PAGE_BREAK -->"

        has_page_markers = page_marker in markdown

        if has_page_markers:
            pages = markdown.split(page_marker)
        else:
            # Page information is unavailable.
            pages = [markdown]

        chunks = []
        chunk_id = 1

        current_section = "Document"

        # --------------------------------------------------
        # 2. PROCESS EACH PAGE
        # --------------------------------------------------

        for page_number, page_text in enumerate(
            pages, start=1
        ):

            lines = page_text.splitlines()

            section_bodies = []
            current_body = []

            # --------------------------------------------------
            # 3. EXTRACT SECTION HEADINGS
            # --------------------------------------------------

            for line in lines:

                if re.match(r"^#{1,6}\s+", line):

                    # Save previous section content.
                    if current_body:
                        section_bodies.append(
                            (
                                current_section,
                                "\n".join(current_body)
                            )
                        )

                    current_section = re.sub(
                        r"^#{1,6}\s+",
                        "",
                        line
                    ).strip()

                    current_body = []

                else:
                    current_body.append(line)

            # Save the last section on this page.
            if current_body:
                section_bodies.append(
                    (
                        current_section,
                        "\n".join(current_body)
                    )
                )

            # --------------------------------------------------
            # 4. CREATE CHUNKS FOR EACH SECTION
            # --------------------------------------------------

            for heading, body in section_bodies:

                body = re.sub(
                    r"\s+",
                    " ",
                    body
                ).strip()

                if not body:
                    continue

                start = 0

                while start < len(body):

                    end = min(
                        start + chunk_size,
                        len(body)
                    )

                    # Prefer sentence boundaries.
                    if end < len(body):

                        sentence_end = body.rfind(
                            ". ",
                            start,
                            end
                        )

                        if sentence_end > start:
                            end = sentence_end + 1

                    piece = body[start:end].strip()

                    if piece:

                        chunks.append({

                            "chunk_id": chunk_id,

                            "paper_title": paper_title,

                            "section": heading,

                            "page_start": (
                                page_number
                                if has_page_markers
                                else None
                            ),

                            "page_end": (
                                page_number
                                if has_page_markers
                                else None
                            ),

                            "source_file": str(markdown_path),

                            "text": piece,

                            "char_count": len(piece),

                            "created_at": datetime.now().isoformat(),

                        })

                        chunk_id += 1

                    if end >= len(body):
                        break

                    # Preserve overlap.
                    start = max(
                        end - overlap,
                        start + 1
                    )

        # --------------------------------------------------
        # 5. SAVE CHUNKS AS JSON
        # --------------------------------------------------

        output_dir = Path("chunks")

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        chunk_file = (
            output_dir /
            f"{markdown_path.stem}_chunks.json"
        )

        chunk_file.write_text(
            json.dumps(
                chunks,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print(f"Page markers detected: {has_page_markers}")
        print(f"Total pages processed: {len(pages)}")
        print(f"Total chunks generated: {len(chunks)}")

        return chunks, str(chunk_file)
