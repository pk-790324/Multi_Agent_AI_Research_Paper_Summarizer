from pathlib import Path
import json
import re
from datetime import datetime

from langchain_core.tools import tool


MAX_CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


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

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


@tool
def section_aware_chunking() -> str:
    """
    Chunk the latest parsed markdown document.
    Save chunks as JSON.
    Return JSON path.
    """

    parsed_dir = Path("parsed_papers")

    if not parsed_dir.exists():
        return "parsed_papers directory not found."

    md_files = list(parsed_dir.glob("*.md"))

    if not md_files:
        return "No parsed markdown files found."

    latest_md = max(md_files, key=lambda f: f.stat().st_mtime)

    markdown = latest_md.read_text(
        encoding="utf-8"
    )

    paper_title = latest_md.stem

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

        if lines[0].startswith("#"):

            heading = lines[0].replace("#", "").strip()

        body = "\n".join(lines[1:]).strip()

        if not body:
            continue

        for piece in split_large_text(body):

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_title": paper_title,
                    "section": heading,
                    "source_file": str(latest_md),
                    "text": piece,
                    "char_count": len(piece),
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

            chunk_id += 1

    output_dir = Path("chunks")

    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{paper_title}_chunks.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return str(output_file)