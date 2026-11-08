
import json
from pathlib import Path

# Import your existing ingestion graph
from graphs.ingestion_graphs import ingestion_graph

from config import PROJECT_DIR,STAGES,PREVIEW_CHAR_LIMIT,CHUNK_PREVIEW_LIMIT






def read_markdown(artifacts):
    markdown_path = artifacts.get("markdown_path")

    if not markdown_path:
        return ""

    path = Path(markdown_path)

    if not path.is_file():
        return ""

    try:
        return path.read_text(encoding="utf-8")[:PREVIEW_CHAR_LIMIT]
    except (OSError, UnicodeError) as exc:
        return f"Could not read Markdown preview: {exc}"


def format_chunks(chunks):
    if not chunks:
        return ""

    sections = []

    for chunk in chunks[:CHUNK_PREVIEW_LIMIT]:
        chunk_text = chunk.get("text", "")

        sections.append(
            f"Chunk ID: {chunk.get('chunk_id', 'Unknown')}\n"
            f"Section: {chunk.get('section', 'Unknown')}\n"
            f"Characters: {chunk.get('char_count', len(chunk_text))}\n"
            f"{'-' * 60}\n"
            f"{chunk_text}"
        )

    result = ("\n\n" + "=" * 60 + "\n\n").join(sections)

    if len(chunks) > CHUNK_PREVIEW_LIMIT:
        result += (
            f"\n\nShowing {CHUNK_PREVIEW_LIMIT} of "
            f"{len(chunks)} chunks."
        )

    return result[:PREVIEW_CHAR_LIMIT]


def build_report(state, statuses):
    artifacts = state.get("artifacts", {})
    chunks = artifacts.get("chunks", [])

    embedding = artifacts.get("embedding", {})

    return {
        "pipeline_status": statuses,
        "file_name": artifacts.get("file_name"),
        "file_path": artifacts.get("file_path"),
        "markdown_path": artifacts.get("markdown_path"),
        "chunk_file": artifacts.get("chunk_file"),
        "chunk_size": artifacts.get("chunk_size"),
        "chunk_overlap": artifacts.get("chunk_overlap"),
        "total_chunks": len(chunks) if isinstance(chunks, list) else None,
        "collection_name": artifacts.get("collection_name"),
        "embedding_count": embedding.get("indexed_chunks", 0),
        "vector_dimension": embedding.get("vector_dimension", 0),
        "dense_model": embedding.get("dense_model", ""),
        "sparse_model": embedding.get("sparse_model", ""),
        "error": state.get("error", ""),
    }



def render_status(statuses, message=""):
    """Render the current state of each ingestion stage."""
    icons = {
        "Pending": "⏳",
        "processing": "🔄",
        "completed": "✅",
        "failed": "❌",
    }

    lines = [
        f"{icons.get(status, '•')} {stage}: {status.title()}"
        for stage, status in statuses.items()
    ]

    if message:
        lines.append("")
        lines.append(message)

    return "\n".join(lines)


def run_ingestion_pipeline(file_path: str):
    """
    Run the LangGraph ingestion pipeline.

    Yields structured progress events.
    """

    statuses = {stage: "Pending" for stage in STAGES}

    markdown_text = ""
    chunk_text = ""
    pipeline_error = ""
    result = None

    def make_event(message=""):
        state = result or {}

        report = build_report(state, statuses)

        if pipeline_error:
            report["error"] = pipeline_error

        return {
            "status": statuses,
            "message": message,
            "report": report,
            "markdown_preview": markdown_text,
            "chunk_preview": chunk_text,
        }

    initial_state = {
        "artifacts": {
            "uploaded_file_path": file_path,
        },
        "error": "",
    }

    try:

        for mode, data in ingestion_graph.stream(
            initial_state,
            stream_mode=["custom", "values"],
        ):

            if mode == "custom":

                stage = data.get("stage")
                status = data.get("status")

                if stage in statuses and status:
                    statuses[stage] = status

                if status == "failed":
                    pipeline_error = (
                        data.get("error")
                        or f"{stage} stage failed."
                    )

                yield make_event("Pipeline is running...")

            elif mode == "values":

                result = data

                artifacts = result.get("artifacts", {})

                if artifacts.get("markdown_path"):
                    markdown_text = read_markdown(artifacts)

                chunks = artifacts.get("chunks", [])

                if chunks:
                    chunk_text = format_chunks(chunks)

                graph_error = result.get("error", "")

                if graph_error:
                    pipeline_error = graph_error

                yield make_event("Pipeline is running...")

        if result is None:
            raise RuntimeError(
                "Ingestion graph finished without returning state."
            )

        final_error = result.get("error", "") or pipeline_error

        if final_error:
            yield make_event(f"Pipeline error: {final_error}")
            return

        for stage in STAGES:
            statuses[stage] = "completed"

        artifacts = result.get("artifacts", {})

        if artifacts.get("markdown_path"):
            markdown_text = read_markdown(artifacts)

        if artifacts.get("chunks"):
            chunk_text = format_chunks(artifacts["chunks"])

        yield make_event("Ingestion completed successfully!")

    except Exception as exc:

        pipeline_error = f"{type(exc).__name__}: {exc}"

        yield make_event(f"Pipeline error: {pipeline_error}")
