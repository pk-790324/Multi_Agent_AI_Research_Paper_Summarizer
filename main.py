
# import json
# from pathlib import Path

# import gradio as gr

# from graphs.ingestion_graphs import ingestion_graph




# def run_ingestion(uploaded_file):
#     """Run ingestion and stream live pipeline status to Gradio."""

#     if uploaded_file is None:
#         yield "Please upload a document first.", "{}", ""
#         return
#     markdown_text = ""
#     chunk_text = ""

#     stages = ["Upload", "Parsing", "Chunking", "Embeddings"]
#     statuses = {stage: "Pending" for stage in stages}

#     def render_status():
#         icons = {
#             "Pending": "⏳",
#             "processing": "🔄",
#             "completed": "✅",
#             "failed": "❌",
#         }

#         lines = []

#         for stage in stages:
#             current = statuses[stage]
#             if current == "Pending":
#                 label = "Pending"
#             elif current == "processing":
#                 label = "Processing..."
#             elif current == "completed":
#                 label = "Completed"
#             else:
#                 label = "Failed"

#             lines.append(f"{icons[current]} {stage}: {label}")

#         return "\n".join(lines)

#     initial_state = {
#         "artifacts": {
#             "uploaded_file_path": str(uploaded_file),
#         },
#         "error": "",
#     }

#     result = None
#     pipeline_error = ""

#     try:
#         # Stream custom events and state updates from LangGraph
#         for mode, data in ingestion_graph.stream(
#             initial_state,
#             stream_mode=["custom", "values"],
#         ):
#             if mode == "custom":
#                 stage = data.get("stage")
#                 status = data.get("status")

#                 if stage in statuses and status:
#                     statuses[stage] = status

#                 pipeline_error = data.get("error", "") or pipeline_error

#                 yield (
#                     render_status(),
#                     json.dumps({
#                         "pipeline_status": statuses,
#                         "message": "Pipeline is running...",
#                     }, indent=2),
#                     markdown_text,
#                     chunk_text,
#                 )
#             elif mode == "values":
#                 result = data

#         if result is None:
#             raise RuntimeError("The graph finished without returning state.")

#         artifacts = result.get("artifacts", {})
#         error = result.get("error", "") or pipeline_error

#         if error:
#             yield (
#                 render_status() + f"\n\n❌ Error: {error}",
#                 json.dumps({"error": error}, indent=2),
#                 "",
#             )
#             return

#         # Build final report
#         report = {}

#         for key, value in artifacts.items():
#             if key in {"chunks", "embeddings", "vectors"}:
#                 try:
#                     report[f"{key}_count"] = len(value)
#                 except TypeError:
#                     report[key] = type(value).__name__
#             elif key in {
#                 "file_name",
#                 "file_path",
#                 "file_extension",
#                 "file_size_bytes",
#                 "markdown_path",
#                 "chunk_file",
#                 "chunk_count",
#                 "embedding_count",
#             }:
#                 report[key] = value

#         report["pipeline_status"] = statuses
#         report["ingestion_status"] = "Completed"

        
#         markdown_path = artifacts.get("markdown_path")

#         if markdown_path and Path(markdown_path).is_file():
#             markdown_text = Path(markdown_path).read_text(
#                 encoding="utf-8"
#             )
#         # Build a readable chunk preview
#         chunks = artifacts.get("chunks", [])

        

#         if chunks:
#             chunk_sections = []

#             for chunk in chunks:
#                 chunk_sections.append(
#                     f"Chunk ID: {chunk.get('chunk_id', '')}\n"
#                     f"Section: {chunk.get('section', 'Unknown')}\n"
#                     f"Characters: {chunk.get('char_count', 0)}\n"
#                     f"{'-' * 60}\n"
#                     f"{chunk.get('text', '')}"
#                 )

#             chunk_text = "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(
#                 chunk_sections)

#         yield (
#             render_status() + "\n\n🎉 Ingestion completed successfully!",
#             json.dumps(report, indent=2, default=str),
#             markdown_text,
#             chunk_text,)
#     except Exception as exc:
#         yield (
#             render_status() + f"\n\n❌ Pipeline error: {exc}",
#             json.dumps({"error": str(exc)}, indent=2),
#             markdown_text,
#             chunk_text,
#         )



# with gr.Blocks(title="Research Paper Ingestion") as demo:

#     gr.Markdown("# Research Paper Ingestion Pipeline")

#     gr.Markdown(
#         """
#         Upload a research document and test your LangGraph ingestion
#         pipeline: file validation, document parsing, chunking, and
#         embedding generation.
#         """
#     )

#     with gr.Row():
#         with gr.Column(scale=1):
#             uploaded_file = gr.File(
#                 label="Upload Research Document",
#                 file_types=[
#                     ".pdf",
#                     ".doc",
#                     ".docx",
#                     ".ppt",
#                     ".pptx",
#                     ".txt",
#                     ".md",
#                     ".rtf",
#                 ],
#                 type="filepath",
#             )

#             run_button = gr.Button(
#                 "Run Data Ingestion",
#                 variant="primary",
#             )

#             status_output = gr.Textbox(
#                 label="Pipeline Status",
#                 interactive=False,
#                 lines=2,
#             )

#         with gr.Column(scale=1):
#             report_output = gr.Code(
#                 label="Ingestion Report",
#                 language="json",
#                 interactive=False,
#             )

    
#     with gr.Row():
#         with gr.Column(scale=1):
#             markdown_output = gr.Textbox(
#                 label="Parsed Markdown Preview",
#                 lines=25,
#                 interactive=False,
#             )

#         with gr.Column(scale=1):
#             chunk_output = gr.Textbox(
#                 label="Chunked Text Preview",
#                 lines=25,
#                 interactive=False,
#             )


#     run_button.click(
#     fn=run_ingestion,
#     inputs=[uploaded_file],
#     outputs=[
#         status_output,
#         report_output,
#         markdown_output,
#         chunk_output,
#     ],
# )


# if __name__ == "__main__":
#     demo.launch(
#             share=True
#     )







from pathlib import Path
import json

import gradio as gr

# Reuse your existing, working LangGraph ingestion pipeline.
from graphs.ingestion_graphs import ingestion_graph


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
UPLOADED_DIR = PROJECT_DIR / "data" / "uploaded_papers"

SUPPORTED_FILES = [
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
    ".md",
    ".rtf",
]

STAGES = ["Upload", "Parsing", "Chunking", "Embeddings"]

PREVIEW_CHAR_LIMIT = 20000
CHUNK_PREVIEW_LIMIT = 20


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def read_markdown(artifacts):
    """Read the Markdown produced by the parser node."""
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
    """Format chunk dictionaries for the Gradio preview."""
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
    """Create a readable report without exposing full vectors."""
    artifacts = state.get("artifacts", {})
    chunks = artifacts.get("chunks", [])

    return {
        "pipeline_status": statuses,
        "file_name": artifacts.get("file_name"),
        "file_path": artifacts.get("file_path"),
        "markdown_path": artifacts.get("markdown_path"),
        "chunk_file": artifacts.get("chunk_file"),
        "total_chunks": len(chunks) if isinstance(chunks, list) else None,
        "collection_name": artifacts.get("collection_name"),
        "embedding_count": artifacts.get("embedding", {}).get("indexed_chunks", 0),
        "vector_dimension":artifacts.get("embedding",{}).get("vector_dimension",0),
        "embedding_model":artifacts.get("embedding",{}).get("embedding_model",""),
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


# =========================================================
# INGESTION PIPELINE CALLBACK
# =========================================================

def run_ingestion(uploaded_file):
    """
    Run the existing LangGraph ingestion workflow.

    Every yield returns four values:
    status, report, Markdown preview, chunk preview.
    """

    statuses = {stage: "Pending" for stage in STAGES}
    markdown_text = ""
    chunk_text = ""
    pipeline_error = ""
    result = None

    def outputs(message=""):
        current_state = result or {}
        report = build_report(current_state, statuses)

        if pipeline_error:
            report["error"] = pipeline_error

        return (
            render_status(statuses, message),
            json.dumps(report, indent=2, default=str),
            markdown_text,
            chunk_text,
        )

    if uploaded_file is None:
        yield (
            "Please upload a document first.",
            "{}",
            "",
            "",
        )
        return

    # Gradio File(type="filepath") normally provides a filepath string.
    file_path = str(uploaded_file)

    initial_state = {
        "artifacts": {
            "uploaded_file_path": file_path,
        },
        "error": "",
    }

    try:
        # The graph is responsible for uploading/saving, parsing,
        # chunking, and generating embeddings.
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

                yield outputs("Pipeline is running...")

            elif mode == "values":
                # Keep the latest graph state.
                result = data

                artifacts = result.get("artifacts", {})

                # Refresh the Markdown preview as soon as parsing
                # has produced its output.
                if artifacts.get("markdown_path"):
                    markdown_text = read_markdown(artifacts)

                # Refresh the chunk preview as soon as chunking ends.
                chunks = artifacts.get("chunks", [])

                if chunks:
                    chunk_text = format_chunks(chunks)

                graph_error = result.get("error", "")

                if graph_error:
                    pipeline_error = graph_error

                yield outputs("Pipeline is running...")

        if result is None:
            raise RuntimeError(
                "The ingestion graph finished without returning state."
            )

        final_error = result.get("error", "") or pipeline_error

        if final_error:
            yield outputs(f"❌ Pipeline error: {final_error}")
            return

        # Mark all stages complete after successful graph execution.
        # This also accommodates graphs whose final custom event
        # was not emitted for a particular stage.
        for stage in STAGES:
            statuses[stage] = "completed"

        artifacts = result.get("artifacts", {})

        if artifacts.get("markdown_path"):
            markdown_text = read_markdown(artifacts)

        if artifacts.get("chunks"):
            chunk_text = format_chunks(artifacts["chunks"])

        yield outputs("🎉 Ingestion completed successfully!")

    except Exception as exc:
        pipeline_error = f"{type(exc).__name__}: {exc}"

        # Preserve the last known state and previews on failure.
        yield outputs(f"❌ Pipeline error: {pipeline_error}")


# =========================================================
# RESEARCH PAPERS PAGE
# =========================================================

def list_uploaded_papers():
    """List files saved by the existing upload service."""
    if not UPLOADED_DIR.exists():
        return (
            "No uploaded papers found yet.\n\n"
            f"Expected directory: `{UPLOADED_DIR}`"
        )

    files = sorted(
        (
            path for path in UPLOADED_DIR.iterdir()
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        return "No research papers have been uploaded yet."

    lines = ["## Uploaded Research Papers", ""]

    for path in files:
        size_kb = path.stat().st_size / 1024
        lines.append(f"- **{path.name}** — {size_kb:.1f} KB")

    return "\n".join(lines)


# =========================================================
# PAGE NAVIGATION
# =========================================================

PAGE_NAMES = [
    "Home",
    "Data Ingestion Pipeline",
    "Multi-Agent Pipeline",
    "Research Papers",
    "Settings",
]


def navigate(page):
    """Show exactly one page while preserving the sidebar."""
    return [
        gr.update(visible=(page == page_name))
        for page_name in PAGE_NAMES
    ]


# =========================================================
# CUSTOM CSS
# =========================================================

CSS = """
.gradio-container {
    max-width: 1600px !important;
}

#app-header {
    margin-bottom: 12px;
}

#sidebar {
    position: sticky;
    top: 12px;
    height: calc(100vh - 30px);
    overflow-y: auto;
    padding: 16px;
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
}

#sidebar button {
    text-align: left;
    min-height: 44px;
}

#main-content {
    padding: 8px 18px;
}

.page-title {
    margin-bottom: 6px;
}

.muted {
    color: var(--body-text-color-subdued);
}

.status-box textarea {
    font-family: monospace !important;
}

"""

# =========================================================
# GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    title="Research Paper AI Assistant",
    theme=gr.themes.Soft(),
    css=CSS,
) as demo:

    gr.Markdown(
        "# 🧠 Research Paper AI Assistant\n"
        "Your workspace for research document processing and AI analysis.",
        elem_id="app-header",
    )

    with gr.Row(equal_height=False):

        # -------------------------------------------------
        # LEFT SIDEBAR: stays visible across all pages
        # -------------------------------------------------

        with gr.Column(
            scale=1,
            min_width=230,
            elem_id="sidebar",
        ):
            gr.Markdown("## 📚 Research AI")
            gr.Markdown("**WORKSPACE**")

            home_btn = gr.Button(
                "🏠  Home",
                variant="primary",
            )

            ingestion_btn = gr.Button(
                "📄  Data Ingestion Pipeline",
            )

            agents_btn = gr.Button(
                "🤖  Multi-Agent Pipeline",
            )

            papers_btn = gr.Button(
                "📚  Research Papers",
            )

            settings_btn = gr.Button(
                "⚙️  Settings",
            )

            gr.Markdown(
                "---\n"
                "<small>Research Paper Summarizer</small>"
            )

        # -------------------------------------------------
        # RIGHT CONTENT: page panels
        # -------------------------------------------------

        with gr.Column(
            scale=5,
            elem_id="main-content",
        ):

            # =========================
            # PAGE 1: HOME
            # =========================

            with gr.Column(visible=True) as home_page:
                gr.Markdown(
                    "# Welcome to Research Paper AI",
                    elem_classes=["page-title"],
                )

                gr.Markdown(
                    "Process research papers, inspect extracted content, "
                    "and prepare documents for multi-agent analysis."
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "### 📄 Data Ingestion\n"
                            "Validate and parse documents, split text "
                            "into chunks, and generate embeddings."
                        )

                        home_ingestion_btn = gr.Button(
                            "Open Data Ingestion →",
                            variant="primary",
                        )

                    with gr.Column():
                        gr.Markdown(
                            "### 🤖 Multi-Agent Research\n"
                            "A dedicated workspace for future research "
                            "summarization and critical analysis agents."
                        )

                        home_agents_btn = gr.Button(
                            "Open Multi-Agent Pipeline →"
                        )

                gr.Markdown("## Pipeline Overview")

                gr.Markdown(
                    "| Stage | Purpose | Status |\n"
                    "|---|---|---|\n"
                    "| Upload | Validate and save the document | Ready |\n"
                    "| Parsing | Convert document to Markdown | Ready |\n"
                    "| Chunking | Split Markdown into chunks | Ready |\n"
                    "| Embeddings | Generate vector representations | Ready |\n"
                    "| Multi-Agent | Summarize and analyze research | Planned |"
                )

                gr.Markdown(
                    "*The overview describes available pipeline stages; "
                    "it does not indicate that a pipeline has run.*",
                    elem_classes=["muted"],
                )

            # =========================
            # PAGE 2: DATA INGESTION
            # =========================

            with gr.Column(visible=False) as ingestion_page:
                gr.Markdown("# 📄 Data Ingestion Pipeline")

                gr.Markdown(
                    "Upload a research document and monitor each stage "
                    "of your existing LangGraph workflow."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_file = gr.File(
                            label="Upload Research Document",
                            file_types=SUPPORTED_FILES,
                            type="filepath",
                        )

                        run_button = gr.Button(
                            "▶ Run Data Ingestion",
                            variant="primary",
                        )

                        status_output = gr.Textbox(
                            label="Live Pipeline Status",
                            value="Waiting for a document...",
                            lines=7,
                            interactive=False,
                            elem_classes=["status-box"],
                        )

                    with gr.Column(scale=1):
                        report_output = gr.Code(
                            label="Ingestion Report",
                            language="json",
                            interactive=False,
                        )

                gr.Markdown("## Document Inspection")

                with gr.Row():
                    with gr.Column():
                        markdown_output = gr.Textbox(
                            label="Parsed Markdown Preview",
                            placeholder=(
                                "Parsed Markdown will appear here "
                                "when parsing completes."
                            ),
                            lines=22,
                            interactive=False,
                        )

                    with gr.Column():
                        chunk_output = gr.Textbox(
                            label="Chunked Text Preview",
                            placeholder=(
                                "Chunked text will appear here "
                                "when chunking completes."
                            ),
                            lines=22,
                            interactive=False,
                        )

                run_button.click(
                    fn=run_ingestion,
                    inputs=[uploaded_file],
                    outputs=[
                        status_output,
                        report_output,
                        markdown_output,
                        chunk_output,
                    ],
                )

            # =========================
            # PAGE 3: MULTI-AGENT
            # =========================

            with gr.Column(visible=False) as agents_page:
                gr.Markdown("# 🤖 Multi-Agent Research Pipeline")

                gr.Markdown(
                    "This workspace is ready for your next development "
                    "milestone. The multi-agent graph has not been "
                    "implemented yet."
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "### 1. Parser Agent\n"
                            "Extract sections, tables, and useful "
                            "research-paper content."
                        )

                    with gr.Column():
                        gr.Markdown(
                            "### 2. Summarizer Agent\n"
                            "Produce a structured summary of the paper."
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "### 3. Critical Analyst Agent\n"
                            "Evaluate methodology, assumptions, "
                            "limitations, and evidence."
                        )

                    with gr.Column():
                        gr.Markdown(
                            "### 4. Synthesis Writer\n"
                            "Combine findings into a coherent final report."
                        )

                gr.Markdown(
                    "### Planned workflow\n\n"
                    "`Orchestrator → Parser → Summarizer → "
                    "Critical Analyst → Synthesis Writer`\n\n"
                    "These are planned components, not active agents yet."
                )

                gr.Button(
                    "Multi-Agent Pipeline — Coming Soon",
                    interactive=False,
                )

            # =========================
            # PAGE 4: RESEARCH PAPERS
            # =========================

            with gr.Column(visible=False) as papers_page:
                gr.Markdown("# 📚 Research Papers")

                gr.Markdown(
                    "Documents saved by your existing upload service."
                )

                refresh_papers_btn = gr.Button(
                    "🔄 Refresh Paper List"
                )

                papers_output = gr.Markdown(
                    "Click Refresh Paper List to view saved documents."
                )

                refresh_papers_btn.click(
                    fn=list_uploaded_papers,
                    outputs=papers_output,
                )

            # =========================
            # PAGE 5: SETTINGS
            # =========================

            with gr.Column(visible=False) as settings_page:
                gr.Markdown("# ⚙️ Settings")

                gr.Markdown(
                    "### Project Information\n\n"
                    f"- **Project:** Research Paper AI Assistant\n"
                    f"- **Framework:** Gradio\n"
                    f"- **Workflow orchestration:** LangGraph\n"
                    f"- **Project directory:** `{PROJECT_DIR}`\n\n"
                    "### Supported document types\n\n"
                    + ", ".join(SUPPORTED_FILES)
                )

    # -----------------------------------------------------
    # CONNECT SIDEBAR BUTTONS TO PAGE PANELS
    # -----------------------------------------------------

    page_outputs = [
        home_page,
        ingestion_page,
        agents_page,
        papers_page,
        settings_page,
    ]

    home_btn.click(
        fn=lambda: navigate("Home"),
        outputs=page_outputs,
    )

    ingestion_btn.click(
        fn=lambda: navigate("Data Ingestion Pipeline"),
        outputs=page_outputs,
    )

    agents_btn.click(
        fn=lambda: navigate("Multi-Agent Pipeline"),
        outputs=page_outputs,
    )

    papers_btn.click(
        fn=lambda: navigate("Research Papers"),
        outputs=page_outputs,
    )

    settings_btn.click(
        fn=lambda: navigate("Settings"),
        outputs=page_outputs,
    )

    # Quick links from the Home page.
    home_ingestion_btn.click(
        fn=lambda: navigate("Data Ingestion Pipeline"),
        outputs=page_outputs,
    )

    home_agents_btn.click(
        fn=lambda: navigate("Multi-Agent Pipeline"),
        outputs=page_outputs,
    )


if __name__ == "__main__":
    demo.launch(
        share=True
    )
