from pathlib import Path
import json
from graphs.research_graphs import research_graph
from agents.orchestrator_agent import MAX_AGENT_ATTEMPTS
import gradio as gr

# Reuse your existing, working LangGraph ingestion pipeline.
from graphs.ingestion_graphs import ingestion_graph

import subprocess
import sys
import json
from pathlib import Path
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
        "chunk_size":artifacts.get("chunk_size"),
        "chunk_overlap":artifacts.get("chunk_overlap"),
        "total_chunks": len(chunks) if isinstance(chunks, list) else None,
        "collection_name": artifacts.get("collection_name"),
        "embedding_count": artifacts.get("embedding", {}).get("indexed_chunks", 0),
        "vector_dimension":artifacts.get("embedding",{}).get("vector_dimension",0),
        "dense_model":artifacts.get("embedding",{}).get("dense_model",""),
        "sparse_model":artifacts.get("embedding",{}).get("sparse_model",""),
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
# Converting Retriever JSON to Markdown
# =========================================================

def format_retrieved_chunks(chunks):

    if not chunks:
        return "### 📚 No research chunks retrieved."

    markdown = "## 📚 Retrieved Research Chunks\n\n"

    markdown += f"**Total Chunks Retrieved:** {len(chunks)}\n\n"

    markdown += "---\n\n"

    for chunk in chunks:

        rank = chunk.get("rank", "N/A")
        score = chunk.get("score", 0)
        paper_title = chunk.get("paper_title", "Unknown Paper")
        section = chunk.get("section", "Unknown Section")
        page_start = chunk.get("page_start", "N/A")
        page_end = chunk.get("page_end", "N/A")
        content = chunk.get("page_content", "")
        source_file = chunk.get("source_file", "Unknown Source")
        chunk_id = chunk.get("chunk_id", "N/A")

        markdown += f"### 🔹 Rank {rank}\n\n"
        markdown += f"**Paper:** {paper_title}\n\n"
        markdown += f"**Section:** {section}\n\n"
        markdown += f"**Relevance Score:** `{score:.4f}`\n\n"
        markdown += f"**Pages:** {page_start} - {page_end}\n\n"
        markdown += f"**Chunk ID:** `{chunk_id}`\n\n"
        markdown += f"**Source File:** `{source_file}`\n\n"
        markdown += "**Content:**\n\n"
        markdown += f"> {content}\n\n"
        markdown += "---\n\n"

    return markdown


# =========================================================
# MULTI AGENT RESEARCH PAPER PIPELINE
# =========================================================


from datetime import datetime

def run_research_pipeline(user_query: str):

    agent_names = [
        "Orchestrator",
        "Retriever",
        "Research Analyst",
        "Critical Analyst",
        "Citation Agent",
        "Synthesis Writer",
    ]

    agent_counts = {
        name: 0 for name in agent_names
    }

    agent_status = {
        name: "Waiting" for name in agent_names
    }

    execution_logs = []

    final_state = {
        "user_query": user_query.strip(),
        "retrieved_chunks": [],
        "retrieval_count": 0,
        "retrieval_attempts": 0,
        "analyst_attempts": 0,
        "critic_attempts": 0,
        "max_critic_attempts": MAX_AGENT_ATTEMPTS,
        "analysis": "",
        "critical_analysis": "",
        "cited_analysis": "",
        "citations": [],
        "final_answer": "",
        "next_agent": "retriever",
        "completed_agents": [],
        "error": None,
    }

    def get_dashboard():

        rows = []

        for index, name in enumerate(agent_names, start=1):

            rows.append([
                index,
                name,
                agent_status[name],
                agent_counts[name]
            ])

        completed = sum(
            status == "Completed"
            for status in agent_status.values()
        )

        failed = sum(
            status == "Failed"
            for status in agent_status.values()
        )

        progress = (
            f"### Pipeline Progress: {completed}/{len(agent_names)} agents completed"
        )

        if failed:
            progress += f" | ❌ {failed} failed"

        return rows, progress

    def get_log():
        return "\n".join(execution_logs)

    if not user_query or not user_query.strip():

        yield (
            "Please enter a research question.",
            "",
            [],
            "",
            "",
            "",
            [],
            get_dashboard()[0],
            get_dashboard()[1],
            ""
        )
        return

    try:

        yield (
            "⏳ Initializing pipeline...",
            "",
            "",
            "",
            "",
            "",
            [],
            get_dashboard()[0],
            get_dashboard()[1],
            get_log()
        )

        for mode, event in research_graph.stream(
            final_state,
            config={"recursion_limit": 30},
            stream_mode=["custom", "updates"]
        ):

            if mode == "custom":

                agent = event.get("agent")
                status = event.get("status")

                if agent not in agent_status:
                    continue

                if status == "running":

                    agent_counts[agent] += 1
                    agent_status[agent] = "Running"

                    timestamp = datetime.now().strftime("%H:%M:%S")

                    execution_logs.append(
                        f"🟡 `{timestamp}` **{agent}** started "
                        f"(Execution #{agent_counts[agent]})"
                    )

                elif status == "completed":

                    agent_status[agent] = "Completed"

                    timestamp = datetime.now().strftime("%H:%M:%S")

                    execution_logs.append(
                        f"🟢 `{timestamp}` **{agent}** completed"
                    )

                elif status == "failed":

                    agent_status[agent] = "Failed"

                    timestamp = datetime.now().strftime("%H:%M:%S")

                    execution_logs.append(
                        f"🔴 `{timestamp}` **{agent}** failed: "
                        f"{event.get('error', 'Unknown error')}"
                    )

                rows, progress = get_dashboard()

                yield (
                    "⚡ Pipeline is running...",
                    final_state.get("final_answer", ""),
                    format_retrieved_chunks(final_state.get("retrieved_chunks",[])),
                    final_state.get("analysis", ""),
                    final_state.get("critical_analysis", ""),
                    final_state.get("cited_analysis", ""),
                    final_state.get("citations", []),
                    rows,
                    progress,
                    get_log()
                )

            elif mode == "updates":

                for node_name, node_update in event.items():

                    if isinstance(node_update, dict):
                        final_state.update(node_update)

        # --------------------------------
        # PIPELINE FINISHED
        # --------------------------------

        if final_state.get("error"):

            status_message = (
                f"❌ Pipeline failed: {final_state['error']}"
            )

        else:

            status_message = (
                "✅ Research pipeline completed successfully!"
            )

        rows, progress = get_dashboard()

        yield (
            status_message,
            final_state.get("final_answer", ""),
            format_retrieved_chunks(final_state.get("retrieved_chunks",[])),
            final_state.get("analysis", ""),
            final_state.get("critical_analysis", ""),
            final_state.get("cited_analysis", ""),
            final_state.get("citations", []),
            rows,
            progress,
            get_log()
        )

    except Exception as error:

        timestamp = datetime.now().strftime("%H:%M:%S")

        execution_logs.append(
            f"🔴 `{timestamp}` **Pipeline Error:** {str(error)}"
        )

        for agent in agent_names:

            if agent_status[agent] == "Running":
                agent_status[agent] = "Failed"

        rows, progress = get_dashboard()

        yield (
            f"❌ Pipeline failed: {str(error)}",
            "",
            format_retrieved_chunks(final_state.get("retrieved_chunks",[])),
            final_state.get("analysis", ""),
            final_state.get("critical_analysis", ""),
            final_state.get("cited_analysis", ""),
            final_state.get("citations", []),
            rows,
            progress,
            get_log()
        )
        
        
# =========================================================
# RETRIEVER EVALUATION CODE
# =========================================================

import gradio as gr
import pandas as pd

from evals.eval_retriever import (
    run_local,
    format_evaluation_report
)


def run_retriever_evaluation():

    project_root = Path(__file__).resolve().parent

    golden_path = (
        project_root /
        "goldens" /
        "retriever_deepeval_goldens.json"
    )

    # Initial state
    yield (
        "🔄 **Generating golden dataset...**",
        "⏳ Waiting for generation...",
        "⏳ Waiting for retrieval...",
        "⏳ Waiting for evaluation...",
        ""
    )

    try:

        # =====================================
        # STEP 1: GOLDEN GENERATION
        # =====================================

        generation = subprocess.run(
            [
                sys.executable,
                "-m",
                "goldens.generator_goldens"
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )

        yield (
            "✅ **Golden dataset generated successfully!**",
            "🔄 **Validating generated dataset...**",
            "⏳ Waiting for retrieval...",
            "⏳ Waiting for evaluation...",
            ""
        )

        # =====================================
        # STEP 2: VALIDATE DATASET
        # =====================================

        if not golden_path.exists():
            raise FileNotFoundError(
                "Golden dataset file was not generated."
            )

        with open(golden_path, "r", encoding="utf-8") as f:
            goldens = json.load(f)

        if not goldens:
            raise ValueError("Golden dataset is empty.")

        yield (
            "✅ **Golden dataset generated successfully!**",
            f"✅ **Dataset validated: {len(goldens)} golden test cases.**",
            "🔄 **Initializing hybrid retriever...**",
            "⏳ Waiting for evaluation...",
            ""
        )

        # =====================================
        # STEP 3: RETRIEVER EVALUATION
        # =====================================

        yield (
            "✅ **Golden generation completed.**",
            f"✅ **{len(goldens)} golden test cases validated.**",
            "🔄 **Running Hybrid Retrieval and collecting contexts...**",
            "⏳ Waiting for evaluation...",
            ""
        )

        result = run_local()

        yield (
            "✅ **Golden generation completed.**",
            f"✅ **{len(goldens)} golden test cases validated.**",
            "✅ **Hybrid retrieval completed.**",
            "🔄 **Calculating DeepEval metrics...**",
            ""
        )

        # =====================================
        # STEP 4: FORMAT RESULTS
        # =====================================

        report = format_evaluation_report(result)

        yield (
            "✅ **Golden generation completed.**",
            f"✅ **{len(goldens)} golden test cases validated.**",
            "✅ **Hybrid retrieval completed.**",
            "✅ **DeepEval evaluation completed!**",
            report
        )

    except subprocess.CalledProcessError as e:

        yield (
            "❌ **Golden generation failed.**",
            "⏹ Dataset validation skipped.",
            "⏹ Retriever evaluation skipped.",
            "⏹ DeepEval evaluation skipped.",
            f"### Generator Error\n```text\n{e.stderr}\n```"
        )

    except Exception as e:

        yield (
            "❌ **Pipeline failed.**",
            "⏹ Check previous stage.",
            "⏹ Evaluation stopped.",
            "❌ **Error occurred.**",
            f"### Error\n```text\n{str(e)}\n```"
        )

        # =====================================
        # STEP 2: VERIFY GENERATED DATASET
        # =====================================

        if not golden_path.exists():
            return "❌ Golden dataset was not generated."

        with open(golden_path, "r", encoding="utf-8") as f:
            goldens = json.load(f)

        if not goldens:
            return "❌ Generated golden dataset is empty."

        generation_message = (
            f"✅ Golden dataset generated successfully!\n\n"
            f"- Total goldens: **{len(goldens)}**\n\n"
        )

        # =====================================
        # STEP 3: RUN RETRIEVER EVALUATION
        # =====================================

        result = run_local()

        report = format_evaluation_report(result)

        # =====================================
        # STEP 4: RETURN COMPLETE REPORT
        # =====================================

        return (
            generation_message
            + "---\n\n"
            + report
        )

    except subprocess.CalledProcessError as e:

        return (
            "## ❌ Golden Dataset Generation Failed\n\n"
            f"```text\n{e.stderr}\n```"
        )

    except Exception as e:

        return (
            "## ❌ Evaluation Error\n\n"
            f"```text\n{str(e)}\n```"
        )
        





# =========================================================
# GENERATOR EVALUATION CODE
# =========================================================

import json
from pathlib import Path

from goldens.generator_faithfulness import (
    generate_faithfulness_goldens
)

from evals.eval_generator import (
    run as run_generator_evaluation,
    format_generator_report
)



def run_generator_evaluation_pipeline():

    project_root = Path(__file__).resolve().parent

    golden_path = (
        project_root /
        "goldens" /
        "faithfulness_goldens.json"
    )

    # =====================================
    # INITIAL STATUS
    # =====================================

    yield (
        "🔄 **Generating Faithfulness Golden Dataset...**",
        "⏳ Waiting for dataset validation...",
        "⏳ Waiting for multi-agent generation...",
        "⏳ Waiting for DeepEval...",
        ""
    )

    try:

        # =====================================
        # STEP 1: GENERATE GOLDENS
        # =====================================

        generate_faithfulness_goldens()

        yield (
            "✅ **Golden dataset generation completed!**",
            "🔄 **Validating generated dataset...**",
            "⏳ Waiting for multi-agent generation...",
            "⏳ Waiting for DeepEval...",
            ""
        )

        # =====================================
        # STEP 2: VALIDATE GOLDENS
        # =====================================

        if not golden_path.exists():
            raise FileNotFoundError(
                "Faithfulness golden dataset was not generated."
            )

        with open(
            golden_path,
            "r",
            encoding="utf-8"
        ) as f:

            goldens = json.load(f)

        if not goldens:
            raise ValueError(
                "Generated faithfulness golden dataset is empty."
            )

        yield (
            "✅ **Golden dataset generated successfully!**",
            f"✅ **Dataset validated: {len(goldens)} test cases.**",
            "🔄 **Preparing multi-agent generator evaluation...**",
            "⏳ Waiting for DeepEval...",
            ""
        )

        # =====================================
        # STEP 3: MULTI-AGENT GENERATION
        # =====================================

        yield (
            "✅ **Golden generation completed.**",
            f"✅ **{len(goldens)} golden test cases validated.**",
            "🔄 **Executing Analyst → Critical Analyst → Citation Agent → Synthesizer...**",
            "⏳ Waiting for DeepEval...",
            ""
        )

        # =====================================
        # STEP 4: GENERATOR + DEEPEVAL
        # =====================================

        result = run_generator_evaluation()

        yield (
            "✅ **Golden generation completed.**",
            f"✅ **{len(goldens)} golden test cases validated.**",
            "✅ **Multi-agent generation completed.**",
            "🔄 **Evaluating Faithfulness and Answer Relevancy...**",
            ""
        )

        # =====================================
        # STEP 5: FORMAT RESULTS
        # =====================================

        report = format_generator_report(result)

        yield (
            "✅ **Golden generation completed.**",
            f"✅ **{len(goldens)} golden test cases validated.**",
            "✅ **Multi-agent generation completed.**",
            "✅ **Generator evaluation completed!**",
            report
        )

    except Exception as e:

        yield (
            "❌ **Generator pipeline failed.**",
            "⏹ Check dataset generation or validation.",
            "⏹ Generator evaluation stopped.",
            "❌ **Evaluation not completed.**",
            f"### Error\n```text\n{str(e)}\n```"
        )




# =============================================================
# RAG PIPELINE EVALUATION CODE
# =============================================================

from evals.eval_rag_pipeline import run_rag_pipeline_gradio







# =========================================================
# PAGE NAVIGATION
# =========================================================

PAGE_NAMES = [
    "Home",
    "Data Ingestion Pipeline",
    "Multi-Agent Pipeline",
    "Research Papers",
    "Retriever",
    "Generator Evaluation",
    "Rag Pipeline Evaluation",
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
    max-width: 1480px !important;
    padding: 20px !important;
}

#app-header {
    margin-bottom: 18px;
    padding: 18px 22px;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    background: var(--background-fill-secondary);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

#sidebar {
    position: sticky;
    top: 12px;
    height: calc(100vh - 30px);
    overflow-y: auto;
    padding: 20px 16px;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    background: var(--background-fill-secondary);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

#sidebar button {
    display: flex;
    justify-content: flex-start;
    text-align: left;
    width: 100%;
    min-height: 46px;
    padding: 0 14px;
    border-radius: 12px;
    font-weight: 600;
    transition: background-color 160ms ease, transform 160ms ease;
}

#sidebar button:hover {
    transform: translateX(2px);
}

#sidebar button:focus-visible,
#main-content button:focus-visible {
    outline: 3px solid #93c5fd;
    outline-offset: 2px;
}

#main-content {
    min-width: 0;
    padding: 8px 18px 24px;
}

#main-content button {
    min-height: 44px;
    padding: 0 18px;
    border-radius: 12px;
    font-weight: 600;
    transition: background-color 160ms ease, transform 160ms ease;
}

#main-content button:hover {
    transform: translateY(-1px);
}

.gradio-container button.primary {
    background: #2563eb;
    border-color: #2563eb;
    color: #ffffff;
}

.gradio-container button.primary:hover {
    background: #1d4ed8;
    border-color: #1d4ed8;
}

.feature-card {
    height: 100%;
    padding: 20px;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    background: var(--background-fill-secondary);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
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

@media (max-width: 800px) {
    .gradio-container {
        padding: 12px !important;
    }

    .gradio-container .row {
        flex-wrap: wrap !important;
    }

    #sidebar {
        position: static;
        height: auto;
        max-height: none;
        margin-bottom: 12px;
    }

    #main-content {
        padding: 8px 4px 20px;
    }

    #app-header {
        padding: 16px;
    }
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
                variant="secondary",
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
            
            retriever_btn=gr.Button(
                "Retriever Evaluation",
            )
            generator_btn=gr.Button(
                "Generator Evaluation",
            )
            
            rag_evaluation_pipeline_btn=gr.Button(
                "Rag Pipeline Evaluation",
                
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
                    with gr.Column(elem_classes=["feature-card"]):
                        gr.Markdown(
                            "### 📄 Data Ingestion\n"
                            "Validate and parse documents, split text "
                            "into chunks, and generate embeddings."
                        )

                        home_ingestion_btn = gr.Button(
                            "Open Data Ingestion →",
                            variant="primary",
                        )

                    with gr.Column(elem_classes=["feature-card"]):
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
                    "| Multi-Agent | Summarize and analyze research | Ready |"
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
                gr.Markdown(
                    """
                    # 🤖 Multi-Agent Research Workspace

                    Ask questions about your indexed research papers.
                    Multiple AI agents will collaborate to generate
                    an evidence-based research report.
                    """
                )

                # --------------------------------
                # RESEARCH QUESTION
                # --------------------------------

                gr.Markdown("## 🔍 Research Question")

                query_input = gr.Textbox(
                    label="Enter your research question",
                    placeholder=(
                        "Example: Explain the abstract of "
                        "The Era of 1-bit LLMs."
                    ),
                    lines=3
                )

                run_button = gr.Button(
                    "🚀 Run Research Pipeline",
                    variant="primary"
                )

                
                # --------------------------------
                # LIVE EXECUTION TRACE
                # --------------------------------

                gr.Markdown("---")
                gr.Markdown("## ⚡ Agent Execution Monitor")

                pipeline_progress = gr.Markdown(
                    "**Progress:** Waiting to start"
                )

                execution_trace = gr.Dataframe(
                    headers=[
                        "Step",
                        "Agent",
                        "Status",
                        "Executions"
                    ],
                    datatype=[
                        "number",
                        "str",
                        "str",
                        "number"
                    ],
                    value=[
                        [1, "Orchestrator", "Waiting", 0],
                        [2, "Retriever", "Waiting", 0],
                        [3, "Research Analyst", "Waiting", 0],
                        [4, "Critical Analyst", "Waiting", 0],
                        [5, "Citation Agent", "Waiting", 0],
                        [6, "Synthesis Writer", "Waiting", 0],
                    ],
                    interactive=False,
                    wrap=True,
                    label="Agent Execution Status"
                )

                

                # --------------------------------
                # INDIVIDUAL AGENT OUTPUTS
                # --------------------------------

                gr.Markdown("---")
                gr.Markdown("## 🧠 Dense + Sparse + RRF + Encoder")
                with gr.Accordion("Retrieved Chunks", open=False):
                    retrieved_chunks = gr.Markdown()
                
                gr.Markdown("## 🧠 Agent Analysis")

                with gr.Accordion("1. Research Analyst", open=False):
                    analysis_output = gr.Markdown()

                with gr.Accordion("2. Critical Analyst", open=False):
                    critical_output = gr.Markdown()

                with gr.Accordion("3. Citation Agent", open=False):
                    cited_analysis_output = gr.Markdown()

                # --------------------------------
                # REFERENCES
                # --------------------------------

                gr.Markdown("---")
                gr.Markdown("## 📚 Research References")

                citations_output = gr.JSON(
                    label="Verified Citation Records"
                )
                
                # --------------------------------
                # PIPELINE STATUS
                # --------------------------------

                pipeline_status = gr.Markdown(
                    "Waiting for your research question..."
                )
                
                # --------------------------------
                # FINAL ANSWER
                # --------------------------------

                gr.Markdown("---")
                gr.Markdown("## 📄 Final Research Report")

                final_answer = gr.Markdown()


                # --------------------------------
                # PIPELINE EXECUTION
                # --------------------------------

                run_button.click(
                fn=run_research_pipeline,
                inputs=[query_input],
                outputs=[
                    pipeline_status,
                    final_answer,
                    retrieved_chunks,
                    analysis_output,
                    critical_output,
                    cited_analysis_output,
                    citations_output,
                    execution_trace,
                    pipeline_progress,
                    #execution_log
                ]
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
            # PAGE 5: RETRIEVER EVALUATIONS
            # =========================

            with gr.Column(visible=False) as retriever_evaluation_page:
                with gr.Tab("🔬 Retriever Evaluation"):

                    gr.Markdown(
                        """
                        # 🔬 Retriever Evaluation Dashboard

                        Evaluate Hybrid Retrieval using DeepEval.

                        **Pipeline:** Golden Generation → Dataset Validation
                        → Hybrid Retrieval → DeepEval → Results
                        """
                    )

                    run_eval_btn = gr.Button(
                        "▶ Run Retriever Evaluation",
                        variant="primary"
                    )

                    # =====================================
                    # PIPELINE STATUS
                    # =====================================

                    gr.Markdown("## ⚙️ Evaluation Pipeline")

                    generation_status = gr.Markdown(
                        "⏳ Waiting for golden dataset generation..."
                    )

                    validation_status = gr.Markdown(
                        "⏳ Waiting for dataset validation..."
                    )

                    retrieval_status = gr.Markdown(
                        "⏳ Waiting for retriever evaluation..."
                    )

                    evaluation_status = gr.Markdown(
                        "⏳ Waiting for DeepEval metrics..."
                    )

                    # =====================================
                    # FINAL REPORT
                    # =====================================

                    gr.Markdown("---")
                    gr.Markdown("## 📊 Evaluation Results")

                    evaluation_output = gr.Markdown(
                        label="DeepEval Evaluation Report"
                    )

                    # =====================================
                    # BUTTON EVENT
                    # =====================================

                    run_eval_btn.click(
                        fn=run_retriever_evaluation,
                        inputs=None,
                        outputs=[
                            generation_status,
                            validation_status,
                            retrieval_status,
                            evaluation_status,
                            evaluation_output
                        ]
                    )
            # =========================
            # PAGE 6: GENERATOR EVALUATION PAGE
            # =========================
            
            with gr.Column(visible=False) as generator_evaluation_page:
                with gr.Tab("🧠 Generator Evaluation"):
                    gr.Markdown(
                        """
                        # 🧠 Generator Evaluation Dashboard

                        Evaluate multi-agent answer generation using DeepEval.

                        **Pipeline:** Golden Generation → Validation
                        → Multi-Agent Generation → Faithfulness
                        → Answer Relevancy
                        """
                    )

                    run_generator_btn = gr.Button(
                        "▶ Run Generator Evaluation",
                        variant="primary"
                    )

                    # =====================================
                    # PIPELINE STATUS
                    # =====================================

                    gr.Markdown("## ⚙️ Generator Evaluation Pipeline")

                    gen_status = gr.Markdown(
                        "⏳ Waiting for golden generation..."
                    )

                    gen_validation_status = gr.Markdown(
                        "⏳ Waiting for dataset validation..."
                    )

                    multi_agent_status = gr.Markdown(
                        "⏳ Waiting for multi-agent generation..."
                    )

                    deepeval_status = gr.Markdown(
                        "⏳ Waiting for DeepEval..."
                    )

                    # =====================================
                    # FINAL REPORT
                    # =====================================

                    gr.Markdown("---")

                    gr.Markdown("## 📊 Generator Evaluation Results")

                    generator_output = gr.Markdown(
                        label="DeepEval Generator Report"
                    )

                    # =====================================
                    # BUTTON EVENT
                    # =====================================

                    run_generator_btn.click(
                        fn=run_generator_evaluation_pipeline,
                        inputs=None,
                        outputs=[
                            gen_status,
                            gen_validation_status,
                            multi_agent_status,
                            deepeval_status,
                            generator_output
                        ]
                    )

                
                
                              
            # =========================
            # PAGE 7: RAG EVALUATION PIPELINE
            # =========================
            
            with gr.Column(visible=False) as rag_evaluation_pipeline_page:
                with gr.Tab("🔬 RAG Pipeline Evaluation"):

                    gr.Markdown(
                        """
                        # End-to-End RAG Pipeline Evaluation

                        Evaluate live hybrid retrieval and multi-agent generation
                        using DeepEval metrics.
                        """
                    )

                    run_rag_btn = gr.Button(
                        "🚀 Run RAG Pipeline Evaluation",
                        variant="primary"
                    )

                    rag_status = gr.Markdown(
                        "Waiting to start evaluation..."
                    )

                    rag_report = gr.Markdown(
                        label="RAG Evaluation Report"
                    )

                    run_rag_btn.click(
                        fn=run_rag_pipeline_gradio,
                        inputs=[],
                        outputs=[
                            rag_status,
                            rag_report
                        ]
                    )

            
            # =========================
            # PAGE 8: SETTINGS
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
        retriever_evaluation_page,
        generator_evaluation_page,
        rag_evaluation_pipeline_page,
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
    
    retriever_btn.click(
        fn=lambda: navigate("Retriever"),
        outputs=page_outputs,
    )
    
    generator_btn.click(
        fn=lambda: navigate("Generator Evaluation"),
        outputs=page_outputs,
    )
    
    rag_evaluation_pipeline_btn.click(
        fn=lambda: navigate("Rag Pipeline Evaluation"),
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
