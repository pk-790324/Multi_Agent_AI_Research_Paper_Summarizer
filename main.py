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
from api.ingestion_service import render_status



import requests
import json
from pathlib import Path


from config import UPLOADED_DIR,SUPPORTED_FILES,STAGES,PROJECT_DIR

API_URL = "http://127.0.0.1:8001/ingestion/upload"


def run_ingestion(uploaded_file,current_user):
    """
    Call FastAPI ingestion endpoint and stream
    progress updates to Gradio.
    """

    statuses = {
        stage: "Pending"
        for stage in STAGES
    }

    if uploaded_file is None:
        yield (
            "Please upload a document first.",
            "{}",
            "",
            ""
        )
        return

    if not current_user or not current_user.get("id"):
        yield (
            "Please log in before running ingestion.",
            "{}",
            "",
            ""
        )
        return

    file_path = Path(uploaded_file)

    markdown_text = ""
    chunk_text = ""
    report = {}

    try:

        # Open the document and send it to FastAPI.
        with file_path.open("rb") as file:

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        file_path.name,
                        file,
                        "application/octet-stream"
                    )
                },
                data={"user_id": current_user["id"]},
                headers={
                    "Accept": "text/event-stream"
                },
                stream=True,
                timeout=(10, None)
            )

            response.raise_for_status()

            # Receive Server-Sent Events from FastAPI.
            for line in response.iter_lines(
                decode_unicode=True
            ):

                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                if not line.startswith("data:"):
                    continue

                event_data = line[len("data:"):].strip()

                try:
                    event = json.loads(event_data)
                except json.JSONDecodeError:
                    continue

                # Update pipeline statuses.
                statuses.update(
                    event.get("status", {})
                )

                # Update report.
                report = event.get("report", {})

                # Update previews.
                markdown_text = event.get(
                    "markdown_preview",
                    markdown_text
                )

                chunk_text = event.get(
                    "chunk_preview",
                    chunk_text
                )

                message = event.get(
                    "message",
                    "Pipeline is running..."
                )

                # Send updates to Gradio UI.
                yield (
                    render_status(statuses, message),
                    json.dumps(
                        report,
                        indent=2,
                        default=str
                    ),
                    markdown_text,
                    chunk_text
                )

    except requests.exceptions.RequestException as exc:

        error_message = f"API request failed: {exc}"

        yield (
            render_status(
                statuses,
                f"❌ {error_message}"
            ),
            json.dumps(
                {"error": error_message},
                indent=2
            ),
            markdown_text,
            chunk_text
        )

    except Exception as exc:

        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        yield (
            render_status(
                statuses,
                f"❌ {error_message}"
            ),
            json.dumps(
                {"error": error_message},
                indent=2
            ),
            markdown_text,
            chunk_text
        )


# =========================================================
# RESEARCH PAPERS PAGE
# =========================================================

import requests

PAPERS_API_URL = "http://127.0.0.1:8001/papers/uploaded"


def list_uploaded_papers():

    try:
        response = requests.get(
            PAPERS_API_URL,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if not data["files"]:
            return "No research papers have been uploaded yet."

        lines = ["## Uploaded Research Papers", ""]

        for paper in data["files"]:

            lines.append(
                f"- **{paper['filename']}** — "
                f"{paper['size_kb']:.1f} KB"
            )

        lines.append("")
        lines.append(f"**Total papers:** {data['total_files']}")

        return "\n".join(lines)

    except requests.exceptions.RequestException as exc:

        return f"❌ Unable to retrieve uploaded papers: {exc}"


# =========================================================
# MULTI AGENT RESEARCH PAPER PIPELINE
# =========================================================



import requests
import json


RESEARCH_API_URL = "http://127.0.0.1:8001/research/run"


def run_research_pipeline(user_query: str):

    if not user_query or not user_query.strip():

        yield (
            "Please enter a research question.",
            "",
            "### No research chunks retrieved.",
            "",
            "",
            "",
            [],
            [],
            "Pipeline Progress: 0/6 agents completed"
        )
        return

    try:

        # Send research query to FastAPI.
        with requests.post(
            RESEARCH_API_URL,
            json={"user_query": user_query.strip()},
            headers={"Accept": "text/event-stream"},
            stream=True,
            timeout=(10, None)
        ) as response:

            response.raise_for_status()

            # Read live SSE events.
            for line in response.iter_lines(
                decode_unicode=True
            ):

                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode("utf-8")

                if not line.startswith("data:"):
                    continue

                event_data = line[len("data:"):].strip()

                try:
                    event = json.loads(event_data)
                except json.JSONDecodeError:
                    continue

                # Extract API response.
                message = event.get(
                    "message",
                    "Research pipeline is running..."
                )

                final_answer = event.get("final_answer", "")

                retrieved_chunks = event.get(
                    "retrieved_chunks_markdown",
                    ""
                )

                analysis = event.get("analysis", "")

                critical_analysis = event.get(
                    "critical_analysis",
                    ""
                )

                cited_analysis = event.get(
                    "cited_analysis",
                    ""
                )

                citations = event.get("citations", [])

                dashboard = event.get(
                    "agent_dashboard",
                    []
                )

                progress = event.get(
                    "pipeline_progress",
                    ""
                )

                # logs = "\n".join(
                #     event.get("execution_logs", [])
                # )

                # Update all Gradio components.
                yield (
                    message,
                    final_answer,
                    retrieved_chunks,
                    analysis,
                    critical_analysis,
                    cited_analysis,
                    citations,
                    dashboard,
                    progress,
                    #logs
                )

    except requests.exceptions.RequestException as exc:

        error_message = f"Research API error: {exc}"

        yield (
            f"❌ {error_message}",
            "",
            "",
            "",
            "",
            "",
            [],
            [],
            "Pipeline failed"
        )

    except Exception as exc:

        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        yield (
            f"❌ {error_message}",
            "",
            "",
            "",
            "",
            "",
            [],
            [],
            "Pipeline failed"
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


# =============================================================
# LOGIN AND SIGN UP CODE
# =============================================================

from services.gradio_auth import (
    signup_user,
    login_user,
    logout_user
)




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

#auth-page {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: calc(100vh - 40px);
    padding: 24px;
    border-radius: 24px;
    background:
        radial-gradient(ellipse at 15% 15%, rgba(147, 197, 253, 0.3), transparent 38%),
        radial-gradient(ellipse at 85% 85%, rgba(196, 181, 253, 0.28), transparent 40%),
        linear-gradient(135deg, #f8fafc, #eff6ff);
}

#auth-page.hide {
    display: none !important;
}

#auth-card {
    box-sizing: border-box;
    width: 100%;
    max-width: 480px;
    margin: auto;
    padding: 36px;
    border: 1px solid rgba(148, 163, 184, 0.3);
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.94);
    box-shadow: 0 24px 64px rgba(30, 64, 175, 0.14);
}

#auth-card input,
#auth-card textarea {
    border-radius: 12px;
}

#auth-card button.primary {
    border: 0;
    background: linear-gradient(110deg, #2563eb, #7c3aed);
    color: #ffffff;
}

#auth-card button.primary:hover {
    background: linear-gradient(110deg, #1d4ed8, #6d28d9);
}

#auth-card [role="tab"] {
    font-weight: 600;
}

@media (max-width: 800px) {
    .gradio-container {
        padding: 12px !important;
    }

    #auth-page {
        min-height: calc(100vh - 24px);
        padding: 12px;
    }

    #auth-card {
        padding: 28px 22px;
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
    # Session state
    current_user = gr.State(value=None)
    # =========================
    #  AUTH PAGE
    # =========================
    with gr.Column(visible=True, elem_id="auth-page") as auth_page:
        with gr.Column(elem_id="auth-card"):
            gr.Markdown("# 📚 Research Paper Summarizer")
            gr.Markdown("Sign in or create an account to continue to your research workspace.")

            with gr.Tabs() as auth_tabs:

                # SIGNUP TAB
                with gr.Tab("Sign Up"):

                    signup_name = gr.Textbox(
                        label="Full Name"
                    )

                    signup_email = gr.Textbox(
                        label="Email"
                    )

                    signup_password = gr.Textbox(
                        label="Password",
                        type="password"
                    )

                    signup_button = gr.Button(
                        "Create Account",
                        variant="primary"
                    )

                # LOGIN TAB
                with gr.Tab("Login", id=1):

                    login_email = gr.Textbox(
                        label="Email"
                    )

                    login_password = gr.Textbox(
                        label="Password",
                        type="password"
                    )

                    login_button = gr.Button(
                        "Login",
                        variant="primary"
                    )

            auth_status = gr.Markdown()
    with gr.Column(visible=False) as protected_app:
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
                
               
                
                # Add logout button

                logout_button = gr.Button(
                    "Logout",
                    variant="stop"
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
                        inputs=[uploaded_file,current_user],
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
                    with gr.Group():

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
                    with gr.Group():
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
                    with gr.Group():

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
        
      # ==================================
    # EVENT HANDLERS
    # ==================================

    signup_button.click(
        fn=signup_user,
        inputs=[
            signup_name,
            signup_email,
            signup_password
        ],
        outputs=[auth_status, auth_tabs]
    )

    login_button.click(
        fn=login_user,
        inputs=[
            login_email,
            login_password
        ],
        outputs=[
            auth_page,
            protected_app,
            auth_status,
            current_user
        ]
    )

    logout_button.click(
        fn=logout_user,
        inputs=None,
        outputs=[
            auth_page,
            protected_app,
            current_user
        ]
    )


if __name__ == "__main__":
    demo.launch(
        share=True
    )
