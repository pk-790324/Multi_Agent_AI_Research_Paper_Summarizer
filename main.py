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



def list_uploaded_papers(current_user):
    access_token = (
        current_user.get("access_token")
        if current_user
        else None
    )

    if not access_token:
        return "Please log in again to view your uploaded papers."

    try:
        response = requests.get(
            PAPERS_API_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10
        )

        if response.status_code == 401:
            return "Your session has expired. Please log in again."

        response.raise_for_status()

        data = response.json()
        papers = data.get("files", [])

        if not papers:
            return "No research papers have been uploaded yet."

        lines = ["## Uploaded Research Papers", ""]

        for paper in papers:

            paper_title = paper.get("paper_title") or "Untitled Paper"

            lines.append(f"- **{paper_title}**")

            # Display all versions
            versions = paper.get("versions", [])

            for version in versions:

                version_number = version.get("version")
                is_active = version.get("is_active", False)
                status = version.get("status", "unknown")

                if is_active:
                    display_status = "🟢 Active"
                elif status.lower() == "inactive":
                    display_status = "⚪ Superseded"
                else:
                    display_status = status.capitalize()

                lines.append(
                    f"  - Version {version_number} — {display_status}"
                )

            lines.append("")

        total_papers = data.get("total_files", len(papers))

        lines.append(f"**Total papers:** {total_papers}")

        return "\n".join(lines)

    except requests.exceptions.RequestException as exc:
        return f"❌ Unable to retrieve uploaded papers: {exc}"

# =========================================================
# MULTI AGENT RESEARCH PAPER PIPELINE
# =========================================================



import requests
import json


RESEARCH_API_URL = "http://127.0.0.1:8001/research/run"


def run_research_pipeline(user_query: str, current_user):

    if not user_query or not user_query.strip():

        yield (
            "Please enter a research question.",
            "",
            "### No research chunks retrieved.",
            "",
            "",
            "",
            [],
            "Pipeline Progress: 0/6 agents completed"
        )
        return

    access_token = (
        current_user.get("access_token")
        if current_user
        else None
    )
    if not access_token:
        yield (
            "Please log in again to continue.",
            "",
            "### No research chunks retrieved.",
            "",
            "",
            "",
            [],
            "Pipeline Progress: 0/6 agents completed"
        )
        return

    try:

        # Send research query to FastAPI.
        with requests.post(
            RESEARCH_API_URL,
            json={"user_query": user_query.strip()},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "text/event-stream",
            },
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
            "Pipeline failed"
        )

        
# =========================================================
# RETRIEVER EVALUATION CODE
# =========================================================

import gradio as gr
import pandas as pd

def run_retriever_evaluation(current_user):
    access_token = (
        current_user.get("access_token")
        if current_user
        else None
    )
    if not access_token:
        yield (
            "❌ Please log in to run retriever evaluation.",
            "⏹ Evaluation stopped.",
            "⏹ Evaluation stopped.",
            "⏹ Evaluation stopped.",
            "",
        )
        return

    yield (
        "🔄 **Running authenticated retriever evaluation...**",
        "⏳ Golden generation and validation are in progress...",
        "⏳ Retrieval is in progress...",
        "⏳ Evaluation is in progress...",
        "",
    )

    try:
        response = requests.post(
            "http://127.0.0.1:8001/evaluation/retriever",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=(10, None),
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"Retriever evaluation API returned "
                f"{response.status_code}: {detail}"
            )

        report = response.json().get("results")
        if not isinstance(report, str):
            raise ValueError(
                "Retriever evaluation API returned no results report."
            )

        yield (
            "✅ **Golden generation completed.**",
            "✅ **Golden dataset validated.**",
            "✅ **Hybrid retrieval completed.**",
            "✅ **DeepEval evaluation completed.**",
            report,
        )

    except (
        requests.exceptions.RequestException,
        ValueError,
        RuntimeError,
    ) as exc:
        yield (
            "❌ **Retriever evaluation failed.**",
            "⏹ Evaluation stopped.",
            "⏹ Evaluation stopped.",
            "❌ **Error occurred.**",
            f"```text\n{exc}\n```",
        )





# =========================================================
# GENERATOR EVALUATION CODE
# =========================================================
import requests


def run_generator_evaluation_pipeline(current_user):

    # =====================================
    # AUTHENTICATION
    # =====================================

    access_token = (
        current_user.get("access_token")
        if current_user
        else None
    )

    if not access_token:
        yield (
            "❌ Please log in to run generator evaluation.",
            "⏹ Evaluation stopped.",
            "⏹ Evaluation stopped.",
            "⏹ Evaluation stopped.",
            "",
        )
        return

    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    base_url = "http://127.0.0.1:8001/evaluation"

    # =====================================
    # INITIAL STATUS
    # =====================================

    yield (
        "🔄 **Starting authenticated generator evaluation...**",
        "⏳ Waiting for golden dataset generation...",
        "⏳ Waiting for multi-agent generation...",
        "⏳ Waiting for DeepEval...",
        "",
    )

    try:

        # =====================================
        # STEP 1: GENERATE GOLDENS
        # =====================================

        golden_response = requests.post(
            f"{base_url}/generator/generate-goldens",
            headers=headers,
            timeout=(10, None),
        )

        if not golden_response.ok:

            try:
                detail = golden_response.json().get(
                    "detail",
                    golden_response.text
                )
            except ValueError:
                detail = golden_response.text

            raise RuntimeError(
                f"Golden generation API returned "
                f"{golden_response.status_code}: {detail}"
            )

        golden_data = golden_response.json()

        yield (
            "✅ **Golden dataset generation completed!**",
            "🔄 **Validating generated dataset...**",
            "⏳ Waiting for multi-agent generation...",
            "⏳ Waiting for DeepEval...",
            "",
        )

        # =====================================
        # STEP 2: VALIDATE GOLDEN RESPONSE
        # =====================================

        if golden_data.get("status") not in ("success", "completed"):
            raise ValueError(
                "Golden generation API did not confirm successful generation."
            )

        total_goldens = golden_data.get("total_goldens")

        if total_goldens is not None and total_goldens == 0:
            raise ValueError(
                "Generated faithfulness golden dataset is empty."
            )

        yield (
            "✅ **Golden dataset generated successfully!**",
            f"✅ **Dataset validated: {total_goldens if total_goldens is not None else 'Available'} test cases.**",
            "🔄 **Running multi-agent generator...**",
            "⏳ Waiting for DeepEval...",
            "",
        )

        # =====================================
        # STEP 3: RUN GENERATOR EVALUATION
        # =====================================

        evaluation_response = requests.post(
            f"{base_url}/generator",
            headers=headers,
            timeout=(10, None),
        )

        if not evaluation_response.ok:

            try:
                detail = evaluation_response.json().get(
                    "detail",
                    evaluation_response.text
                )
            except ValueError:
                detail = evaluation_response.text

            raise RuntimeError(
                f"Generator evaluation API returned "
                f"{evaluation_response.status_code}: {detail}"
            )

        evaluation_data = evaluation_response.json()

        # =====================================
        # STEP 4: VALIDATE EVALUATION REPORT
        # =====================================

        report = evaluation_data.get("report")

        if not isinstance(report, str) or not report.strip():
            raise ValueError(
                "Generator evaluation API returned no results report."
            )

        # =====================================
        # STEP 5: DISPLAY FINAL RESULTS
        # =====================================

        yield (
            "✅ **Golden generation completed.**",
            "✅ **Golden dataset validated.**",
            "✅ **Multi-agent generator completed.**",
            "✅ **DeepEval evaluation completed.**",
            report,
        )

    except (
        requests.exceptions.RequestException,
        ValueError,
        RuntimeError,
    ) as exc:

        yield (
            "❌ **Generator evaluation failed.**",
            "⏹ Evaluation stopped.",
            "⏹ Evaluation stopped.",
            "❌ **Error occurred.**",
            f"```text\n{exc}\n```",
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


def get_sidebar_user_name(user):
    name = user.get("name") if isinstance(user, dict) else None
    label = f"◉  {name}" if isinstance(name, str) and name else "◉ Profile"
    return gr.update(label=label)


# =========================================================
# CUSTOM CSS
# =========================================================

CSS = """
:root {
    --indigo-950: #1e1b4b;
    --indigo-900: #312e81;
    --indigo-800: #3730a3;
    --indigo-700: #4338ca;
    --indigo-600: #4f46e5;
    --indigo-500: #6366f1;
    --indigo-400: #818cf8;
    --indigo-300: #a5b4fc;
    --indigo-200: #c7d2fe;
    --indigo-100: #e0e7ff;
    --indigo-50: #eef2ff;
    --page-bg: #f5f5ff;
    --muted-text: #625f89;
}

body {
    background: var(--page-bg) !important;
}

.gradio-container {
    max-width: 1600px !important;
    min-height: 100vh;
    padding: 24px !important;
    background: transparent !important;
}

#protected-app {
    min-height: calc(100vh - 48px);
}

#workspace-layout {
    align-items: stretch;
    flex-wrap: nowrap !important;
    gap: 20px;
}

#sidebar {
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    position: sticky;
    top: 20px;
    box-sizing: border-box;
    flex: 0 0 258px !important;
    height: calc(100vh - 48px);
    max-height: calc(100vh - 48px);
    min-height: 0;
    overflow-x: hidden;
    overflow-y: scroll !important;
    gap: 4px;
    padding: 20px 14px 14px;
    border: 1px solid var(--indigo-900);
    border-radius: 20px;
    background: var(--indigo-950) !important;
    box-shadow: 0 18px 42px rgba(49, 46, 129, 0.16);
    color: var(--indigo-50);
}

#sidebar > * {
    box-sizing: border-box;
    flex: 0 0 auto;
    width: 100%;
}

#sidebar-brand {
    margin: 0 4px 12px;
    padding: 2px 8px 10px;
    color: #ffffff;
}

#sidebar-brand h2,
#sidebar-brand h3 {
    margin: 0 0 5px;
    color: #ffffff;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

#sidebar-brand p {
    margin: 0;
    color: var(--indigo-300);
    font-size: 0.72rem;
    font-weight: 650;
    letter-spacing: 0.12em;
}

.sidebar-section-label {
    margin: 12px 8px 4px;
    color: var(--indigo-300);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.13em;
}

#sidebar button {
    display: flex;
    justify-content: flex-start;
    text-align: left;
    width: 100%;
    min-height: 42px;
    padding: 0 12px;
    border: 1px solid transparent !important;
    border-radius: 11px;
    background: transparent !important;
    box-shadow: none !important;
    color: var(--indigo-200) !important;
    font-weight: 550;
    transition: background-color 160ms ease, color 160ms ease, transform 160ms ease;
}

#sidebar button:hover {
    transform: translateX(2px);
    background: var(--indigo-900) !important;
    color: #ffffff !important;
}

#sidebar button.primary {
    min-height: 46px;
    margin-bottom: 8px;
    border-color: var(--indigo-100) !important;
    background: var(--indigo-50) !important;
    color: var(--indigo-900) !important;
    font-weight: 700;
}

#sidebar button.primary:hover {
    background: #ffffff !important;
    color: var(--indigo-950) !important;
}

#sidebar-bottom {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: auto;
    padding-top: 16px;
    border-top: 1px solid var(--indigo-900);
}

#sidebar #user-profile,
#sidebar #user-profile > .wrap {
    display: flex;
    flex-direction: column-reverse;
    gap: 8px;
}

#sidebar #user-profile .label-wrap {
    min-height: 48px;
    align-items: center;
    padding: 0 12px;
    border-radius: 12px;
    background: var(--indigo-900) !important;
    color: #ffffff !important;
    transition: background-color 160ms ease;
}

#sidebar #user-profile .label-wrap:hover {
    background: var(--indigo-800) !important;
}

#sidebar #user-profile .label-wrap span {
    color: #ffffff !important;
    font-weight: 650;
}

#sidebar #user-profile .label-wrap:focus-visible {
    outline: 3px solid var(--indigo-400);
    outline-offset: 2px;
}

#sidebar #user-profile > .wrap > :not(.label-wrap) {
    padding: 8px;
    border: 1px solid var(--indigo-800);
    border-radius: 14px;
    background: var(--indigo-900) !important;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.3);
}

#sidebar #user-profile #logout-button button {
    transform: none !important;
    border-radius: 9px;
    color: var(--indigo-100) !important;
}

#sidebar #user-profile #logout-button button:hover {
    transform: none !important;
    background: var(--indigo-800) !important;
    color: #ffffff !important;
}

#sidebar button:focus-visible,
#main-content button:focus-visible {
    outline: 3px solid var(--indigo-400);
    outline-offset: 2px;
}

#main-content {
    min-width: 0;
    flex: 1 1 0 !important;
    padding: 8px 14px 28px;
    color: var(--indigo-950);
}

#main-content h1,
#main-content h2,
#main-content h3 {
    color: var(--indigo-950);
    letter-spacing: -0.03em;
}

#main-content h1 {
    font-size: clamp(1.7rem, 3vw, 2.25rem);
}

#main-content p {
    color: var(--muted-text);
    line-height: 1.65;
}

#main-content input,
#main-content textarea,
#main-content select {
    border-radius: 11px !important;
}

#main-content input:focus,
#main-content textarea:focus {
    border-color: var(--indigo-500) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

#main-content button {
    min-height: 44px;
    padding: 0 17px;
    border-radius: 11px;
    font-weight: 650;
    transition: background-color 160ms ease, border-color 160ms ease, transform 160ms ease;
}

#main-content button:hover {
    transform: translateY(-1px);
}

.gradio-container button.primary {
    background: var(--indigo-600);
    border-color: var(--indigo-600);
    color: #ffffff;
}

.gradio-container button.primary:hover {
    background: var(--indigo-700);
    border-color: var(--indigo-700);
}

.feature-card {
    height: 100%;
    padding: 22px;
    border: 1px solid var(--indigo-100);
    border-radius: 18px;
    background: #ffffff;
    box-shadow: 0 10px 28px rgba(49, 46, 129, 0.07);
}

.feature-card h3 {
    margin-top: 2px;
    color: var(--indigo-950);
}

.page-title {
    margin-bottom: 8px;
}

.muted {
    color: var(--muted-text);
}

.status-box textarea {
    font-family: monospace !important;
    line-height: 1.55;
}

#auth-page {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: calc(100vh - 48px);
    padding: 28px;
    border-radius: 26px;
    background: var(--indigo-50);
}

#auth-page.hide {
    display: none !important;
}

#auth-card {
    box-sizing: border-box;
    width: 100%;
    max-width: 480px;
    margin: auto;
    padding: 38px;
    border: 1px solid var(--indigo-200);
    border-radius: 24px;
    background: #ffffff;
    box-shadow: 0 28px 80px rgba(49, 46, 129, 0.12);
}

#auth-card h1 {
    margin-bottom: 8px;
    color: var(--indigo-950);
    letter-spacing: -0.04em;
}

#auth-card p {
    color: var(--muted-text);
    line-height: 1.6;
}

#auth-card input,
#auth-card textarea {
    border-radius: 11px;
}

#auth-card button.primary {
    border: 0;
    background: var(--indigo-600);
    color: #ffffff;
}

#auth-card button.primary:hover {
    background: var(--indigo-700);
}

#auth-card [role="tab"] {
    font-weight: 650;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 14px !important;
    }

    #auth-page {
        min-height: calc(100vh - 28px);
        padding: 14px;
    }

    #auth-card {
        padding: 30px 24px;
    }

    #workspace-layout {
        flex-direction: column !important;
        flex-wrap: wrap !important;
    }

    #sidebar {
        position: static;
        height: calc(100vh - 28px);
        min-height: 0;
        max-height: calc(100vh - 28px);
        margin-bottom: 12px;
        flex-basis: auto !important;
        overflow-x: hidden;
        overflow-y: scroll !important;
    }

    #main-content {
        padding: 8px 4px 20px;
    }

    .gradio-container .row {
        flex-wrap: wrap !important;
    }
}
"""

# =========================================================
# GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    title="Research Paper AI Assistant",
) as demo:
    # Session state
    current_user = gr.State(value=None)
    # =========================
    #  AUTH PAGE
    # =========================
    with gr.Column(visible=True, elem_id="auth-page") as auth_page:
        with gr.Column(elem_id="auth-card"):
            gr.Markdown("# ◈ Research Paper Summarizer")
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
    with gr.Column(visible=False, elem_id="protected-app") as protected_app:
        with gr.Row(equal_height=False, elem_id="workspace-layout"):

            # -------------------------------------------------
            # LEFT SIDEBAR: stays visible across all pages
            # -------------------------------------------------

            with gr.Column(
                scale=1,
                min_width=230,
                elem_id="sidebar",
            ):
                gr.Markdown(
                    "## ◈ Research AI\n"
                    "<p>RESEARCH WORKSPACE</p>",
                    elem_id="sidebar-brand",
                )

                home_btn = gr.Button(
                    "⌂  Home",
                    variant="primary",
                    elem_id="home-nav",
                )

                gr.Markdown("LIBRARY", elem_classes=["sidebar-section-label"])

                ingestion_btn = gr.Button(
                    "▤  Document ingestion",
                    elem_id="ingestion-nav",
                )

                agents_btn = gr.Button(
                    "✦  Research workspace",
                    elem_id="research-nav",
                )

                papers_btn = gr.Button(
                    "▧  Research papers",
                    elem_id="papers-nav",
                )
                
                gr.Markdown("EVALUATION", elem_classes=["sidebar-section-label"])

                retriever_btn=gr.Button(
                    "⌕  Retriever",
                    elem_id="retriever-nav",
                )
                generator_btn=gr.Button(
                    "◉  Generator",
                    elem_id="generator-nav",
                )
                
                rag_evaluation_pipeline_btn=gr.Button(
                    "⤳  RAG pipeline",
                    elem_id="rag-nav",
                )

                gr.Markdown("PREFERENCES", elem_classes=["sidebar-section-label"])

                settings_btn = gr.Button(
                    "⚙  Settings",
                    elem_id="settings-nav",
                )

                with gr.Column(elem_id="sidebar-bottom"):
                    with gr.Accordion(
                        "◉ Profile",
                        open=False,
                        elem_id="user-profile",
                    ) as user_profile:
                        logout_button = gr.Button(
                            "↪  Log out",
                            variant="secondary",
                            elem_id="logout-button",
                        )

            # -------------------------------------------------
            # RIGHT CONTENT: page panels
            # -------------------------------------------------

            with gr.Column(
                scale=5,
                elem_id="main-content",
            ):

                # =========================
                # PAGE 0: HOME
                # =========================

                with gr.Column(visible=True) as home_page:
                    gr.Markdown(
                        "# ◈ Research Paper AI Assistant",
                        elem_classes=["page-title"],
                    )
                    gr.Markdown(
                        "A multi-agent system that ingests research papers, "
                        "generates embeddings, retrieves evidence, and produces "
                        "cited research reports — with built-in evaluation."
                    )

                    # Row 1: Core pipeline
                    with gr.Row(equal_height=True):
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown(
                                "### 📄 Data Ingestion Pipeline\n"
                                "Upload PDF / DOCX / TXT research papers. "
                                "The LangGraph pipeline validates, parses to Markdown, "
                                "chunks the text, and generates vector embeddings — "
                                "all in a single streaming workflow."
                            )
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown(
                                "### 🤖 Multi-Agent Research\n"
                                "Ask a question and six specialised agents collaborate: "
                                "**Orchestrator → Retriever → Research Analyst → "
                                "Critical Analyst → Citation Agent → Synthesis Writer** "
                                "to deliver an evidence-based research report."
                            )

                    # Row 2: Papers + Retrieval
                    with gr.Row(equal_height=True):
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown(
                                "### 📚 Research Papers Library\n"
                                "Browse every paper you have uploaded. "
                                "Track multiple versions of each document and "
                                "see which version is currently active."
                            )
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown(
                                "### 🔍 Hybrid Retrieval (Dense + Sparse + RRF)\n"
                                "Combines dense vector search, sparse BM25, "
                                "and Reciprocal Rank Fusion with a cross-encoder "
                                "re-ranker for high-precision chunk retrieval."
                            )

                    # Row 3: Evaluations
                    with gr.Row(equal_height=True):
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown(
                                "### 📊 Retriever Evaluation\n"
                                "Auto-generates a golden dataset, runs hybrid "
                                "retrieval, and scores results with **DeepEval** "
                                "metrics (Contextual Precision, Recall, Relevancy)."
                            )
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown(
                                "### 📊 Generator Evaluation\n"
                                "Builds faithfulness test cases, runs the full "
                                "multi-agent pipeline, and evaluates answers with "
                                "**Faithfulness** and **Answer Relevancy** metrics."
                            )

                    # Row 4: End-to-end + Tech
                    with gr.Row(equal_height=True):
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown(
                                "### ⚡ End-to-End RAG Evaluation\n"
                                "Combines retriever and generator evaluations into "
                                "a single pipeline — golden generation, live retrieval, "
                                "multi-agent answering, and DeepEval scoring."
                            )
                        with gr.Column(elem_classes=["feature-card"]):
                            gr.Markdown(
                                "### 🛠️ Tech Stack\n"
                                "**LangGraph** orchestration · **FastAPI** backend · "
                                "**Gradio** UI · **Qdrant** vector store · "
                                "**DeepEval** evaluation · **Google Gemini** LLM · "
                                "SSE streaming for real-time progress."
                            )

                # =========================
                # PAGE 1: DATA INGESTION
                # =========================

                with gr.Column(visible=False) as ingestion_page:
                    gr.Markdown("# Data Ingestion Pipeline")

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
                        "# Multi-Agent Research Workspace\n\n"
                        "Ask questions about your indexed research papers. "
                        "Multiple AI agents will collaborate to generate "
                        "an evidence-based research report."
                    )

                    # --------------------------------
                    # RESEARCH QUESTION
                    # --------------------------------

                    gr.Markdown("## Research Question")

                    query_input = gr.Textbox(
                        label="Enter your research question",
                        placeholder=(
                            "Example: Explain the abstract of "
                            "The Era of 1-bit LLMs."
                        ),
                        lines=3
                    )

                    research_run_button = gr.Button(
                        "Run Research Pipeline",
                        variant="primary"
                    )

                    with gr.Accordion("Details", open=False):
                        # --------------------------------
                        # LIVE EXECUTION TRACE
                        # --------------------------------

                        gr.Markdown("---")
                        gr.Markdown("## Agent Execution Monitor")

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
                        gr.Markdown("## Dense + Sparse + RRF + Encoder")
                        with gr.Accordion("Retrieved Chunks", open=False):
                            retrieved_chunks = gr.Markdown()

                        gr.Markdown("## Agent Analysis")

                        with gr.Accordion("1. Research Analyst", open=False):
                            analysis_output = gr.Markdown()

                        with gr.Accordion("2. Critical Analyst", open=False):
                            critical_output = gr.Markdown()

                        with gr.Accordion("3. Citation Agent", open=False):
                            cited_analysis_output = gr.Markdown()

                    # --------------------------------
                    # PIPELINE STATUS
                    # --------------------------------

                    pipeline_status = gr.Markdown()

                    # --------------------------------
                    # FINAL ANSWER
                    # --------------------------------

                    gr.Markdown("---")
                    gr.Markdown("## Final Research Report")

                    final_answer = gr.Markdown()

                    # --------------------------------
                    # PIPELINE EXECUTION
                    # --------------------------------

                    research_run_button.click(
                        fn=run_research_pipeline,
                        inputs=[query_input, current_user],
                        outputs=[
                            pipeline_status,
                            final_answer,
                            retrieved_chunks,
                            analysis_output,
                            critical_output,
                            cited_analysis_output,
                            execution_trace,
                            pipeline_progress,
                        ]
                    )

                # =========================
                # PAGE 4: RESEARCH PAPERS
                # =========================

                with gr.Column(visible=False) as papers_page:
                    gr.Markdown("# Research Papers")

                    gr.Markdown(
                        "Documents saved by your existing upload service."
                    )

                    refresh_papers_btn = gr.Button(
                        "Refresh Paper List"
                    )

                    papers_output = gr.Markdown(
                        "Click Refresh Paper List to view saved documents."
                    )

                    refresh_papers_btn.click(
                        fn=list_uploaded_papers,
                        inputs=current_user,
                        outputs=papers_output,
                    )

                # =========================
                # PAGE 5: RETRIEVER EVALUATIONS
                # =========================

                with gr.Column(visible=False) as retriever_evaluation_page:
                    with gr.Group():

                        gr.Markdown(
                            "# Retriever Evaluation Dashboard\n\n"
                            "Evaluate Hybrid Retrieval using DeepEval.\n\n"
                            "**Pipeline:** Golden Generation → Dataset Validation "
                            "→ Hybrid Retrieval → DeepEval → Results"
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
                            inputs=current_user,
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
                            "# Generator Evaluation Dashboard\n\n"
                            "Evaluate multi-agent answer generation using DeepEval.\n\n"
                            "**Pipeline:** Golden Generation → Validation "
                            "→ Multi-Agent Generation → Faithfulness "
                            "→ Answer Relevancy"
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
                            inputs=current_user,
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
                            "# End-to-End RAG Pipeline Evaluation\n\n"
                            "Evaluate live hybrid retrieval and multi-agent generation "
                            "using DeepEval metrics."
                        )

                        run_rag_btn = gr.Button(
                            "Run RAG Pipeline Evaluation",
                            variant="primary"
                        )

                        # =====================================
                        # EVALUATION STATUS
                        # =====================================

                        rag_golden_generation_status = gr.Markdown(
                            "⏳ Waiting for golden generation..."
                        )

                        rag_golden_validation_status = gr.Markdown(
                            "⏳ Waiting for dataset validation..."
                        )

                        rag_pipeline_status = gr.Markdown(
                            "⏳ Waiting for RAG pipeline execution..."
                        )

                        rag_evaluation_status = gr.Markdown(
                            "⏳ Waiting for DeepEval evaluation..."
                        )

                        # =====================================
                        # EVALUATION REPORT
                        # =====================================

                        gr.Markdown("## 📊 RAG Evaluation Report")

                        rag_evaluation_report = gr.Markdown(
                            "Your evaluation report will appear here."
                        )

                        # =====================================
                        # BUTTON EVENT
                        # =====================================

                        run_rag_btn.click(
                            fn=run_rag_pipeline_gradio,
                            inputs=current_user,
                            outputs=[
                                rag_golden_generation_status,
                                rag_golden_validation_status,
                                rag_pipeline_status,
                                rag_evaluation_status,
                                rag_evaluation_report,
                            ],
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

    login_event = login_button.click(
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
    login_event.then(
        fn=get_sidebar_user_name,
        inputs=current_user,
        outputs=user_profile,
    )

    logout_event = logout_button.click(
        fn=logout_user,
        inputs=None,
        outputs=[
            auth_page,
            protected_app,
            current_user
        ]
    )
    logout_event.then(
        fn=get_sidebar_user_name,
        inputs=current_user,
        outputs=user_profile,
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="indigo",
            neutral_hue="indigo",
        ),
        css=CSS,
    )
