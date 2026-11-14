from pathlib import Path
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline
from evals.harness import load_goldens
from judge_llm.judge_llm import judge_llm

load_dotenv()


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

THRESHOLD = 0.7
JUDGE_MODEL = judge_llm


# =========================================================
# 1. USER-SPECIFIC GOLDEN PATH
# =========================================================

def get_golden_path(owner_id: str) -> Path:

    return (
        BASE_DIR
        / "goldens"
        / owner_id
        / "faithfulness_goldens.json"
    )


# =========================================================
# 2. CREATE DEEPEVAL METRICS
# =========================================================

def create_metrics():

    return [
        ContextualRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),

        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),

        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]


# =========================================================
# 3. FORMAT ONE TEST CASE RESULT
# =========================================================

def format_rag_result(
    evaluation_result,
    index: int,
    query: str,
    answer: str,
):

    lines = [
        f"## Test Case {index}",
        f"**Query:** {query}",
        "",
        "### Generated Answer",
        answer,
        "",
        "### Metric Results",
    ]

    for test in evaluation_result.test_results:

        for metric in test.metrics_data:

            score = metric.score
            threshold = metric.threshold
            success = metric.success
            reason = metric.reason

            status = "PASS" if success else "FAIL"

            lines.extend([
                f"#### {metric.name}: {status}",
                f"- **Score:** {score}",
                f"- **Threshold:** {threshold}",
                f"- **Reason:** {reason or 'No reason provided'}",
                "",
            ])

    return "\n".join(lines)


# =========================================================
# 4. MAIN RAG EVALUATION FUNCTION
# =========================================================

def run_rag_evaluation(current_user: dict):

    # =====================================
    # AUTHENTICATED USER
    # =====================================

    if not current_user or not current_user.get("_id"):
        raise ValueError(
            "Authenticated user information is missing."
        )

    owner_id = str(current_user["_id"])

    # =====================================
    # LOAD USER-SPECIFIC GOLDENS
    # =====================================

    golden_path = get_golden_path(owner_id)

    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found for this user: {golden_path}"
        )

    goldens = load_goldens(str(golden_path))

    if not goldens:
        raise ValueError(
            "No RAG evaluation golden test cases found."
        )

    # =====================================
    # INITIALIZE USER-SCOPED RAG PIPELINE
    # =====================================

    rag = RagPipeline(owner_id=owner_id)

    metrics = create_metrics()

    total = len(goldens)

    all_reports = []
    errors = []

    passed_cases = 0
    failed_cases = 0

    # =====================================
    # SERIAL RAG EVALUATION
    # =====================================

    for index, golden in enumerate(goldens, start=1):

        query = golden.get("query")

        if not query:
            errors.append(
                f"Test case {index}: Missing query in golden dataset."
            )
            failed_cases += 1
            continue

        try:

            # ---------------------------------
            # A. RUN LIVE RAG PIPELINE
            # ---------------------------------

            rag_result = rag.invoke(query)

            answer = rag_result.get("answer", "")
            context = rag_result.get("context", [])

            if not answer:
                raise ValueError(
                    "RAG pipeline returned an empty answer."
                )

            if not context:
                raise ValueError(
                    "RAG pipeline returned an empty retrieval context."
                )

            # ---------------------------------
            # B. BUILD DEEPEVAL TEST CASE
            # ---------------------------------

            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=context,
            )

            # ---------------------------------
            # C. RUN DEEPEVAL
            # ---------------------------------

            evaluation_result = evaluate(
                test_cases=[test_case],
                metrics=metrics,
            )

            # ---------------------------------
            # D. FORMAT RESULT
            # ---------------------------------

            case_report = format_rag_result(
                evaluation_result=evaluation_result,
                index=index,
                query=query,
                answer=answer,
            )

            all_reports.append(case_report)

            # ---------------------------------
            # E. COUNT PASSED / FAILED CASES
            # ---------------------------------

            case_metrics = [
                metric
                for test in evaluation_result.test_results
                for metric in test.metrics_data
            ]

            if case_metrics and all(
                metric.success for metric in case_metrics
            ):
                passed_cases += 1
            else:
                failed_cases += 1

        except Exception as exc:

            failed_cases += 1

            errors.append(
                f"Test case {index} failed: {str(exc)}"
            )

            all_reports.append(
                f"## Test Case {index} - ERROR\n"
                f"**Query:** {query}\n\n"
                f"**Error:** `{str(exc)}`"
            )

    # =====================================
    # BUILD FINAL REPORT
    # =====================================

    total_processed = passed_cases + failed_cases

    pass_rate = (
        (passed_cases / total_processed) * 100
        if total_processed
        else 0
    )

    report = [
        "# End-to-End RAG Pipeline Evaluation",
        "",
        "## Evaluation Summary",
        f"- **Total golden cases:** {total}",
        f"- **Passed cases:** {passed_cases}",
        f"- **Failed cases:** {failed_cases}",
        f"- **Pass rate:** {pass_rate:.2f}%",
        "",
        "---",
        "",
        "\n\n---\n\n".join(all_reports),
    ]

    if errors:
        report.extend([
            "",
            "## Errors",
            *[f"- {error}" for error in errors],
        ])

    return {
        "status": (
            "success"
            if failed_cases == 0
            else "partial"
        ),
        "total_goldens": total,
        "total_tests": total_processed,
        "passed": passed_cases,
        "failed": failed_cases,
        "pass_rate": round(pass_rate, 2),
        "errors": errors,
        "report": "\n".join(report),
    }




# =========================================================
# RAG PIPELINE EVALUATION
# =========================================================

import requests



# =========================================================
# RAG PIPELINE EVALUATION
# =========================================================

import requests


def run_rag_pipeline_gradio(current_user):

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
            "❌ Please log in to run RAG pipeline evaluation.",
            "⏹ Evaluation stopped.",
            "⏹ Evaluation stopped.",
            "⏹ Evaluation stopped.",
            "❌ Please log in to view the report.",
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
        "🔄 **Starting authenticated RAG evaluation...**",
        "⏳ Waiting for golden generation...",
        "⏳ Waiting for RAG pipeline execution...",
        "⏳ Waiting for DeepEval evaluation...",
        "Evaluation has not started yet.",
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

        if golden_data.get("status") not in (
            "success",
            "completed",
        ):
            raise ValueError(
                "Golden generation was not confirmed successful."
            )

        total_goldens = golden_data.get("total_goldens")

        if total_goldens is not None and total_goldens == 0:
            raise ValueError(
                "Generated golden dataset is empty."
            )

        # =====================================
        # STEP 2: GOLDEN VALIDATION COMPLETED
        # =====================================

        yield (
            "✅ **Golden generation completed.**",
            f"✅ **Dataset validated: {total_goldens if total_goldens is not None else 'Available'} test cases.**",
            "🔄 **Preparing live RAG pipeline...**",
            "⏳ Waiting for DeepEval evaluation...",
            "",
        )

        # =====================================
        # STEP 3: RUN RAG EVALUATION
        # =====================================

        evaluation_response = requests.post(
            f"{base_url}/rag",
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
                f"RAG evaluation API returned "
                f"{evaluation_response.status_code}: {detail}"
            )

        evaluation_data = evaluation_response.json()

        # =====================================
        # STEP 4: VALIDATE REPORT
        # =====================================

        report = evaluation_data.get("report")

        if not isinstance(report, str) or not report.strip():
            raise ValueError(
                "RAG evaluation API returned no results report."
            )

        # =====================================
        # STEP 5: FINAL RESULTS
        # =====================================

        yield (
            "✅ **Golden generation completed.**",
            "✅ **Golden dataset validated.**",
            "✅ **RAG pipeline execution completed.**",
            "✅ **DeepEval evaluation completed.**",
            report,
        )

    except (
        requests.exceptions.RequestException,
        ValueError,
        RuntimeError,
    ) as exc:

        yield (
            "❌ **RAG pipeline evaluation failed.**",
            "⏹ Evaluation stopped.",
            "⏹ Evaluation stopped.",
            "❌ **Error occurred.**",
            f"```text\n{exc}\n```",
        )

