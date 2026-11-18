# evals/generator_evaluation.py

import json
import html
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

from evals.harness import load_goldens

from evals.generator_eval_graph import generator_eval_graph

from judge_llm.judge_llm import judge_llm


load_dotenv()


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

JUDGE_MODEL = judge_llm

THRESHOLD = 0.7


# ==========================================
# 1. GET USER-SPECIFIC GOLDEN PATH
# ==========================================

def get_golden_path(owner_id: str):

    return (
        BASE_DIR
        / "goldens"
        / owner_id
        / "faithfulness_goldens.json"
    )


# ==========================================
# 2. CONVERT GOLDEN INTO RETRIEVED CHUNKS
# ==========================================

def build_retrieved_chunks(golden):

    return [{

        "page_content": golden["ideal_context"],

        "chunk_id": golden.get("chunk_id"),

        "document_id": golden.get("document_id"),

        "version": golden.get("version"),

        "paper_title": golden.get("paper_title"),

        "section": golden.get("section"),

        "page_start": golden.get("page_start"),

        "page_end": golden.get("page_end"),

        "source_file": golden.get("source_file"),

        "char_count": golden.get("char_count"),

        "created_at": golden.get("created_at"),

        "rank": 1,

        "score": 1.0,

    }]


# ==========================================
# 3. RUN MULTI-AGENT GENERATOR
# ==========================================

def run_generator(query, golden):

    golden_chunks = build_retrieved_chunks(golden)

    initial_state = {

        "user_query": query,

        # Use golden context as evaluation reference
        "retrieved_chunks": golden_chunks,

        "retrieval_count": len(golden_chunks),

        # Agent outputs
        "analysis": "",

        "critical_analysis": "",

        "cited_analysis": "",

        "citations": [],

        # Final output
        "final_answer": "",

        # Orchestrator
        "next_agent": "retriever",

        "completed_agents": [],

        # Error handling
        "error": None,

    }

    result = generator_eval_graph.invoke(
        initial_state,
        config={
            "recursion_limit": 30
        }
    )

    if result.get("error"):
        raise RuntimeError(result["error"])

    answer = result.get("final_answer")

    if not answer:
        raise ValueError(
            "Generator graph returned an empty answer."
        )

    return answer


# ==========================================
# 4. FORMAT GENERATOR REPORT
# ==========================================

def format_generator_report(result):

    test_results = result.test_results

    total_tests = len(test_results)
    passed_tests = 0
    metric_summary = {}
    metric_names = []
    test_case_sections = []
    test_score_data = []

    for index, test in enumerate(test_results):

        metrics_data = test.metrics_data or []

        test_passed = all(
            metric.success for metric in metrics_data
        )

        if test_passed:
            passed_tests += 1

        status = (
            "✅ PASSED"
            if test_passed
            else "❌ FAILED"
        )

        metric_rows = []
        test_scores = {}

        for metric in metrics_data:

            name = str(metric.name)
            score = metric.score
            threshold = metric.threshold
            reason = metric.reason or "No reason provided"
            metric_status = "PASS" if metric.success else "FAIL"
            score_text = (
                "N/A"
                if score is None
                else f"{score:.2f}"
                if isinstance(score, (int, float))
                else str(score)
            )
            threshold_text = (
                "N/A"
                if threshold is None
                else f"{threshold:.2f}"
                if isinstance(threshold, (int, float))
                else str(threshold)
            )

            if name not in metric_names:
                metric_names.append(name)

            test_scores[name] = f"{score_text} ({metric_status})"

            metric_rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{html.escape(score_text)}</td>"
                f"<td>{html.escape(threshold_text)}</td>"
                f"<td>{metric_status}</td>"
                f"<td>{html.escape(str(reason))}</td>"
                "</tr>"
            )

            if name not in metric_summary:
                metric_summary[name] = {
                    "scores": [],
                    "passed": 0,
                    "failed": 0
                }

            if score is not None:
                metric_summary[name]["scores"].append(score)

            if metric.success:
                metric_summary[name]["passed"] += 1
            else:
                metric_summary[name]["failed"] += 1

        metric_body = "".join(metric_rows) or (
            '<tr><td colspan="5">No metrics were returned.</td></tr>'
        )

        test_case_sections.append(
            f"""
            <section class="evaluation-test-case">
                <h3>{status} — test_case_{index + 1}</h3>
                <p class="evaluation-value"><strong>Input:</strong> {html.escape(str(test.input))}</p>
                <p class="evaluation-value"><strong>Generated Answer:</strong></p>
                <div class="evaluation-value">{html.escape(str(test.actual_output))}</div>
                <h4>Evaluation Metrics</h4>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Score</th>
                                <th>Threshold</th>
                                <th>Result</th>
                                <th>Reason</th>
                            </tr>
                        </thead>
                        <tbody>{metric_body}</tbody>
                    </table>
                </div>
            </section>
            """
        )

        test_score_data.append((index + 1, test_passed, test_scores))

    aggregate_rows = []

    for name, data in metric_summary.items():

        scores = data["scores"]
        average = sum(scores) / len(scores) if scores else 0
        total = data["passed"] + data["failed"]
        pass_rate = data["passed"] / total * 100 if total else 0

        aggregate_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{average:.2f}</td>"
            f"<td>{pass_rate:.2f}%</td>"
            f"<td>{data['passed']}</td>"
            f"<td>{data['failed']}</td>"
            f"<td>{total}</td>"
            "</tr>"
        )

    score_table_headers = "".join(
        f"<th>{html.escape(name)}</th>"
        for name in metric_names
    )
    score_table_rows = []

    for index, test_passed, test_scores in test_score_data:
        score_cells = "".join(
            f"<td>{html.escape(test_scores.get(name, 'N/A'))}</td>"
            for name in metric_names
        )
        score_table_rows.append(
            "<tr>"
            f"<td>test_case_{index}</td>"
            f"{score_cells}"
            f"<td>{'Passed' if test_passed else 'Failed'}</td>"
            "</tr>"
        )

    failed_tests = total_tests - passed_tests
    pass_rate = passed_tests / total_tests * 100 if total_tests else 0
    test_case_content = "".join(test_case_sections) or (
        "<p>No test cases were evaluated.</p>"
    )
    aggregate_body = "".join(aggregate_rows) or (
        '<tr><td colspan="6">No metrics were evaluated.</td></tr>'
    )
    score_table_body = "".join(score_table_rows) or (
        f'<tr><td colspan="{len(metric_names) + 2}">No test cases were evaluated.</td></tr>'
    )

    return f"""
    <div class="evaluation-report">
        <h2>🚀 DeepEval Generator Evaluation Results</h2>

        <details class="evaluation-test-cases">
            <summary>Test Cases ({total_tests})</summary>
            {test_case_content}
        </details>

        <h2>📊 Aggregate Metrics</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Average Score</th>
                        <th>Pass Rate</th>
                        <th>Passed</th>
                        <th>Failed</th>
                        <th>Total</th>
                    </tr>
                </thead>
                <tbody>{aggregate_body}</tbody>
            </table>
        </div>

        <h2>🎯 Overall Generator Evaluation</h2>
        <ul>
            <li>Total Tests: <strong>{total_tests}</strong></li>
            <li>Passed: <strong>{passed_tests}</strong></li>
            <li>Failed: <strong>{failed_tests}</strong></li>
            <li>Overall Pass Rate: <strong>{pass_rate:.2f}%</strong></li>
        </ul>

        <h2>Test Case Score Summary</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Test Case</th>
                        {score_table_headers}
                        <th>Overall Result</th>
                    </tr>
                </thead>
                <tbody>{score_table_body}</tbody>
            </table>
        </div>
    </div>
    """


# ==========================================
# 5. RUN GENERATOR EVALUATION FOR ONE USER
# ==========================================

def run_generator_evaluation(current_user: dict):

    if not current_user or current_user.get("_id") is None:

        raise ValueError(
            "An authenticated user is required."
        )

    owner_id = str(current_user["_id"])

    print("\n" + "=" * 60)

    print("GENERATOR EVALUATION")

    print("=" * 60)

    # --------------------------------------
    # LOAD USER-SPECIFIC GOLDENS
    # --------------------------------------

    golden_path = get_golden_path(owner_id)

    if not golden_path.exists():

        raise FileNotFoundError(
            f"Generator golden dataset not found: {golden_path}"
        )

    goldens = load_goldens(str(golden_path))

    if not goldens:

        raise ValueError(
            "Generator golden dataset is empty."
        )

    print(f"Owner ID: {owner_id}")

    print(f"Total goldens: {len(goldens)}")

    test_cases = []

    errors = []

    # --------------------------------------
    # RUN GENERATOR
    # --------------------------------------

    for index, golden in enumerate(
        goldens,
        start=1
    ):

        query = golden["query"]

        ideal_context = golden["ideal_context"]

        evaluation_context = (
            [ideal_context]
            if isinstance(ideal_context, str)
            else ideal_context
        )

        print("\n" + "=" * 60)

        print(
            f"GENERATOR TEST: {index}/{len(goldens)}"
        )

        print("=" * 60)

        try:

            answer = run_generator(
                query=query,
                golden=golden
            )

            print("\nGenerated Answer:")

            print(answer)

            test_case = LLMTestCase(

                input=query,

                actual_output=answer,

                retrieval_context=evaluation_context,

            )

            test_cases.append(test_case)

        except Exception as error:

            print(
                f"Failed to process sample {index}: {error}"
            )

            errors.append({

                "test_id": golden.get(
                    "id",
                    f"g{index:03d}"
                ),

                "query": query,

                "error": str(error)

            })

    if not test_cases:

        raise ValueError(
            "No valid generator test cases found."
        )

    # --------------------------------------
    # DEFINE METRICS
    # --------------------------------------

    metrics = [

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

    # --------------------------------------
    # RUN DEEPEVAL
    # --------------------------------------

    print("\nRunning DeepEval metrics...")

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    # --------------------------------------
    # FORMAT REPORT
    # --------------------------------------

    report = format_generator_report(result)

    # --------------------------------------
    # STRUCTURED RESULTS
    # --------------------------------------

    test_results = []

    for index, test in enumerate(
        result.test_results,
        start=1
    ):

        metric_results = []

        for metric in test.metrics_data:

            metric_results.append({

                "name": metric.name,

                "score": metric.score,

                "threshold": metric.threshold,

                "success": metric.success,

                "reason": metric.reason,

            })

        test_results.append({

            "test_id": f"test_{index}",

            "input": test.input,

            "actual_output": test.actual_output,

            "passed": all(
                metric.success
                for metric in test.metrics_data
            ),

            "metrics": metric_results

        })

    passed_tests = sum(
        1
        for test in test_results
        if test["passed"]
    )

    total_tests = len(test_results)

    return {

        "status": "completed",

        "owner_id": owner_id,

        "total_goldens": len(goldens),

        "total_tests": total_tests,

        "passed": passed_tests,

        "failed": total_tests - passed_tests,

        "pass_rate": round(
            passed_tests / total_tests * 100,
            2
        ) if total_tests else 0,

        "errors": errors,

        "results": test_results,

        "report": report,

    }