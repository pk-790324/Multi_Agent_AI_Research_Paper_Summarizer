
import html

from dotenv import load_dotenv
from judge_llm.judge_llm import judge_llm
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric
)

from services.hybrid_search_service import HybridSearchService

from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary
)



load_dotenv()
# ==========================================
# helper function
# ==========================================

def extract_retrieval_context(retrieved):

    retrieval_context = []

    for doc in retrieved:

        if isinstance(doc, dict):

            content = (
                doc.get("text")
                or doc.get("page_content")
                or doc.get("content")
            )

        else:

            content = doc.page_content

        if content:
            retrieval_context.append(content)

    return retrieval_context

# ==========================================
# CONFIGURATION
# ==========================================

GOLDEN_PATH = "goldens/retriever_deepeval_goldens.json"

THRESHOLD = 0.7

JUDGE_MODEL ="My-Project-LLM"


# ==========================================
# RETRIEVER EVALUATION
# ==========================================

def run(retriever, owner_id, golden_path):

    goldens = load_goldens(golden_path)
    test_cases = []

    for g in goldens:

        query = g["query"]

        retrieved = retriever.search(
            query=query,
            user_id=owner_id,
            top_k=3
        )

        retrieval_context = extract_retrieval_context(
            retrieved
        )

        expected_retrieval_context = g[
            "expected_retrieval_context"
        ]

        test_case = LLMTestCase(
            input=query,
            expected_output=g["ideal_answer"],
            actual_output="Generator not evaluated",
            retrieval_context=retrieval_context,
            expected_retrieval_context=expected_retrieval_context
        )

        test_cases.append(test_case)

    # Keep your existing metrics and evaluate() code here.

    # 4. DEFINE METRICS

    metrics = [

        ContextualRecallMetric(
            threshold=THRESHOLD,
            model=judge_llm,
            include_reason=True
        ),

        ContextualPrecisionMetric(
            threshold=THRESHOLD,
            model=judge_llm,
            include_reason=True
        )

    ]

    # 5. RUN EVALUATION

    # 5. RUN EVALUATION

    result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False),
    hyperparameters={
        "retriever": "hybrid",
        "embedding_model": "your_embedding_model",
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "top_k": 3,
        "judge_model": JUDGE_MODEL,
        "golden_set": GOLDEN_PATH
    }
)

    return result


# evals/eval_retriever.py

def run_local():
    retriever = HybridSearchService()

    results = run(retriever)

    return results
# ==========================================
# STANDALONE EXECUTION
# ==========================================

if __name__ == "__main__":

    result = run_local()

    print_summary(
        "hybrid_retriever",
        summarize_by_metric(result)
    )



def format_evaluation_report(result):

    test_results = result.test_results

    total_tests = len(test_results)
    passed_tests = 0

    metric_summary = {}
    metric_names = []
    test_case_sections = []
    test_score_data = []

    for i, test in enumerate(test_results):

        metrics_data = test.metrics_data or []

        test_passed = all(
            metric.success for metric in metrics_data
        )

        if test_passed:
            passed_tests += 1

        status = "✅ PASSED" if test_passed else "❌ FAILED"

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

            # Aggregate scores
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
                <h3>{status} — test_case_{i}</h3>
                <p class="evaluation-value"><strong>Input:</strong> {html.escape(str(test.input))}</p>
                <p class="evaluation-value"><strong>Actual Output:</strong> {html.escape(str(test.actual_output))}</p>
                <p class="evaluation-value"><strong>Expected Output:</strong> {html.escape(str(test.expected_output))}</p>
                <h4>Metrics</h4>
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

        test_score_data.append((i, test_passed, test_scores))

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
    overall_pass_rate = passed_tests / total_tests * 100 if total_tests else 0
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
        <h2>🚀 DeepEval Retriever Evaluation Results</h2>

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

        <h2>🎯 Overall Evaluation</h2>
        <ul>
            <li>Total Tests: <strong>{total_tests}</strong></li>
            <li>Passed: <strong>{passed_tests}</strong></li>
            <li>Failed: <strong>{failed_tests}</strong></li>
            <li>Pass Rate: <strong>{overall_pass_rate:.2f}%</strong></li>
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