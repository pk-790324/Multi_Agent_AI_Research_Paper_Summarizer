
from dotenv import load_dotenv
from judge_llm.judge_llm import judge_llm
from deepeval import evaluate
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

def run(retriever):

    # 1. LOAD GOLDEN DATASET

    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []

    # 2. RETRIEVE DOCUMENTS

    for g in goldens:

        query = g["query"]

        retrieved = retriever.search(query)

        retrieval_context = extract_retrieval_context(retrieved)
        
        
        # Ground-truth context from golden dataset
        expected_retrieval_context = g[
            "expected_retrieval_context"
        ]

        # 3. BUILD TEST CASE

        test_case = LLMTestCase(

            input=query,

            expected_output=g["ideal_answer"],

            actual_output="Generator not evaluated",

            retrieval_context=retrieval_context,

            expected_retrieval_context=expected_retrieval_context

        )

        test_cases.append(test_case)

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

    report = []

    total_tests = len(test_results)
    passed_tests = 0

    metric_summary = {}

    report.append("# 🚀 DeepEval Retriever Evaluation Results\n")

    # =====================================
    # INDIVIDUAL TEST CASES
    # =====================================

    for i, test in enumerate(test_results):

        metrics_data = test.metrics_data

        test_passed = all(
            metric.success for metric in metrics_data
        )

        if test_passed:
            passed_tests += 1

        status = "✅ PASSED" if test_passed else "❌ FAILED"

        report.append(f"## {status} — test_case_{i}\n")

        report.append(f"**Input:** {test.input}\n")
        report.append(f"**Actual Output:** {test.actual_output}\n")
        report.append(f"**Expected Output:** {test.expected_output}\n")

        report.append("### Metrics\n")

        for metric in metrics_data:

            name = metric.name
            score = metric.score
            threshold = metric.threshold
            reason = metric.reason or "No reason provided"

            status = "PASS" if metric.success else "FAIL"

            report.append(f"**{status} — {name}**")

            report.append(f"- Score: {score}")
            report.append(f"- Threshold: {threshold}")
            report.append(f"- Reason: {reason}\n")

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

        report.append("---\n")

    # =====================================
    # AGGREGATE METRICS
    # =====================================

    report.append("# 📊 Aggregate Metrics\n")

    report.append(
        "| Metric | Average Score | Pass Rate | Passed | Failed | Total |"
    )
    report.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for name, data in metric_summary.items():

        scores = data["scores"]

        average = (
            sum(scores) / len(scores)
            if scores else 0
        )

        total = data["passed"] + data["failed"]

        pass_rate = (
            data["passed"] / total * 100
            if total else 0
        )

        report.append(
            f"| {name} | {average:.2f} "
            f"| {pass_rate:.2f}% "
            f"| {data['passed']} "
            f"| {data['failed']} "
            f"| {total} |"
        )

    # =====================================
    # OVERALL RESULTS
    # =====================================

    failed_tests = total_tests - passed_tests

    overall_pass_rate = (
        passed_tests / total_tests * 100
        if total_tests else 0
    )

    report.append("\n# 🎯 Overall Evaluation\n")

    report.append(f"- Total Tests: **{total_tests}**")
    report.append(f"- Passed: **{passed_tests}**")
    report.append(f"- Failed: **{failed_tests}**")
    report.append(f"- Pass Rate: **{overall_pass_rate:.2f}%**")

    return "\n\n".join(report)