# evals/generator_evaluation.py

import json
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

    report = []

    total_tests = len(test_results)

    passed_tests = 0

    metric_summary = {}

    report.append(
        "# 🚀 DeepEval Generator Evaluation Results"
    )

    # =====================================
    # INDIVIDUAL TEST CASES
    # =====================================

    for index, test in enumerate(test_results):

        metrics_data = test.metrics_data

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

        report.append(
            f"## {status} — test_case_{index + 1}"
        )

        report.append(
            f"**Input:** {test.input}"
        )

        report.append(
            f"**Generated Answer:**\n\n{test.actual_output}"
        )

        report.append("### Evaluation Metrics")

        for metric in metrics_data:

            name = metric.name

            score = metric.score

            threshold = metric.threshold

            reason = (
                metric.reason
                or "No reason provided"
            )

            metric_status = (
                "PASS"
                if metric.success
                else "FAIL"
            )

            report.append(
                f"#### {metric_status} — {name}"
            )

            report.append(f"- **Score:** {score}")

            report.append(
                f"- **Threshold:** {threshold}"
            )

            report.append(
                f"- **Reason:** {reason}"
            )

            if name not in metric_summary:

                metric_summary[name] = {
                    "scores": [],
                    "passed": 0,
                    "failed": 0
                }

            if score is not None:

                metric_summary[name]["scores"].append(
                    score
                )

            if metric.success:

                metric_summary[name]["passed"] += 1

            else:

                metric_summary[name]["failed"] += 1

        report.append("---")

    # =====================================
    # AGGREGATE METRICS
    # =====================================

    report.append("# 📊 Aggregate Metrics")

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

        total = (
            data["passed"] + data["failed"]
        )

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

    pass_rate = (
        passed_tests / total_tests * 100
        if total_tests else 0
    )

    report.append(
        "# 🎯 Overall Generator Evaluation"
    )

    report.append(
        f"- Total Tests: **{total_tests}**"
    )

    report.append(
        f"- Passed: **{passed_tests}**"
    )

    report.append(
        f"- Failed: **{failed_tests}**"
    )

    report.append(
        f"- Overall Pass Rate: **{pass_rate:.2f}%**"
    )

    return "\n\n".join(report)


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