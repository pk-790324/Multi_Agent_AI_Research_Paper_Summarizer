from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary,
)

from evals.generator_eval_graph import generator_eval_graph
from judge_llm.judge_llm import judge_llm

load_dotenv()


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

GOLDEN_PATH = "goldens/faithfulness_goldens.json"

JUDGE_MODEL = judge_llm

THRESHOLD = 0.7


# --------------------------------------------------
# CONVERT GOLDEN INTO RETRIEVED CHUNK FORMAT
# --------------------------------------------------

def build_retrieved_chunks(golden):

    return [{
        "page_content": golden["ideal_context"],

        "chunk_id": golden.get("chunk_id"),
        "paper_title": golden.get("paper_title"),
        "section": golden.get("section"),
        "page_start": golden.get("page_start"),
        "page_end": golden.get("page_end"),
        "source_file": golden.get("source_file"),
        "char_count": golden.get("char_count"),
        "created_at": golden.get("created_at"),

        # Evaluation metadata
        "rank": 1,
        "score": 1.0,
    }]

# --------------------------------------------------
# RUN MULTI-AGENT GENERATOR
# --------------------------------------------------

def run_generator(query, golden):

    golden_chunks = build_retrieved_chunks(golden)

    initial_state = {

        # User input
        "user_query": query,

        # Golden context instead of Qdrant retrieval
        "retrieved_chunks": golden_chunks,
        "retrieval_count": len(golden_chunks),

        # Agent outputs
        "analysis": "",
        "critical_analysis": "",
        "cited_analysis": "",
        "citations": [],

        # Final output
        "final_answer": "",

        # Orchestrator control
        "next_agent": "retriever",
        "completed_agents": [],

        # Error handling
        "error": None,
    }

    # Execute the evaluation graph
    result = generator_eval_graph.invoke(
        initial_state,
        config={
            "recursion_limit": 30
        }
    )

    return result["final_answer"]


# --------------------------------------------------
# GENERATOR EVALUATION
# --------------------------------------------------

def run():

    # 1. Load golden dataset
    goldens = load_goldens(GOLDEN_PATH)
    goldens=goldens[:1]

    if not goldens:
        raise ValueError("Generator golden dataset is empty.")

    test_cases = []

    # 2. Run multi-agent generator
    for index, golden in enumerate(goldens, start=1):

        query = golden["query"]

        ideal_context = golden["ideal_context"]

        evaluation_context = (
            [ideal_context]
            if isinstance(ideal_context, str)
            else ideal_context
        )

        print("\n" + "=" * 70)
        print(f"GENERATOR EVALUATION: {index}/{len(goldens)}")
        print("=" * 70)

        print(f"\nQuery: {query}")

        try:

            answer = run_generator(
                query=query,
                golden=golden,
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

            print(f"Failed to process sample {index}: {error}")

    if not test_cases:
        raise ValueError(
            "No valid generator test cases found."
        )

    # 3. Define metrics
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

    # 4. Run DeepEval
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    return result

   


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    result = run()

    print_summary(
        "generator",
        summarize_by_metric(result)
    )


def format_generator_report(result):

    test_results = result.test_results

    report = []

    total_tests = len(test_results)
    passed_tests = 0

    metric_summary = {}

    report.append("# 🚀 DeepEval Generator Evaluation Results")

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

        status = "✅ PASSED" if test_passed else "❌ FAILED"

        report.append(
            f"## {status} — test_case_{index}"
        )

        report.append(f"**Input:** {test.input}")

        report.append(
            f"**Generated Answer:**\n\n{test.actual_output}"
        )

        report.append("### Evaluation Metrics")

        for metric in metrics_data:

            name = metric.name
            score = metric.score
            threshold = metric.threshold

            reason = metric.reason or "No reason provided"

            metric_status = (
                "PASS" if metric.success else "FAIL"
            )

            report.append(
                f"#### {metric_status} — {name}"
            )

            report.append(f"- **Score:** {score}")
            report.append(f"- **Threshold:** {threshold}")
            report.append(f"- **Reason:** {reason}")

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

    pass_rate = (
        passed_tests / total_tests * 100
        if total_tests else 0
    )

    report.append("# 🎯 Overall Generator Evaluation")

    report.append(f"- Total Tests: **{total_tests}**")
    report.append(f"- Passed: **{passed_tests}**")
    report.append(f"- Failed: **{failed_tests}**")
    report.append(f"- Overall Pass Rate: **{pass_rate:.2f}%**")

    return "\n\n".join(report)