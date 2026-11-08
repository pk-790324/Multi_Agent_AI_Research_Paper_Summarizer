
# from dotenv import load_dotenv

# from deepeval import evaluate
# from deepeval.test_case import LLMTestCase
# from deepeval.metrics import (
#     FaithfulnessMetric,
#     AnswerRelevancyMetric,
#     ContextualRelevancyMetric,
# )

# from src.rag_pipeline import RagPipeline
# from evals.harness import (
#     load_goldens,
#     summarize_by_metric,
#     print_summary,
# )

# from judge_llm.judge_llm import judge_llm

# load_dotenv()


# GOLDEN_PATH = "goldens/faithfulness_goldens.json"

# JUDGE_MODEL = judge_llm
# THRESHOLD = 0.7


# def run(rag):

#     # 1. LOAD GOLDENS
#     goldens = load_goldens(GOLDEN_PATH)
#     goldens=goldens[:1]

#     if not goldens:
#         raise ValueError("No golden test cases found.")

#     test_cases = []

#     # 2. RUN LIVE RAG PIPELINE
#     for index, golden in enumerate(goldens, start=1):

#         query = golden["query"]

#         print(f"\nRunning RAG evaluation {index}/{len(goldens)}")
#         print("Query:", query)

#         result = rag.invoke(query)

#         # 3. BUILD DEEPEVAL TEST CASE
#         test_case = LLMTestCase(
#             input=query,
#             actual_output=result["answer"],
#             retrieval_context=result["context"],
#         )

#         test_cases.append(test_case)

#     # 4. DEFINE METRICS
#     metrics = [
#         ContextualRelevancyMetric(
#             threshold=THRESHOLD,
#             model=JUDGE_MODEL,
#             include_reason=True,
#         ),

#         FaithfulnessMetric(
#             threshold=THRESHOLD,
#             model=JUDGE_MODEL,
#             include_reason=True,
#         ),

#         AnswerRelevancyMetric(
#             threshold=THRESHOLD,
#             model=JUDGE_MODEL,
#             include_reason=True,
#         ),
#     ]

#     # 5. RUN EVALUATION
#     evaluation_result = evaluate(
#         test_cases=test_cases,
#         metrics=metrics,
#     )

#     # Return raw result for the Gradio report formatter
#     return evaluation_result


# def run_local():

#     rag = RagPipeline()

#     return run(rag)


# if __name__ == "__main__":

#     result = run_local()

#     summary = summarize_by_metric(result)

#     print_summary("rag_pipeline", summary)


# in serial way



from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline
from evals.harness import (
    load_goldens,
    summarize_by_metric,
    print_summary,
)

from judge_llm.judge_llm import judge_llm

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_goldens.json"

JUDGE_MODEL = judge_llm
THRESHOLD = 0.7


def run(rag):

    # 1. LOAD GOLDENS
    goldens = load_goldens(GOLDEN_PATH)

    if not goldens:
        raise ValueError("No golden test cases found.")

    # 2. DEFINE METRICS
    metrics = [
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

    all_results = []

    # 3. SERIAL EVALUATION
    for index, golden in enumerate(goldens, start=1):

        query = golden["query"]

        print("\n" + "=" * 60)
        print(f"RAG EVALUATION {index}/{len(goldens)}")
        print("=" * 60)

        print("\nQUERY:", query)

        # A. RUN LIVE RAG
        rag_result = rag.invoke(query)

        # B. BUILD TEST CASE
        test_case = LLMTestCase(
            input=query,
            actual_output=rag_result["answer"],
            retrieval_context=rag_result["context"],
        )

        # C. EVALUATE THIS TEST CASE ONLY
        print("\nRunning DeepEval metrics...")

        evaluation_result = evaluate(
            test_cases=[test_case],
            metrics=metrics,
        )

        all_results.append(evaluation_result)

        print(f"\nCompleted test case {index}/{len(goldens)}")

    return all_results


def run_local():

    rag = RagPipeline()

    return run(rag)


if __name__ == "__main__":

    results = run_local()

    for index, result in enumerate(results, start=1):

        print("\n" + "#" * 60)
        print(f"RESULT FOR TEST CASE {index}")
        print("#" * 60)

        summary = summarize_by_metric(result)

        print_summary("rag_pipeline", summary)



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


def format_rag_result(result, index, query, answer):

    lines = [
        f"## Test Case {index}",
        f"**Query:** {query}",
        "",
        "### Generated Answer",
        answer,
        "",
        "### Metric Results",
    ]

    for test in result.test_results:

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


def run_rag_pipeline_gradio():

    goldens = load_goldens(GOLDEN_PATH)

    if not goldens:
        yield "No golden test cases found.", ""
        return

    rag = RagPipeline()
    metrics = create_metrics()

    total = len(goldens)
    report = ""

    for index, golden in enumerate(goldens, start=1):

        query = golden["query"]

        # Show progress before running this test
        yield (
            f"Running RAG evaluation: {index}/{total}",
            report
        )

        try:

            # 1. LIVE RAG EXECUTION
            rag_result = rag.invoke(query)

            # 2. BUILD TEST CASE
            test_case = LLMTestCase(
                input=query,
                actual_output=rag_result["answer"],
                retrieval_context=rag_result["context"],
            )

            # 3. EVALUATE ONE TEST CASE
            evaluation_result = evaluate(
                test_cases=[test_case],
                metrics=metrics,
            )

            # 4. FORMAT RESULT
            report += format_rag_result(
                evaluation_result,
                index,
                query,
                rag_result["answer"],
            )

            report += "\n\n---\n\n"

            # Update Gradio after each test case
            yield (
                f"Completed {index}/{total} test cases",
                report
            )

        except Exception as e:

            report += (
                f"## Test Case {index} - ERROR\n"
                f"**Query:** {query}\n\n"
                f"**Error:** `{str(e)}`\n\n---\n\n"
            )

            yield (
                f"Error in test case {index}/{total}: {str(e)}",
                report
            )

    yield (
        f"RAG evaluation completed: {total} test cases processed.",
        report
    )
