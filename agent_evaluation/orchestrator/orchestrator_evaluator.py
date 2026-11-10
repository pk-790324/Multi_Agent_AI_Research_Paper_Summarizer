
import json
from pathlib import Path
from copy import deepcopy

from agents.orchestrator_agent import orchestrator_agent


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

GOLDEN_PATH = (
    BASE_DIR
    / "agent_evaluation"
    / "orchestrator"
    / "orchestrator_goldens.json"
)

RESULT_PATH = (
    BASE_DIR
    / "agent_evaluation"
    / "orchestrator"
    / "orchestrator_results.json"
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_goldens():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def is_empty(value):
    """Accept common representations of cleared state values."""
    return value is None or value == "" or value == [] or value == {}


def check_feedback(test_id, state, output):
    """
    Validate feedback propagation and state cleanup.
    Returns None for tests that do not require feedback checks.
    """

    if test_id == "T13":
        expected_feedback = state.get("critique", "")
        return output.get("analyst_feedback") == expected_feedback

    if test_id == "T14":
        expected_feedback = state.get("critical_analysis", "")
        return output.get("analyst_feedback") == expected_feedback

    if test_id == "T15":
        expected_feedback = state.get("critical_analysis", "")

        feedback_received = (
            output.get("retrieval_feedback") == expected_feedback
        )

        analysis_cleared = is_empty(output.get("analysis"))
        critique_cleared = is_empty(output.get("critical_analysis"))
        cited_analysis_cleared = is_empty(output.get("cited_analysis"))
        citations_cleared = is_empty(output.get("citations"))

        return all([
            feedback_received,
            analysis_cleared,
            critique_cleared,
            cited_analysis_cleared,
            citations_cleared
        ])

    return None


# --------------------------------------------------
# RUN ORCHESTRATOR EVALUATION
# --------------------------------------------------

def evaluate_orchestrator():

    goldens = load_goldens()
    total = len(goldens)

    # LLM metrics
    llm_correct = 0
    llm_successful = 0

    # Final routing metrics
    final_correct = 0

    # Execution metrics
    fallback_count = 0
    override_count = 0
    llm_skipped_count = 0
    error_count = 0

    # Feedback metrics
    feedback_tests = 0
    feedback_correct_count = 0

    results = []

    print("\n" + "=" * 70)
    print("ORCHESTRATOR EVALUATION")
    print("=" * 70)

    for index, test in enumerate(goldens, start=1):

        test_id = test["id"]
        scenario = test["scenario"]

        state = deepcopy(test["input"])

        # Expected final decision
        expected_final = test["expected_agent"]

        # Expected raw LLM decision
        # Defaults to expected_agent for older golden cases.
        expected_llm = test.get(
            "expected_llm_agent",
            expected_final
        )

        print(f"\nTEST {index}/{total}: {test_id}")
        print("Scenario:", scenario)

        try:

            # --------------------------------------
            # EXECUTE ACTUAL ORCHESTRATOR
            # --------------------------------------

            output = orchestrator_agent(state)

            llm_agent = output.get("llm_selected_agent")
            final_agent = output.get("next_agent")

            routing_reason = output.get("routing_reason")

            # --------------------------------------
            # DETECT FALLBACK AND LLM SKIP
            # --------------------------------------

            fallback_used = (
                routing_reason == "Deterministic fallback routing."
            )

            llm_skipped = (
                llm_agent is None and not fallback_used
            )

            if fallback_used:
                fallback_count += 1

            if llm_skipped:
                llm_skipped_count += 1

            # --------------------------------------
            # LLM DECISION EVALUATION
            # --------------------------------------

            llm_available = llm_agent is not None

            if llm_available:

                llm_successful += 1

                llm_is_correct = (
                    llm_agent == expected_llm
                )

                if llm_is_correct:
                    llm_correct += 1

            else:
                llm_is_correct = None

            # --------------------------------------
            # FINAL ROUTING EVALUATION
            # --------------------------------------

            final_is_correct = (
                final_agent == expected_final
            )

            if final_is_correct:
                final_correct += 1

            # --------------------------------------
            # PYTHON OVERRIDE DETECTION
            # --------------------------------------

            was_overridden = (
                llm_available
                and llm_agent != final_agent
            )

            if was_overridden:
                override_count += 1

            # --------------------------------------
            # FEEDBACK PROPAGATION EVALUATION
            # --------------------------------------

            feedback_result = check_feedback(
                test_id,
                state,
                output
            )

            if feedback_result is not None:

                feedback_tests += 1

                if feedback_result:
                    feedback_correct_count += 1

            # --------------------------------------
            # STORE RESULTS
            # --------------------------------------

            result = {
                "id": test_id,
                "scenario": scenario,

                "expected_llm_agent": expected_llm,
                "llm_selected_agent": llm_agent,
                "llm_correct": llm_is_correct,

                "expected_agent": expected_final,
                "final_agent": final_agent,
                "final_correct": final_is_correct,

                "fallback_used": fallback_used,
                "llm_skipped": llm_skipped,
                "python_override": was_overridden,

                "feedback_correct": feedback_result,

                "routing_reason": routing_reason,
                "error": None
            }

            results.append(result)

            # --------------------------------------
            # PRINT TEST RESULT
            # --------------------------------------

            print("Expected LLM Agent:", expected_llm)
            print("LLM Selected Agent:", llm_agent)
            print("LLM Correct:", llm_is_correct)

            print("Expected Final Agent:", expected_final)
            print("Final Agent:", final_agent)
            print("Final Correct:", final_is_correct)

            print("Fallback Used:", fallback_used)
            print("LLM Skipped:", llm_skipped)
            print("Python Override:", was_overridden)

            if feedback_result is not None:
                print("Feedback Correct:", feedback_result)

        except Exception as error:

            error_count += 1

            print("[ERROR]", str(error))

            results.append({
                "id": test_id,
                "scenario": scenario,

                "expected_llm_agent": expected_llm,
                "llm_selected_agent": None,
                "llm_correct": None,

                "expected_agent": expected_final,
                "final_agent": None,
                "final_correct": False,

                "fallback_used": False,
                "llm_skipped": False,
                "python_override": False,

                "feedback_correct": None,

                "routing_reason": None,
                "error": str(error)
            })

    # --------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------

    # Raw LLM accuracy:
    # Correct LLM decisions / successful LLM calls.

    llm_accuracy = (
        llm_correct / llm_successful * 100
        if llm_successful else 0
    )

    # Final orchestrator accuracy:
    # Correct final decisions / all test cases.

    final_accuracy = (
        final_correct / total * 100
        if total else 0
    )

    # LLM coverage:
    # Successful LLM decisions / all test cases.

    llm_coverage = (
        llm_successful / total * 100
        if total else 0
    )

    fallback_rate = (
        fallback_count / total * 100
        if total else 0
    )

    override_rate = (
        override_count / llm_successful * 100
        if llm_successful else 0
    )

    feedback_accuracy = (
        feedback_correct_count / feedback_tests * 100
        if feedback_tests else 0
    )

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    report = {
        "total_tests": total,
        "error_count": error_count,

        "llm_correct": llm_correct,
        "llm_successful": llm_successful,
        "llm_accuracy": round(llm_accuracy, 2),
        "llm_accuracy_when_available": round(llm_accuracy, 2),
        "llm_coverage": round(llm_coverage, 2),

        "final_correct": final_correct,
        "final_accuracy": round(final_accuracy, 2),

        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_rate, 2),

        "llm_skipped_count": llm_skipped_count,

        "override_count": override_count,
        "override_rate": round(override_rate, 2),

        "feedback_tests": feedback_tests,
        "feedback_correct": feedback_correct_count,
        "feedback_accuracy": round(feedback_accuracy, 2),

        "results": results
    }

    # --------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total Tests: {total}")
    print(f"Execution Errors: {error_count}")

    print("\n--- LLM PERFORMANCE ---")
    print(f"LLM Successful Calls: {llm_successful}")
    print(f"LLM Skipped Calls: {llm_skipped_count}")
    print(f"LLM Coverage: {llm_coverage:.2f}%")
    print(f"LLM Routing Accuracy: {llm_accuracy:.2f}%")

    print("\n--- FINAL ORCHESTRATOR PERFORMANCE ---")
    print(f"Final Correct: {final_correct}/{total}")
    print(f"Final Routing Accuracy: {final_accuracy:.2f}%")

    print("\n--- PYTHON CONTROL ---")
    print(f"Fallback Count: {fallback_count}")
    print(f"Fallback Rate: {fallback_rate:.2f}%")
    print(f"Python Override Count: {override_count}")
    print(f"Python Override Rate: {override_rate:.2f}%")

    print("\n--- FEEDBACK PROPAGATION ---")
    print(f"Feedback Tests: {feedback_tests}")
    print(f"Feedback Correct: {feedback_correct_count}")
    print(f"Feedback Accuracy: {feedback_accuracy:.2f}%")

    # --------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nResults saved to:", RESULT_PATH)

    return report


if __name__ == "__main__":
    evaluate_orchestrator()
