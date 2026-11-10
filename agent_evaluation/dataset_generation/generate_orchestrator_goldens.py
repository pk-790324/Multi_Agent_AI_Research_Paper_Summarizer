
import json
import sys
import argparse
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from config import MAX_AGENT_ATTEMPTS


CHUNKS_DIR = PROJECT_ROOT / "chunks"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "agent_evaluation"
    / "orchestrator"
)

OUTPUT_FILE = OUTPUT_DIR / "orchestrator_goldens.json"


# ============================================================
# CHUNK LOADER
# ============================================================

def load_chunks(
    max_chunks_per_file=3,
    max_total_chunks=20
):
    """
    Load actual chunk records from all JSON files
    inside the parent chunks directory.

    Supports:
        1. JSON list of chunks
        2. JSON object containing {"chunks": [...]}
    """

    if not CHUNKS_DIR.exists():
        raise FileNotFoundError(
            f"Chunks directory not found: {CHUNKS_DIR}"
        )

    json_files = sorted(CHUNKS_DIR.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {CHUNKS_DIR}"
        )

    all_chunks = []

    for json_file in json_files:

        print(f"\nReading: {json_file.name}")

        try:
            with open(
                json_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            if isinstance(data, dict):
                data = data.get("chunks", [])

            if not isinstance(data, list):
                print("Skipping: expected a list of chunks.")
                continue

            file_count = 0

            for chunk in data:

                if not isinstance(chunk, dict):
                    continue

                text = chunk.get("text", "")

                if not isinstance(text, str) or not text.strip():
                    continue

                # Preserve the actual chunk metadata.
                normalized_chunk = {
                    "chunk_id": chunk.get("chunk_id"),
                    "paper_title": chunk.get("paper_title", ""),
                    "section": chunk.get("section", ""),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "source_file": chunk.get("source_file", ""),
                    "text": text,
                    "char_count": chunk.get(
                        "char_count",
                        len(text)
                    ),
                    "created_at": chunk.get("created_at"),
                }

                all_chunks.append(normalized_chunk)

                file_count += 1

                if (
                    max_chunks_per_file
                    and file_count >= max_chunks_per_file
                ):
                    break

                if (
                    max_total_chunks
                    and len(all_chunks) >= max_total_chunks
                ):
                    print("Maximum chunk limit reached.")
                    return all_chunks

        except (json.JSONDecodeError, OSError) as error:

            print(
                f"Error reading {json_file.name}: {error}"
            )

    return all_chunks


# ============================================================
# STATE BUILDER
# ============================================================

def build_state(
    chunk,
    query,
    retrieved=True,
    analysis="",
    critical_analysis="",
    cited_analysis="",
    citations=None,
    **extra
):
    """
    Construct an input state using fields from
    the actual ResearchState TypedDict.
    """

    state = {

        # USER INPUT
        "user_query": query,

        # RETRIEVAL
        "retrieved_chunks": (
            [chunk] if retrieved else []
        ),

        "retrieval_count": (
            1 if retrieved else 0
        ),

        "retrieval_attempts": 0,
        "retrieval_feedback": "",

        # AGENT ATTEMPTS
        "analyst_attempts": 0,
        "critic_attempts": 0,
        "max_critic_attempts": MAX_AGENT_ATTEMPTS,

        # REVIEW CONTROL
        "review_limit_reached": False,

        # AGENT OUTPUTS
        "analysis": analysis,
        "critical_analysis": critical_analysis,
        "analyst_feedback": "",

        "cited_analysis": cited_analysis,
        "citations": citations or [],

        # STATUS FIELDS
        "retrieval_status": "completed" if retrieved else "pending",
        "retrieval_error": None,

        "citation_status": (
            "completed" if cited_analysis else "pending"
        ),
        "citation_error": None,

        # FINAL OUTPUT
        "final_answer": "",

        # ORCHESTRATOR CONTROL
        "next_agent": "",
        "llm_selected_agent": None,
        "completed_agents": [],

        # ERROR HANDLING
        "error": None,
    }

    # Allow individual test scenarios to override fields.
    state.update(extra)

    return state


# ============================================================
# TEST CASE GENERATOR
# ============================================================

def generate_test_cases(chunk, chunk_index):

    paper = chunk.get(
        "paper_title",
        "Unknown Paper"
    )

    section = chunk.get(
        "section",
        "Unknown Section"
    )

    query = (
        f"Explain the main information presented in "
        f"the section '{section}' of the paper '{paper}'."
    )

    analysis = (
        f"The analysis explains the retrieved evidence "
        f"from the paper '{paper}'."
    )

    rejected_critique = """
VERDICT: REJECTED

The analysis contains unsupported claims.
Revise the answer using only the retrieved evidence.
"""

    approved_critique = """
VERDICT: APPROVED

The analysis is supported by the retrieved evidence.
"""

    retrieval_critique = """
VERDICT: NEED_MORE_RETRIEVAL

Additional evidence is required to answer the question.
"""

    invalid_critique = """
VERDICT: MAYBE

The critic response does not contain a valid verdict.
"""

    missing_verdict_critique = """
The analysis needs additional review.
No explicit verdict was provided.
"""

    cases = []

    def add_case(
        test_id,
        scenario,
        state,
        expected_llm_agent,
        expected_agent
    ):

        cases.append({

            "id": f"C{chunk_index:03d}_{test_id}",

            "scenario": (
                f"{scenario} | Paper: {paper}"
            ),

            "input": state,

            "expected_llm_agent": expected_llm_agent,

            "expected_agent": expected_agent,
        })

    # --------------------------------------------------------
    # T01: INITIAL RETRIEVAL
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        retrieved=False
    )

    add_case(
        "T01",
        "No retrieved documents",
        state,
        "retriever",
        "retriever"
    )

    # --------------------------------------------------------
    # T02: RETRIEVAL COMPLETED
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query
    )

    add_case(
        "T02",
        "Documents exist but analysis is missing",
        state,
        "analyst",
        "analyst"
    )

    # --------------------------------------------------------
    # T03: ANALYSIS COMPLETED
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis
    )

    add_case(
        "T03",
        "Analysis exists but critic evaluation is missing",
        state,
        "critic",
        "critic"
    )

    # --------------------------------------------------------
    # T04: CRITIC REJECTED
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critical_analysis=rejected_critique
    )

    add_case(
        "T04",
        "Critic rejected the analysis",
        state,
        "analyst",
        "analyst"
    )

    # --------------------------------------------------------
    # T05: CRITIC APPROVED, CITATION MISSING
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critical_analysis=approved_critique
    )

    add_case(
        "T05",
        "Critic approved but citation is missing",
        state,
        "citation",
        "citation"
    )

    # --------------------------------------------------------
    # T06: CITATION COMPLETED
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critical_analysis=approved_critique,
        cited_analysis=(
            f"Cited analysis based on the paper '{paper}'."
        ),
        citations=[
            {
                "source_id": "S1",
                "paper_title": paper,
                "section": section,
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
            }
        ]
    )

    add_case(
        "T06",
        "Critic approved and citation exists",
        state,
        "synthesizer",
        "synthesizer"
    )

    # --------------------------------------------------------
    # T07: MORE RETRIEVAL REQUIRED
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critical_analysis=retrieval_critique,
        cited_analysis="Old cited analysis",
        citations=[
            {"source_id": "S1"}
        ]
    )

    add_case(
        "T07",
        "Critic requests additional retrieval",
        state,
        "retriever",
        "retriever"
    )

    # --------------------------------------------------------
    # T08: INVALID VERDICT
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critical_analysis=invalid_critique
    )

    add_case(
        "T08",
        "Critic returns an invalid verdict",
        state,
        "critic",
        "critic"
    )

    # --------------------------------------------------------
    # T09: RETRIEVER ATTEMPT LIMIT
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        retrieved=False,
        retrieval_attempts=MAX_AGENT_ATTEMPTS
    )

    add_case(
        "T09",
        "Retriever attempt limit reached",
        state,
        "retriever",
        "citation"
    )

    # --------------------------------------------------------
    # T10: ANALYST ATTEMPT LIMIT
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analyst_attempts=MAX_AGENT_ATTEMPTS
    )

    add_case(
        "T10",
        "Analyst attempt limit reached",
        state,
        "analyst",
        "citation"
    )

    # --------------------------------------------------------
    # T11: CRITIC ATTEMPT LIMIT
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critical_analysis=rejected_critique,
        critic_attempts=MAX_AGENT_ATTEMPTS,
        max_critic_attempts=MAX_AGENT_ATTEMPTS
    )

    add_case(
        "T11",
        "Critic attempt limit reached",
        state,
        "analyst",
        "citation"
    )

    # --------------------------------------------------------
    # T12: REVIEW LIMIT ALREADY REACHED
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        review_limit_reached=True
    )

    add_case(
        "T12",
        "Review limit already reached",
        state,
        None,
        "synthesizer"
    )

    # --------------------------------------------------------
    # T13: CRITIQUE ALIAS
    #
    # Your orchestrator supports:
    # state.get("critical_analysis") or state.get("critique")
    #
    # "critique" is intentionally included as a runtime
    # compatibility field, although it is not declared
    # in ResearchState.
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critique=rejected_critique
    )

    add_case(
        "T13",
        "Critique alias field is used",
        state,
        "analyst",
        "analyst"
    )

    # --------------------------------------------------------
    # T14: REJECTION FEEDBACK
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critical_analysis=rejected_critique
    )

    add_case(
        "T14",
        "Critic feedback is passed to analyst",
        state,
        "analyst",
        "analyst"
    )

    # --------------------------------------------------------
    # T15: RETRIEVAL FEEDBACK AND STATE CLEANUP
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critical_analysis=retrieval_critique,
        cited_analysis="Old cited analysis",
        citations=[
            {"source_id": "S1"}
        ]
    )

    add_case(
        "T15",
        "Retrieval feedback is passed and old state cleared",
        state,
        "retriever",
        "retriever"
    )

    # --------------------------------------------------------
    # T16: MISSING VERDICT
    # --------------------------------------------------------

    state = build_state(
        chunk,
        query,
        analysis=analysis,
        critical_analysis=missing_verdict_critique
    )

    add_case(
        "T16",
        "Critic response exists but verdict is missing",
        state,
        "critic",
        "critic"
    )

    return cases


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(cases):

    valid_agents = {
        "retriever",
        "analyst",
        "critic",
        "citation",
        "synthesizer",
    }

    for case in cases:

        if not isinstance(case.get("input"), dict):
            raise ValueError(
                f"Invalid input state: {case['id']}"
            )

        expected_llm = case["expected_llm_agent"]
        expected_final = case["expected_agent"]

        if (
            expected_llm is not None
            and expected_llm not in valid_agents
        ):
            raise ValueError(
                f"Invalid expected LLM agent: {case['id']}"
            )

        if expected_final not in valid_agents:
            raise ValueError(
                f"Invalid expected final agent: {case['id']}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate orchestrator golden datasets "
            "from real research paper chunks."
        )
    )

    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=3
    )

    parser.add_argument(
        "--max-total-chunks",
        type=int,
        default=20
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("ORCHESTRATOR GOLDEN DATASET GENERATOR")
    print("=" * 65)

    print("Project:", PROJECT_ROOT)
    print("Chunks directory:", CHUNKS_DIR)
    print("MAX_AGENT_ATTEMPTS:", MAX_AGENT_ATTEMPTS)

    # 1. Load chunks
    chunks = load_chunks(
        max_chunks_per_file=args.max_chunks_per_file,
        max_total_chunks=args.max_total_chunks
    )

    if not chunks:
        raise ValueError("No valid chunks found.")

    print(f"\nTotal chunks loaded: {len(chunks)}")

    # 2. Generate test cases
    all_cases = []

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"\nGenerating test cases for chunk {index}"
        )

        print("Paper:", chunk.get("paper_title"))
        print("Section:", chunk.get("section"))

        cases = generate_test_cases(
            chunk,
            index
        )

        all_cases.extend(cases)

    # 3. Validate
    validate_dataset(all_cases)

    # 4. Save
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_cases,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("DATASET GENERATION COMPLETED")
    print("=" * 65)

    print("Chunks processed:", len(chunks))
    print("Scenarios per chunk: 16")
    print("Total test cases:", len(all_cases))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
