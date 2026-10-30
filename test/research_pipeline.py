
from graphs.research_graphs import research_graph


def run_research_pipeline():

    print("\n" + "=" * 70)
    print("MULTI-AGENT RESEARCH PAPER SUMMARIZER")
    print("=" * 70)

    # --------------------------------------------------
    # 1. USER QUESTION
    # --------------------------------------------------

    user_query = input(
        "\nEnter your research question: "
    ).strip()

    if not user_query:
        print("Please enter a valid question.")
        return

    # --------------------------------------------------
    # 2. INITIAL GRAPH STATE
    # --------------------------------------------------

    initial_state = {
    "user_query": user_query,
    "retrieved_chunks": [],
    "retrieval_count": 0,
    "analysis": "",
    "critical_analysis": "",
    "cited_analysis": "",
    "citations": [],
    "final_answer": "",
    "next_agent": "retriever",
    "completed_agents": [],
    "error": None,
}

    # --------------------------------------------------
    # 3. EXECUTE GRAPH
    # --------------------------------------------------

    print("\nStarting research pipeline...\n")

    try:

        result = research_graph.invoke(
            initial_state,
            config={
                "recursion_limit": 30
            }
        )

        # --------------------------------------------------
        # 4. DISPLAY FINAL ANSWER
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("FINAL RESEARCH ANSWER")
        print("=" * 70)

        print(
            result.get(
                "final_answer",
                "No final answer generated."
            )
        )

        # --------------------------------------------------
        # 5. DISPLAY CITATIONS
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("REFERENCES")
        print("=" * 70)

        for citation in result.get("citations", []):

            print(
                f"\n[{citation.get('source_id')}] "
                f"{citation.get('paper_title')}"
            )

            print(
                f"Section: {citation.get('section')}"
            )

            print(
                f"Page: {citation.get('page_start')} - "
                f"{citation.get('page_end')}"
            )

            print(
                f"Chunk ID: {citation.get('chunk_id')}"
            )

        print("\nPipeline completed successfully!")

    except Exception as e:

        print("\nPipeline execution failed!")
        print(f"Error: {e}")


if __name__ == "__main__":
    run_research_pipeline()
