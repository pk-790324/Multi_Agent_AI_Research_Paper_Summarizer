
import json
from datetime import datetime

# Import your existing graph and configuration
from graphs.research_graphs import research_graph
from config import MAX_AGENT_ATTEMPTS,AGENT_NAMES


def format_retrieved_chunks(chunks):

    if not chunks:
        return "### 📚 No research chunks retrieved."

    markdown = "## 📚 Retrieved Research Chunks\n\n"
    markdown += f"**Total Chunks Retrieved:** {len(chunks)}\n\n"
    markdown += "---\n\n"

    for chunk in chunks:

        rank = chunk.get("rank", "N/A")
        score = chunk.get("score", 0)
        paper_title = chunk.get("paper_title", "Unknown Paper")
        section = chunk.get("section", "Unknown Section")
        page_start = chunk.get("page_start", "N/A")
        page_end = chunk.get("page_end", "N/A")
        content = chunk.get("page_content", "")
        source_file = chunk.get("source_file", "Unknown Source")
        chunk_id = chunk.get("chunk_id", "N/A")

        markdown += f"### 🔹 Rank {rank}\n\n"
        markdown += f"**Paper:** {paper_title}\n\n"
        markdown += f"**Section:** {section}\n\n"
        markdown += f"**Relevance Score:** `{score:.4f}`\n\n"
        markdown += f"**Pages:** {page_start} - {page_end}\n\n"
        markdown += f"**Chunk ID:** `{chunk_id}`\n\n"
        markdown += f"**Source File:** `{source_file}`\n\n"
        markdown += "**Content:**\n\n"
        markdown += f"> {content}\n\n"
        markdown += "---\n\n"

    return markdown


def run_research_pipeline(user_query: str,user_id: str,):

    agent_counts = {
        name: 0 for name in AGENT_NAMES
    }

    agent_status = {
        name: "Waiting" for name in AGENT_NAMES
    }

    execution_logs = []

    final_state = {
        "user_id": user_id,
        "user_query": user_query.strip(),
        "retrieved_chunks": [],
        "retrieval_count": 0,
        "retrieval_attempts": 0,
        "analyst_attempts": 0,
        "critic_attempts": 0,
        "max_critic_attempts": MAX_AGENT_ATTEMPTS,
        "analysis": "",
        "critical_analysis": "",
        "cited_analysis": "",
        "citations": [],
        "final_answer": "",
        "next_agent": "retriever",
        "completed_agents": [],
        "error": None,
    }

    def get_dashboard():

        rows = []

        for index, name in enumerate(AGENT_NAMES, start=1):

            rows.append([
                index,
                name,
                agent_status[name],
                agent_counts[name]
            ])

        completed = sum(
            status == "Completed"
            for status in agent_status.values()
        )

        failed = sum(
            status == "Failed"
            for status in agent_status.values()
        )

        progress = (
            f"Pipeline Progress: "
            f"{completed}/{len(AGENT_NAMES)} agents completed"
        )

        if failed:
            progress += f" | {failed} failed"

        return rows, progress

    def make_event(message):

        rows, progress = get_dashboard()

        return {
            "message": message,
            "agent_dashboard": rows,
            "pipeline_progress": progress,
            "execution_logs": execution_logs.copy(),
            "final_answer": final_state.get("final_answer", ""),
            "retrieved_chunks": final_state.get("retrieved_chunks", []),
            "retrieved_chunks_markdown": format_retrieved_chunks(
                final_state.get("retrieved_chunks", [])
            ),
            "analysis": final_state.get("analysis", ""),
            "critical_analysis": final_state.get("critical_analysis", ""),
            "cited_analysis": final_state.get("cited_analysis", ""),
            "citations": final_state.get("citations", []),
            "retrieval_count": final_state.get("retrieval_count", 0),
            "retrieval_attempts": final_state.get("retrieval_attempts", 0),
            "analyst_attempts": final_state.get("analyst_attempts", 0),
            "critic_attempts": final_state.get("critic_attempts", 0),
            "completed_agents": final_state.get("completed_agents", []),
            "error": final_state.get("error"),
        }

    if not user_query or not user_query.strip():

        yield make_event("Please enter a research question.")
        return

    try:

        yield make_event("Initializing research pipeline...")

        for mode, event in research_graph.stream(
            final_state,
            config={"recursion_limit": 30},
            stream_mode=["custom", "updates"]
        ):

            if mode == "custom":

                agent = event.get("agent")
                status = event.get("status")

                if agent not in agent_status:
                    continue

                timestamp = datetime.now().strftime("%H:%M:%S")

                if status == "running":

                    agent_counts[agent] += 1
                    agent_status[agent] = "Running"

                    execution_logs.append(
                        f"🟡 `{timestamp}` **{agent}** started "
                        f"(Execution #{agent_counts[agent]})"
                    )

                elif status == "completed":

                    agent_status[agent] = "Completed"

                    execution_logs.append(
                        f"🟢 `{timestamp}` **{agent}** completed"
                    )

                elif status == "failed":

                    agent_status[agent] = "Failed"

                    execution_logs.append(
                        f"🔴 `{timestamp}` **{agent}** failed: "
                        f"{event.get('error', 'Unknown error')}"
                    )

                yield make_event("Research pipeline is running...")

            elif mode == "updates":
                print("\n========== LANGGRAPH NODE UPDATE ==========")
                print(event)

                for node_name, node_update in event.items():

                    print(f"\nNode executed: {node_name}")
                    print("Node returned:", node_update)

                    if isinstance(node_update, dict):

                        final_state.update(node_update)

                        print("Updated state keys:", list(final_state.keys()))

                print("\n========== CURRENT OUTPUTS ==========")
                print("Analysis:", final_state.get("analysis"))
                print("Critical Analysis:", final_state.get("critical_analysis"))
                print("Cited Analysis:", final_state.get("cited_analysis"))
                print("Final Answer:", final_state.get("final_answer"))
                print("Retrieved chunks:", len(final_state.get("retrieved_chunks", [])))

                yield make_event("Research graph state updated.")

        # Final pipeline result

        if final_state.get("error"):

            yield make_event(
                f"Research pipeline failed: {final_state['error']}"
            )

        else:
            yield make_event(
                "Research pipeline completed successfully!"
            )

    except Exception as error:

        timestamp = datetime.now().strftime("%H:%M:%S")

        execution_logs.append(
            f"🔴 `{timestamp}` **Pipeline Error:** {str(error)}"
        )

        for agent in AGENT_NAMES:

            if agent_status[agent] == "Running":
                agent_status[agent] = "Failed"

        final_state["error"] = (
            f"{type(error).__name__}: {error}"
        )

        yield make_event(
            f"Research pipeline failed: {final_state['error']}"
        )
