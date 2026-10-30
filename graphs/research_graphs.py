from pathlib import Path

from langgraph.graph import StateGraph, START, END

from state.state import ResearchState

from agents.orchestrator_agent import orchestrator_agent
from agents.retriever_agent import retriever_agent
from agents.analyst_agent import analyst_agent
from agents.critic_agent import critic_agent
from agents.synthesizer_agent import synthesizer_agent
from agents.citation_agent import citation_agent

from langgraph.config import get_stream_writer



def track_agent(agent_name, agent_function):

    def tracked_node(state):

        writer = get_stream_writer()

        # Notify Gradio that this agent has started.
        writer({
            "agent": agent_name,
            "status": "running"
        })

        try:
            result = agent_function(state)

            # Notify Gradio that this agent has completed.
            writer({
                "agent": agent_name,
                "status": "completed"
            })

            return result

        except Exception as error:

            writer({
                "agent": agent_name,
                "status": "failed",
                "error": str(error)
            })

            raise

    return tracked_node





# --------------------------------------------------
# Create Graph
# --------------------------------------------------

workflow = StateGraph(ResearchState)


# --------------------------------------------------
# Add Agents
# --------------------------------------------------

workflow.add_node(
    "orchestrator",
    track_agent("Orchestrator", orchestrator_agent)
)

workflow.add_node(
    "retriever",
    track_agent("Retriever", retriever_agent)
)

workflow.add_node(
    "analyst",
    track_agent("Research Analyst", analyst_agent)
)

workflow.add_node(
    "critic",
    track_agent("Critical Analyst", critic_agent)
)

workflow.add_node(
    "citation",
    track_agent("Citation Agent", citation_agent)
)

workflow.add_node(
    "synthesizer",
    track_agent("Synthesis Writer", synthesizer_agent)
)


# --------------------------------------------------
# Start
# --------------------------------------------------

workflow.add_edge(
    START,
    "orchestrator"
)


# --------------------------------------------------
# Orchestrator Routing
# --------------------------------------------------

def route_agent(state: ResearchState):

    return state["next_agent"]


workflow.add_conditional_edges(
    "orchestrator",
    route_agent,
    {
        "retriever": "retriever",
        "analyst": "analyst",
        "critic": "critic",
        "citation":"citation",
        "synthesizer": "synthesizer",
    }
)


# --------------------------------------------------
# Agent → Orchestrator
# --------------------------------------------------

workflow.add_edge(
    "retriever",
    "orchestrator"
)

workflow.add_edge(
    "analyst",
    "orchestrator"
)

workflow.add_edge(
    "critic",
    "orchestrator"
)

workflow.add_edge("citation", "orchestrator")


# --------------------------------------------------
# Synthesizer → END
# --------------------------------------------------

workflow.add_edge(
    "synthesizer",
    END
)


# --------------------------------------------------
# Compile
# --------------------------------------------------

research_graph = workflow.compile()


output_dir = Path("graphs_output")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "research_graph.png"

png_data = research_graph.get_graph().draw_mermaid_png()

with open(output_file, "wb") as f:
    f.write(png_data)

print(f"research_graph saved successfully to: {output_file.resolve()}")