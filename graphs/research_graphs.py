from pathlib import Path

from langgraph.graph import StateGraph, START, END

from state.state import ResearchState

from agents.orchestrator_agent import orchestrator_agent
from agents.retriever_agent import retriever_agent
from agents.analyst_agent import analyst_agent
from agents.critic_agent import critic_agent
from agents.synthesizer_agent import synthesizer_agent


# --------------------------------------------------
# Create Graph
# --------------------------------------------------

workflow = StateGraph(ResearchState)


# --------------------------------------------------
# Add Agents
# --------------------------------------------------

workflow.add_node(
    "orchestrator",
    orchestrator_agent
)

workflow.add_node(
    "retriever",
    retriever_agent
)

workflow.add_node(
    "analyst",
    analyst_agent
)

workflow.add_node(
    "critic",
    critic_agent
)

workflow.add_node(
    "synthesizer",
    synthesizer_agent
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