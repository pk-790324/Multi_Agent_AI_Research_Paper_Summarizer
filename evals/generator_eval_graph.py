
"""
evals/generator_eval_graph.py

Production orchestration with golden-context injection.
Actual Qdrant retrieval is bypassed.
"""

from langgraph.graph import StateGraph, START, END

from state.state import ResearchState

from agents.orchestrator_agent import orchestrator_agent
from agents.analyst_agent import analyst_agent
from agents.critic_agent import critic_agent
from agents.synthesizer_agent import synthesizer_agent
from agents.citation_agent import citation_agent


# --------------------------------------------------
# GOLDEN CONTEXT INJECTOR
# --------------------------------------------------

def golden_context_node(state: ResearchState):
    """
    Replaces the actual retriever.

    The golden context is already present in
    state['retrieved_chunks'].

    No Qdrant search or embedding generation happens.
    """

    chunks = state.get("retrieved_chunks", [])

    print("\nGOLDEN CONTEXT INJECTOR")
    print(f"Injected chunks: {len(chunks)}")

    return {
        "retrieved_chunks": chunks,
        "retrieval_count": len(chunks),
    }


# ---------------------------------------------
# ORCHESTRATOR ROUTING
# ---------------------------------------------

def route_agent(state: ResearchState):

    return state["next_agent"]


# ---------------------------------------------
# BUILD GENERATOR EVALUATION GRAPH
# ---------------------------------------------

workflow = StateGraph(ResearchState)


# Reuse actual production agents
workflow.add_node("orchestrator", orchestrator_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("citation", citation_agent)
workflow.add_node("synthesizer", synthesizer_agent)


# START
workflow.add_edge(START, "orchestrator")


# ORCHESTRATOR ROUTING
workflow.add_conditional_edges(
    "orchestrator",
    route_agent,
    {
        "analyst": "analyst",
        "critic": "critic",
        "citation": "citation",
        "synthesizer": "synthesizer",
    }
)


# AGENT -> ORCHESTRATOR
workflow.add_edge("analyst", "orchestrator")
workflow.add_edge("critic", "orchestrator")
workflow.add_edge("citation", "orchestrator")


# FINAL OUTPUT
workflow.add_edge("synthesizer", END)


# COMPILE
generator_eval_graph = workflow.compile()