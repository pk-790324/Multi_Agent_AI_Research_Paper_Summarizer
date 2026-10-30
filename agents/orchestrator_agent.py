
from typing import Literal
from pydantic import BaseModel, Field

from state.state import ResearchState


class RoutingDecision(BaseModel):
    next_agent: Literal[
        "retriever",
        "analyst",
        "critic",
        "citation",
        "synthesizer",
    ] = Field(
        description="The next agent that should execute."
    )


def orchestrator_agent(state: ResearchState):

    print("\n" + "=" * 60)
    print("ORCHESTRATOR AGENT")
    print("=" * 60)

    # Read the research question
    query = state.get("user_query") or state.get("user_query")

    print("Research question:", repr(query))
    print("State keys:", list(state.keys()))

    # Read the outputs from previous agents
    documents = state.get("retrieved_chunks", [])
    analysis = state.get("analysis", "")

    # Support either naming convention for the critic output
    critique = (
        state.get("critical_analysis")
        or state.get("critique", "")
    )

    cited_analysis = state.get("cited_analysis", "")

    print("Retrieved chunks:", len(documents))
    print("Analysis exists:", bool(analysis))
    print("Critique exists:", bool(critique))
    print("Cited analysis exists:", bool(cited_analysis))

    # 1. Retrieve documents
    if not documents:
        next_agent = "retriever"

    # 2. Analyze retrieved documents
    elif not analysis:
        next_agent = "analyst"

    # 3. Critically evaluate the analysis
    elif not critique:
        next_agent = "critic"

    # 4. Add citations to the analysis
    elif not cited_analysis:
        next_agent = "citation"

    # 5. Generate the final response
    else:
        next_agent = "synthesizer"

    print(f"[ORCHESTRATOR] Decision: {next_agent}")

    return {
        "next_agent": next_agent
    }
