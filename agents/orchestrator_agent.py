from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state.state import ResearchState
from llm.llm import llm


from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class RoutingDecision(BaseModel):
    next_agent: Literal[
        "retriever",
        "analyst",
        "critic",
        "synthesizer",
    ] = Field(
        description="The next agent that should execute."
    )


orchestrator_prompt = ChatPromptTemplate.from_template("""
You are the orchestrator of a multi-agent
research paper analysis system.

Your job is ONLY to decide which agent should execute next.

User question:
{user_query}

Retrieved documents:
{retrieved_docs}

Analysis:
{analysis}

Critique:
{critique}

Retrieval count:
{retrieval_count}

Rules:

1. If there are no retrieved documents:
   choose retriever.

2. If documents exist but there is no analysis:
   choose analyst.

3. If analysis exists but there is no critique:
   choose critic.

4. If the critic says NEED_MORE_RETRIEVAL:
   choose retriever.

5. If the critic says APPROVED:
   choose synthesizer.

6. If retrieval_count >= 3:
   choose synthesizer.

Return ONLY the structured routing decision.
""")


def orchestrator_agent(state):
    print("\n" + "=" * 60)
    print("ORCHESTRATOR AGENT")
    print("=" * 60)

    documents = state.get("retrieved_docs", [])
    analysis = state.get("analysis", "")
    critique = state.get("critique", "")

    retrieval_count = state.get(
        "retrieval_count",
        0
    )

    if not documents:
        return {
            "next_agent": "retriever"
        }

    if not analysis:
        return {
            "next_agent": "analyst"
        }

    if not critique:
        return {
            "next_agent": "critic"
        }

    if "NEED_MORE_RETRIEVAL" in critique:

        if retrieval_count < 3:
            return {
                "next_agent": "retriever"
            }

    return {
        "next_agent": "synthesizer"
    }