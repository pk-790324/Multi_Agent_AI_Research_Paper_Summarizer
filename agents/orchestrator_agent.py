# rom langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from state.state import ResearchState
# from llm.llm import llm


# from typing import Literal

# from pydantic import BaseModel, Field
# from langchain_core.prompts import ChatPromptTemplate


# class RoutingDecision(BaseModel):
#     next_agent: Literal[
#         "retriever",
#         "analyst",
#         "critic",
#         "synthesizer",
#     ] = Field(
#         description="The next agent that should execute."
#     )


# orchestrator_prompt = ChatPromptTemplate.from_template("""
# You are the orchestrator of a multi-agent
# research paper analysis system.

# Your job is ONLY to decide which agent should execute next.

# User question:
# {user_query}

# Retrieved documents:
# {retrieved_docs}

# Analysis:
# {analysis}

# Critique:
# {critique}

# Retrieval count:
# {retrieval_count}

# Rules:

# 1. If there are no retrieved documents:
#    choose retriever.

# 2. If documents exist but there is no analysis:
#    choose analyst.

# 3. If analysis exists but there is no critique:
#    choose critic.

# 4. If the critic says NEED_MORE_RETRIEVAL:
#    choose retriever.

# 5. If the critic says APPROVED:
#    choose synthesizer.

# 6. If retrieval_count >= 3:
#    choose synthesizer.

# Return ONLY the structured routing decision.
# """)


# def orchestrator_agent(state:ResearchState):
#     print("\n" + "=" * 60)
#     print("ORCHESTRATOR AGENT")
#     print("=" * 60)
#     print("\n" + "=" * 60)
#     print("ORCHESTRATOR RECEIVED QUERY:")
#     print(">>>", state.get("user_query"))

#     documents = state.get("retrieved_docs", [])
#     analysis = state.get("analysis", "")
#     critique = state.get("critique", "")

#     retrieval_count = state.get(
#         "retrieval_count",
#         0
#     )

#     # Hard ceiling: avoid infinite retrieval loops.
#     # If we have retried 3+ times and still have no documents,
#     # hand off to the synthesizer which will give a graceful response.
#     if retrieval_count >= 3:
#         print(f"[ORCHESTRATOR] Max retrieval attempts reached ({retrieval_count}). Forcing synthesizer.")
#         return {
#             "next_agent": "synthesizer"
#         }

#     if not documents:
#         return {
#             "next_agent": "retriever"
#         }

#     if not analysis:
#         return {
#             "next_agent": "analyst"
#         }

#     if not critique:
#         return {
#             "next_agent": "critic"
#         }

#     if "NEED_MORE_RETRIEVAL" in critique:
#         return {
#             "next_agent": "retriever"
#         }

#     return {
#         "next_agent": "synthesizer"
#     }



from typing import Literal

from pydantic import BaseModel, Field


from state.state import ResearchState


class RoutingDecision(BaseModel):
    next_agent: Literal[
        "retriever",
        "analyst",
        "critic",
        "synthesizer",
    ] = Field(
        description="The next agent that should execute."
    )


def orchestrator_agent(state: ResearchState):

    # ========================================================
    # ORCHESTRATOR DEBUG
    # ========================================================

    print("\n" + "=" * 60)
    print("ORCHESTRATOR AGENT")
    print("=" * 60)

    user_query = state.get("user_query")

    print("ORCHESTRATOR RECEIVED QUERY:")
    print(">>>", repr(user_query))

    print("State keys:")
    print(">>>", list(state.keys()))

    # ========================================================
    # READ STATE
    # ========================================================

    documents = state.get(
        "retrieved_docs",
        []
    )

    analysis = state.get(
        "analysis",
        ""
    )

    critique = state.get(
        "critique",
        ""
    )

    retrieval_count = state.get(
        "retrieval_count",
        0
    )

    # ========================================================
    # DEBUG STATE
    # ========================================================

    print(
        "Retrieved documents:",
        len(documents)
    )

    print(
        "Analysis exists:",
        bool(analysis)
    )

    print(
        "Critique exists:",
        bool(critique)
    )

    print(
        "Retrieval count:",
        retrieval_count
    )

    # ========================================================
    # ROUTING
    # ========================================================

    # --------------------------------------------------------
    # 1. No documents -> Retriever
    # --------------------------------------------------------

    if not documents:

        print(
            "[ORCHESTRATOR] "
            "Decision: retriever"
        )

        return {
            "next_agent": "retriever"
        }

    # --------------------------------------------------------
    # 2. Documents exist but no analysis -> Analyst
    # --------------------------------------------------------

    if not analysis:

        print(
            "[ORCHESTRATOR] "
            "Decision: analyst"
        )

        return {
            "next_agent": "analyst"
        }

    # --------------------------------------------------------
    # 3. Analysis exists but no critique -> Critic
    # --------------------------------------------------------

    if not critique:

        print(
            "[ORCHESTRATOR] "
            "Decision: critic"
        )

        return {
            "next_agent": "critic"
        }

    # --------------------------------------------------------
    # 4. Critic requests more retrieval
    # --------------------------------------------------------

    if "NEED_MORE_RETRIEVAL" in critique:

        # Prevent infinite retrieval loop

        if retrieval_count >= 3:

            print(
                "[ORCHESTRATOR] "
                "Maximum retrieval attempts reached."
            )

            print(
                "[ORCHESTRATOR] "
                "Decision: synthesizer"
            )

            return {
                "next_agent": "synthesizer"
            }

        print(
            "[ORCHESTRATOR] "
            "Critic requested more retrieval."
        )

        print(
            "[ORCHESTRATOR] "
            "Decision: retriever"
        )

        return {
            "next_agent": "retriever"
        }

    # --------------------------------------------------------
    # 5. Critique exists and does not request retrieval
    # --------------------------------------------------------

    print(
        "[ORCHESTRATOR] "
        "Decision: synthesizer"
    )

    return {
        "next_agent": "synthesizer"
    }