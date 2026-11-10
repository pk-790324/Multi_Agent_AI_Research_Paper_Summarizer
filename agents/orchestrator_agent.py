
import re
import json

from langchain_core.messages import SystemMessage, HumanMessage

from state.state import ResearchState
from llm.llm import llm
from config import MAX_AGENT_ATTEMPTS

VALID_AGENTS = {
    "retriever",
    "analyst",
    "critic",
    "citation",
    "synthesizer",
}




def parse_routing_response(response):

    # Extract response text
    content = response.content if hasattr(response, "content") else str(response)

    if isinstance(content, list):
        content = " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    content = str(content).strip()

    # Try JSON response first
    try:
        data = json.loads(content)

        if isinstance(data, dict):
            agent = str(data.get("next_agent", "")).lower().strip()

            if agent in VALID_AGENTS:
                return agent

    except (json.JSONDecodeError, TypeError):
        pass

    # Accept plain-text responses:
    # retriever
    # next_agent: retriever
    # "retriever"
    first_line = content.splitlines()[0].strip().lower()

    first_line = first_line.replace("`", "").replace('"', "").replace("'", "")

    match = re.fullmatch(
        r"(?:next_agent\s*[:=]\s*)?"
        r"(retriever|analyst|critic|citation|synthesizer)[.!]?",
        first_line,
    )

    if match:
        return match.group(1)

    return None


def fallback_routing(state):

    documents = state.get("retrieved_chunks", [])
    analysis = state.get("analysis", "")
    critique = state.get("critical_analysis", "")
    cited_analysis = state.get("cited_analysis", "")

    verdict_match = re.search(
        r"^\s*VERDICT\s*:\s*"
        r"(APPROVED|REJECTED|NEED_MORE_RETRIEVAL)\b",
        critique or "",
        re.IGNORECASE | re.MULTILINE,
    )

    verdict = verdict_match.group(1).upper() if verdict_match else None

    if verdict == "NEED_MORE_RETRIEVAL":
        return "retriever"

    if verdict == "REJECTED":
        return "analyst"

    if verdict == "APPROVED":
        return "citation" if not cited_analysis else "synthesizer"

    if not documents:
        return "retriever"

    if not analysis:
        return "analyst"

    if not critique:
        return "critic"

    return "critic"


def orchestrator_agent(state: ResearchState):

    print("\n" + "=" * 60)
    print("ORCHESTRATOR AGENT")
    print("=" * 60)

    query = state.get("user_query", "")
    documents = state.get("retrieved_chunks", [])
    analysis = state.get("analysis", "")

    critique = (
        state.get("critical_analysis")
        or state.get("critique", "")
    )

    cited_analysis = state.get("cited_analysis", "")

    verdict_match = re.search(
        r"^\s*VERDICT\s*:\s*"
        r"(APPROVED|REJECTED|NEED_MORE_RETRIEVAL)\b",
        critique or "",
        re.IGNORECASE | re.MULTILINE,
    )

    verdict = verdict_match.group(1).upper() if verdict_match else "NOT_AVAILABLE"

    print("Research question:", repr(query))
    print("Retrieved chunks:", len(documents))
    print("Analysis exists:", bool(analysis))
    print("Critique exists:", bool(critique))
    print("Critic verdict:", verdict)
    print("Cited analysis exists:", bool(cited_analysis))

    if state.get("review_limit_reached", False):
        return {
            "next_agent": "synthesizer",
            "llm_selected_agent": None,
            "routing_reason": (
                "Attempt limit reached. Proceeding to final synthesis."
            ),
        }

    # -------------------------------------------------
    # LLM-BASED ROUTING
    # -------------------------------------------------

    system_prompt = """
You are the Orchestrator Agent of a multi-agent
research paper summarization system.

Choose exactly ONE next agent.

AVAILABLE AGENTS:
- retriever
- analyst
- critic
- citation
- synthesizer

ROUTING RULES:

1. If there are no retrieved documents, choose retriever.

2. If documents exist but analysis is missing, choose analyst.

3. If analysis exists but critic evaluation is missing,
   choose critic.

4. If critic verdict is NEED_MORE_RETRIEVAL,
   choose retriever.

5. If critic verdict is REJECTED,
   choose analyst.

6. If critic verdict is APPROVED and cited analysis
   is missing, choose citation.

7. If critic verdict is APPROVED and cited analysis
   exists, choose synthesizer.

8. If a critic response exists but its verdict is
   invalid or missing, choose critic.

IMPORTANT:
Return ONLY ONE agent name.
Do not return JSON.
Do not provide explanations.
Do not use markdown.
Valid responses are:
retriever
analyst
critic
citation
synthesizer
"""

    human_prompt = f"""
USER QUERY:
{query}

RETRIEVED DOCUMENT COUNT:
{len(documents)}

ANALYSIS AVAILABLE:
{bool(analysis)}

CRITIC VERDICT:
{verdict}

CRITIC FEEDBACK:
{critique}

CITED ANALYSIS AVAILABLE:
{bool(cited_analysis)}

Which agent should execute next?
"""
    llm_selected_agent = None
    try:

        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )

        next_agent = parse_routing_response(response)
        llm_selected_agent = next_agent
        if next_agent is None:
            raise ValueError(
                f"Could not parse routing response: {response.content}"
            )

        routing_reason = "LLM selected the next agent."

    except Exception as e:

        print("[ORCHESTRATOR] LLM routing failed:", str(e))

        next_agent = fallback_routing(state)
        routing_reason = "Deterministic fallback routing."

    # -------------------------------------------------
    # Enforce critic-based routing
    # -------------------------------------------------

    if verdict == "REJECTED":
        next_agent = "analyst"
        routing_reason = "Critic rejected the analysis. Sending feedback to analyst."

    elif verdict == "NEED_MORE_RETRIEVAL":
        next_agent = "retriever"
        routing_reason = "Critic requires additional evidence."

    elif verdict == "APPROVED":
        next_agent = (
            "citation" if not cited_analysis
            else "synthesizer"
        )
        routing_reason = "Critic approved the analysis."

    retrieval_attempts = state.get("retrieval_attempts", 0)
    analyst_attempts = state.get("analyst_attempts", 0)
    review_limit_reached = False
    critic_attempts = state.get("critic_attempts", 0)
    max_critic_attempts = state.get(
        "max_critic_attempts",
        MAX_AGENT_ATTEMPTS,
    )

    if verdict != "APPROVED" and critic_attempts >= max_critic_attempts:
        next_agent = "citation"
        review_limit_reached = True
        routing_reason = (
            "Critic attempt limit reached. Proceeding to citation."
        )

    elif next_agent == "retriever":
        if retrieval_attempts >= MAX_AGENT_ATTEMPTS:
            next_agent = "citation"
            review_limit_reached = True
            routing_reason = (
                "Retriever attempt limit reached. Proceeding to citation."
            )
        else:
            retrieval_attempts += 1

    elif next_agent == "analyst":
        if analyst_attempts >= MAX_AGENT_ATTEMPTS:
            next_agent = "citation"
            review_limit_reached = True
            routing_reason = (
                "Analyst attempt limit reached. Proceeding to citation."
            )
        else:
            analyst_attempts += 1

    print("\n[ORCHESTRATOR] Decision:", next_agent)
    print("[ORCHESTRATOR] Reason:", routing_reason)

    result = {
        "next_agent": next_agent,
        "llm_selected_agent": llm_selected_agent,
        "routing_reason": routing_reason,
        "retrieval_attempts": retrieval_attempts,
        "analyst_attempts": analyst_attempts,
        "critic_attempts": critic_attempts,
        "review_limit_reached": review_limit_reached,
    }

    # Pass critic feedback to the appropriate agent
    if verdict == "NEED_MORE_RETRIEVAL" and next_agent == "retriever":

        result.update({
            "retrieval_feedback": critique,
            "analysis": "",
            "critical_analysis": "",
            "cited_analysis": "",
            "citations": [],
        })

    elif verdict == "REJECTED" and next_agent == "analyst":

        result.update({
            "analyst_feedback": critique,
            "critical_analysis": "",
            "cited_analysis": "",
            "citations": [],
        })

    return result
