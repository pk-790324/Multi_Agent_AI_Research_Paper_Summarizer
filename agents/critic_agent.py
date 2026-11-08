
from langchain_core.prompts import ChatPromptTemplate

from state.state import ResearchState
from llm.llm import llm


# ============================================================
# CRITIC PROMPT
# ============================================================


critic_prompt = ChatPromptTemplate.from_template("""
You are a critical reviewer in a research paper analysis system.

Your responsibility is to evaluate the Analyst's response
against the retrieved research evidence.

USER QUESTION:
{user_query}

RETRIEVED EVIDENCE:
{context}

ANALYST'S ANALYSIS:
{analysis}

PREVIOUS ANALYST FEEDBACK:
{previous_feedback}

EVALUATION INSTRUCTIONS:

1. Check whether the analysis answers the user's question.
2. Verify whether important claims are supported by evidence.
3. Identify unsupported or misleading claims.
4. Identify missing important information.
5. Check technical and methodological correctness.
6. Do not introduce information from your own knowledge.
7. Do not request additional retrieval if existing evidence
   is sufficient to correct the analysis.
8. Be specific. Explain exactly what the analyst must fix.

DECISION RULES:

A. APPROVED:
Use when the analysis correctly answers the question
and all important claims are supported by evidence.

B. REJECTED:
Use when the retrieved evidence is sufficient, but the
analyst's response contains errors, unsupported claims,
incorrect interpretations, or missing explanations that
can be corrected using the existing evidence.

C. NEED_MORE_RETRIEVAL:
Use when the retrieved evidence genuinely does not contain
enough information to answer the question correctly.

STRICT OUTPUT FORMAT:

VERDICT: APPROVED
REASON: Explain why the analysis is acceptable.

OR

VERDICT: REJECTED
REASON: Explain what is wrong.
REQUIRED_CORRECTIONS:
- Specific correction 1
- Specific correction 2

OR

VERDICT: NEED_MORE_RETRIEVAL
REASON: Explain what evidence is missing.
REQUIRED_EVIDENCE:
- Specific missing evidence 1
- Specific missing evidence 2

Return exactly one verdict.
Do not mix verdicts.
""")




# ============================================================
# CRITIC AGENT
# ============================================================


def critic_agent(state: ResearchState):

    print("\n" + "=" * 60)
    print("CRITIC AGENT")
    print("=" * 60)

    user_query = state.get("user_query", "").strip()
    retrieved_chunks = state.get("retrieved_chunks", [])
    analysis = state.get("analysis", "")

    previous_feedback = state.get("analyst_feedback", "")
    critic_attempts = state.get("critic_attempts", 0) + 1

    print("Research question:", user_query)
    print("Retrieved chunks:", len(retrieved_chunks))
    print("Analysis available:", bool(analysis))
    print("Previous feedback:", bool(previous_feedback))

    # Missing analysis means the analyst must run.
    if not analysis:
        return {
            "critical_analysis": (
                "VERDICT: REJECTED\n"
                "REASON: Analyst's analysis is missing.\n"
                "REQUIRED_CORRECTIONS:\n"
                "- Generate an evidence-supported analysis."
            ),
            "critic_attempts": critic_attempts,
        }

    if not retrieved_chunks:
        return {
            "critical_analysis": (
                "VERDICT: NEED_MORE_RETRIEVAL\n"
                "REASON: No retrieved research evidence is available."
            ),
            "critic_attempts": critic_attempts,
        }

    MAX_CONTEXT_DOCS = 5
    MAX_CONTEXT_CHARS = 3000
    MAX_ANALYSIS_CHARS = 2000

    context_parts = []
    used = 0

    for index, doc in enumerate(
        retrieved_chunks[:MAX_CONTEXT_DOCS],
        start=1
    ):

        content = doc.get(
            "page_content",
            doc.get("text", "")
        )

        if not content:
            continue

        content = content.strip()

        paper_title = doc.get("paper_title", "Unknown")
        section = doc.get("section", "Unknown")
        page_start = doc.get("page_start")
        page_end = doc.get("page_end")
        chunk_id = doc.get("chunk_id", "Unknown")

        if page_start is not None and page_end is not None:
            page_info = (
                str(page_start)
                if page_start == page_end
                else f"{page_start}-{page_end}"
            )
        else:
            page_info = "Unknown"

        header = (
            f"[Source {index}]\n"
            f"Paper: {paper_title}\n"
            f"Section: {section}\n"
            f"Page: {page_info}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n"
        )

        remaining = MAX_CONTEXT_CHARS - used

        if len(header) >= remaining:
            break

        content = content[:remaining - len(header)]

        entry = header + content

        context_parts.append(entry)
        used += len(entry)

        if used >= MAX_CONTEXT_CHARS:
            break

    context = "\n\n".join(context_parts)

    if not context:
        return {
            "critical_analysis": (
                "VERDICT: NEED_MORE_RETRIEVAL\n"
                "REASON: Retrieved chunks contain no usable text."
            ),
            "critic_attempts": critic_attempts,
        }

    try:

        chain = critic_prompt | llm

        response = chain.invoke({
            "user_query": user_query,
            "context": context,
            "analysis": analysis[:MAX_ANALYSIS_CHARS],
            "previous_feedback": previous_feedback or "None",
        })

        critique = response.content.strip()

        print("\nCRITIC EVALUATION:")
        print(critique)

        return {
            "critical_analysis": critique,
            "critic_attempts": critic_attempts,
        }

    except Exception as e:

        print("CRITIC ERROR:", str(e))

        return {
            "critical_analysis": "",
            "error": f"Critic failed: {str(e)}",
            "critic_attempts": critic_attempts,
        }
