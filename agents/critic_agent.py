
from langchain_core.prompts import ChatPromptTemplate

from state.state import ResearchState
from llm.llm import llm


# ============================================================
# CRITIC PROMPT
# ============================================================

critic_prompt = ChatPromptTemplate.from_template("""
You are a critical reviewer in a research paper analysis system.

Your responsibility is to evaluate the quality, correctness,
and evidence support of the Analyst's response.

USER QUESTION:
{user_query}

RETRIEVED EVIDENCE:
{context}

ANALYST'S ANALYSIS:
{analysis}

EVALUATION INSTRUCTIONS:

1. Check whether the analysis answers the user's question.
2. Verify whether important claims are supported by the retrieved evidence.
3. Identify unsupported or potentially misleading claims.
4. Identify important missing information.
5. Check whether the analysis accurately represents the research findings.
6. Check whether the methodology and technical explanations are correct
   according to the supplied evidence.
7. Do not introduce information from your own knowledge.
8. Do not request additional retrieval unless the missing information
   is genuinely necessary to answer the user's question.

DECISION:

If the evidence is sufficient and the analysis is acceptable, return:

VERDICT: APPROVED
REASON: <explain why the analysis is acceptable>

If important evidence is missing, return:

VERDICT: NEED_MORE_RETRIEVAL
REASON: <explain exactly what evidence is missing>

IMPORTANT:
- Be specific in your evaluation.
- Do not approve unsupported claims.
- Do not request retrieval simply because additional information
  would be interesting.
""")


# ============================================================
# CRITIC AGENT
# ============================================================

def critic_agent(state: ResearchState):

    print("\n" + "=" * 60)
    print("CRITIC AGENT")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. READ STATE
    # --------------------------------------------------------

    user_query = state.get("user_query", "").strip()

    retrieved_chunks = state.get(
        "retrieved_chunks",
        []
    )

    analysis = state.get(
        "analysis",
        ""
    )

    print("Research question:", user_query)
    print("Retrieved chunks:", len(retrieved_chunks))
    print("Analysis available:", bool(analysis))

    # --------------------------------------------------------
    # 2. VALIDATE INPUT
    # --------------------------------------------------------

    if not analysis:

        return {
            "critical_analysis": (
                "VERDICT: NEED_MORE_RETRIEVAL\n"
                "REASON: Analyst's analysis is missing."
            )
        }

    if not retrieved_chunks:

        return {
            "critical_analysis": (
                "VERDICT: NEED_MORE_RETRIEVAL\n"
                "REASON: No retrieved research evidence is available."
            )
        }

    # --------------------------------------------------------
    # 3. BUILD CONTEXT
    # --------------------------------------------------------

    MAX_CONTEXT_DOCS = 5
    MAX_CONTEXT_CHARS = 3000
    MAX_ANALYSIS_CHARS = 2000

    context_parts = []
    used = 0

    for index, doc in enumerate(
        retrieved_chunks[:MAX_CONTEXT_DOCS],
        start=1
    ):

        # Retrieved chunks are dictionaries
        content = doc.get(
            "page_content",
            doc.get("text", "")
        )

        if not content:
            continue

        content = content.strip()

        paper_title = doc.get(
            "paper_title",
            "Unknown"
        )

        section = doc.get(
            "section",
            "Unknown"
        )

        page_start = doc.get("page_start")
        page_end = doc.get("page_end")

        chunk_id = doc.get(
            "chunk_id",
            "Unknown"
        )

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

        body_budget = remaining - len(header)

        content = content[:body_budget]

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
            )
        }

    analysis = analysis[:MAX_ANALYSIS_CHARS]

    # --------------------------------------------------------
    # 4. EVALUATE ANALYSIS
    # --------------------------------------------------------

    try:

        chain = critic_prompt | llm

        response = chain.invoke({
            "user_query": user_query,
            "context": context,
            "analysis": analysis,
        })

        critique = response.content.strip()

        print("\nCRITIC EVALUATION:")
        print(critique)

        # ----------------------------------------------------
        # 5. RETURN RESULT
        # ----------------------------------------------------

        return {
            "critical_analysis": critique
        }

    except Exception as e:

        print("CRITIC ERROR:", str(e))

        return {
            "critical_analysis": "",
            "error": f"Critic failed: {str(e)}"
        }
