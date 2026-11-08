
from langchain_core.prompts import ChatPromptTemplate

from state.state import ResearchState
from llm.llm import llm


# ============================================================
# ANALYST PROMPT
# ============================================================

analyst_prompt = ChatPromptTemplate.from_template("""
You are a Research Analyst Agent in a multi-agent
research paper summarization system.

USER RESEARCH QUESTION:
{query}

RETRIEVED RESEARCH EVIDENCE:
{context}

ONLY VALID SOURCE LABELS:
{valid_sources}

PREVIOUS ANALYSIS:
{previous_analysis}

CRITIC FEEDBACK FROM PREVIOUS ATTEMPT:
{analyst_feedback}

REVISION RULES:

1. If critic feedback is present, revise the previous analysis
   according to that feedback. The critic's REQUIRED_CORRECTIONS
   are mandatory, not suggestions.

2. Remove each claim the critic identifies as unsupported,
   including its citation. Do not retain it in different words.

3. Use only facts stated in RETRIEVED RESEARCH EVIDENCE. Do not
   add background knowledge, explanations, or technical details
   from memory.

4. Cite factual claims only with a label from ONLY VALID SOURCE
   LABELS, and only when that source supports the claim. Never
   invent, renumber, or copy a citation label from the previous
   analysis unless it is in the valid-label list.

5. Add information specifically requested by the critic only
   when it is supported by the retrieved evidence. If it is not
   present, say that the provided evidence does not establish it.

6. Answer the user's question, preserve supported findings, and
   remove unsupported material. Keep the answer concise; do not
   repeat points or add unnecessary sections.

7. Finish every sentence and section. Before responding, check
   that each correction is handled, every citation is valid,
   and the final character is not part of an unfinished sentence.

Write the corrected analysis only. Do not describe the revision
process or mention the critic or internal agents.
""")


# ============================================================
# ANALYST AGENT
# ============================================================

def analyst_agent(state: ResearchState):

    print("\\n" + "=" * 60)
    print("ANALYST AGENT")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. READ STATE
    # --------------------------------------------------------

    user_query = state.get("user_query", "")

    retrieved_chunks = state.get(
        "retrieved_chunks", []
    )

    analyst_feedback = state.get(
        "analyst_feedback", ""
    )

    previous_analysis = state.get("analysis", "")

    print("Research question:", user_query)
    print("Retrieved chunks:", len(retrieved_chunks))
    print("Previous critic feedback:", bool(analyst_feedback))

    if analyst_feedback:
        print("\\nCRITIC FEEDBACK:")
        print(analyst_feedback)

    # --------------------------------------------------------
    # 2. BUILD CONTEXT
    # --------------------------------------------------------

    MAX_CONTEXT_DOCS = 5
    MAX_CONTEXT_CHARS = 3500

    docs = retrieved_chunks[:MAX_CONTEXT_DOCS]

    context_parts = []
    current_length = 0

    for index, doc in enumerate(docs, start=1):

        content = doc.get(
            "page_content",
            doc.get("text", "")
        )

        if not content:
            continue

        content = content.strip()

        if current_length >= MAX_CONTEXT_CHARS:
            break

        remaining = MAX_CONTEXT_CHARS - current_length

        content = content[:remaining]

        paper_title = doc.get(
            "paper_title", "Unknown"
        )

        section = doc.get(
            "section", "Unknown"
        )

        chunk_id = doc.get(
            "chunk_id", "Unknown"
        )

        page_start = doc.get("page_start")
        page_end = doc.get("page_end")

        if page_start is not None and page_end is not None:
            page_info = (
                str(page_start)
                if page_start == page_end
                else f"{page_start}-{page_end}"
            )
        else:
            page_info = "Unknown"

        chunk_text = (
            f"[Source {index}]\n"
            f"Paper: {paper_title}\n"
            f"Section: {section}\n"
            f"Page: {page_info}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n{content}"
        )

        context_parts.append(chunk_text)

        current_length += len(content)

    context = "\n\n".join(context_parts)

    if not context:
        return {
            "analysis": "",
            "error": "Analyst cannot proceed: no usable retrieved evidence."
        }

    valid_sources = ", ".join(
        f"[Source {index}]"
        for index in range(1, len(context_parts) + 1)
    )

    # --------------------------------------------------------
    # 3. GENERATE ANALYSIS
    # --------------------------------------------------------

    chain = analyst_prompt | llm

    response = chain.invoke({
        "query": user_query,
        "context": context,
        "valid_sources": valid_sources,
        "previous_analysis": (
            previous_analysis
            if previous_analysis
            else "No previous analysis. This is the initial analysis."
        ),
        "analyst_feedback": (
            analyst_feedback
            if analyst_feedback
            else "No previous feedback. This is the initial analysis."
        )
    })

    analysis = response.content.strip()

    print("\\nANALYST OUTPUT:")
    print(analysis)
    print("\\nANALUST FEEDBACK")
    print(analyst_feedback)

    # --------------------------------------------------------
    # 4. RETURN UPDATED STATE
    # --------------------------------------------------------

    return {
        "analysis": analysis,
        "analyst_feedback": ""
    }
