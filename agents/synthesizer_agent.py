
from langchain_core.prompts import ChatPromptTemplate

from state.state import ResearchState
from llm.llm import llm


# ============================================================
# SYNTHESIZER PROMPT
# ============================================================

synthesizer_prompt = ChatPromptTemplate.from_template("""
You are the final response generation agent.

Your responsibility is to produce an accurate,
relevant, evidence-grounded answerusing the retrieved research evidence, analyst's
analysis, and critic's evaluation.

USER QUESTION:
{user_query}

RETRIEVED EVIDENCE:
{context}

ANALYST'S ANALYSIS:
{analysis}

CRITIC'S EVALUATION:
{critique}

CITATION-VERIFIED ANALYSIS:
{cited_analysis}


INSTRUCTIONS:

1. Answer the user's question directly.
2. Use only information supported by the supplied evidence.
3. Do not invent facts or introduce unsupported background knowledge.
4. Combine the strongest findings from the analyst's analysis.
5. Respect the critic's evaluation and identified limitations.
6. Preserve valid inline source citations such as [S1] and [S2].
7. Do not create new citation IDs.
8. If evidence is insufficient, explicitly state this.
9. Structure the answer with meaningful headings where appropriate.
10. Mention paper titles, sections, and page numbers when available.
11. Explain technical concepts clearly without changing their meaning.
12. Do not mention internal agents or the orchestration process.
13. Match the response length to the complexity of the user's question.
14. For simple factual questions, provide a concise,
    direct answer without unnecessary explanations,
    reference lists, or unrelated findings.
15. For complex research questions, provide detailed
    explanations with relevant evidence and citations.
16. Include only citations that directly support
    the claims made in the answer. Do not include
    unrelated bibliographic references.
17. Never confuse authors of cited papers with authors
    of the primary research paper. If the primary
    paper's author information is unavailable, explicitly
    state that the exact author count cannot be determined.

EXPECTED OUTPUT STRUCTURE:

## Answer
Direct answer to the user's question.

## Key Findings
Important findings supported by the retrieved evidence.

## Explanation
A clear explanation of the relevant concepts and methodology.

## Evidence and Sources
Supporting claims with their original citation IDs.

## Limitations
Important limitations or missing evidence, if applicable.

FINAL ANSWER:
""")


# ============================================================
# SYNTHESIZER AGENT
# ============================================================

def synthesizer_agent(state: ResearchState):

    print("\n" + "=" * 60)
    print("SYNTHESIZER AGENT")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. READ STATE
    # --------------------------------------------------------

    user_query = state.get("user_query", "").strip()

    retrieved_chunks = state.get(
        "retrieved_chunks",
        []
    )

    analysis = state.get("analysis", "")

    critique = (
        state.get("critical_analysis")
        or state.get("critique", "")
    )

    cited_analysis = state.get(
        "cited_analysis",
        ""
    )

    print("Research question:", user_query)
    print("Retrieved chunks:", len(retrieved_chunks))
    print("Analysis available:", bool(analysis))
    print("Critique available:", bool(critique))
    print("Cited analysis available:", bool(cited_analysis))

    # --------------------------------------------------------
    # 2. VALIDATE INPUT
    # --------------------------------------------------------

    if not user_query:
        return {
            "final_answer": "Unable to generate an answer: research question is missing."
        }

    if not retrieved_chunks:
        return {
            "final_answer": (
                "I could not generate a research-based answer because "
                "no relevant research-paper evidence was retrieved."
            )
        }

    # --------------------------------------------------------
    # 3. BUILD CONTEXT FROM RETRIEVED CHUNKS
    # --------------------------------------------------------

    MAX_CONTEXT_DOCS = 8
    MAX_CONTEXT_CHARS = 6000

    context_parts = []
    used = 0

    for index, doc in enumerate(
        retrieved_chunks[:MAX_CONTEXT_DOCS],
        start=1
    ):

        # Retrieved chunks are dictionaries
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

        # Your hybrid search service returns page_content
        content = doc.get(
            "page_content",
            doc.get("text", "")
        )

        if not content:
            continue

        content = content.strip()

        # Format page information
        if page_start is not None and page_end is not None:

            if page_start == page_end:
                page_info = str(page_start)
            else:
                page_info = f"{page_start}-{page_end}"

        else:
            page_info = "Unknown"

        header = (
            f"[Retrieved Chunk {index}]\n"
            f"Paper: {paper_title}\n"
            f"Section: {section}\n"
            f"Page: {page_info}\n"
            f"Chunk ID: {chunk_id}\n\n"
            f"Content:\n"
        )

        entry = header + content

        remaining = MAX_CONTEXT_CHARS - used

        if len(entry) > remaining:

            body_budget = remaining - len(header)

            if body_budget <= 0:
                break

            entry = header + content[:body_budget]

        context_parts.append(entry)

        used += len(entry)

        if used >= MAX_CONTEXT_CHARS:
            break

    context = "\n\n".join(context_parts)

    print("Context characters:", len(context))
    print("Context chunks:", len(context_parts))

    if not context:
        return {
            "final_answer": (
                "The retrieved chunks did not contain usable text."
            )
        }

    # --------------------------------------------------------
    # 4. GENERATE FINAL ANSWER
    # --------------------------------------------------------

    try:

        chain = synthesizer_prompt | llm

        response = chain.invoke({
            "user_query": user_query,
            "context": context,
            "analysis": analysis or "No analysis available.",
            "critique": critique or "No critical evaluation available.",
            "cited_analysis": (
                cited_analysis
                or "No citation-verified analysis available."
            ),
        })

        final_answer = response.content

        print("\nFINAL ANSWER GENERATED SUCCESSFULLY")

        return {
            "final_answer": final_answer
        }

    except Exception as e:

        print("SYNTHESIZER ERROR:", str(e))

        return {
            "final_answer": "",
            "error": f"Synthesizer failed: {str(e)}"
        }
