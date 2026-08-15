from langchain_core.prompts import ChatPromptTemplate
from state.state import ResearchState
from llm.llm import llm

synthesizer_prompt = ChatPromptTemplate.from_template("""
You are the Synthesizer Agent in a multi-agent
research paper analysis system.

Your job is to produce the final answer to the user's question
using the retrieved evidence, analyst's analysis, and critic's
evaluation.

USER QUESTION:
{user_query}

RETRIEVED EVIDENCE:
{context}

ANALYST'S ANALYSIS:
{analysis}

CRITIC'S EVALUATION:
{critique}


INSTRUCTIONS:

1. Answer the user's question directly.
2. Use only information supported by the retrieved evidence.
3. Do not invent facts or information.
4. Do not introduce information from your own knowledge.
5. Combine the strongest points from the analyst's analysis.
6. Respect the critic's verification.
7. If the evidence is insufficient, clearly state that.
8. Structure the answer clearly.
9. When possible, mention the paper section or page information
   available in the retrieved metadata.
10. If supporting background from cited works is included in the evidence,
    explicitly mention the short author-style reference names such as
    Bahdanau et al., Hochreiter & Schmidhuber, Sutskever et al., or similar,
    instead of raw numbers.
11. Do not mention the internal agents or orchestration process.


FINAL ANSWER:
""")


def synthesizer_agent(state:ResearchState):

    # -----------------------------------------
    # Build context from retrieved documents
    # -----------------------------------------
    
    print("\n" + "=" * 60)
    print("SYNTHESIZER AGENT")
    print("=" * 60)

    # -------------------------------------------------------
    # Truncate context to stay within Groq's free-tier limit.
    # Budget (chars ≈ tokens * 4):
    #   prompt template              ≈  600 chars
    #   query + analysis + critique  ≈ 2,000 chars
    #   evidence                     ≈ 3,000 chars
    #   total                        ≈ 5,600 chars  < 6,000 TPM
    # -------------------------------------------------------
    MAX_CONTEXT_DOCS  = 8
    MAX_CONTEXT_CHARS = 3000

    context_parts = []
    used = 0

    for doc in state["retrieved_docs"][:MAX_CONTEXT_DOCS]:
        metadata = doc.metadata

        paper_title    = metadata.get("paper_title", "Unknown")
        section        = metadata.get("section", "Unknown")
        page           = metadata.get("page_number", "Unknown")
        citation_ids   = metadata.get("citation_ids", [])
        citation_labels = metadata.get("citation_labels", [])

        citation_text = ""
        if citation_labels:
            citation_text = f"\nCitations: [{', '.join(str(v) for v in citation_labels)}]"
        elif citation_ids:
            citation_text = f"\nCitations: [{', '.join(str(v) for v in citation_ids)}]"

        content = doc.page_content.strip()
        header  = f"Paper: {paper_title}\nSection: {section}\nPage: {page}{citation_text}\n\nContent:\n"
        entry   = header + content

        if used + len(entry) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - used
            if remaining < 120:
                break
            # Keep the header and truncate only the body
            body_budget = remaining - len(header)
            if body_budget > 0:
                entry = header + content[:body_budget]
            else:
                break

        context_parts.append(entry)
        used += len(entry)

    context = "\n\n".join(context_parts)


    # -----------------------------------------
    # Generate final answer
    # -----------------------------------------

    chain = synthesizer_prompt | llm

    response = chain.invoke({
        "user_query": state["user_query"],
        "context": context,
        "analysis": state["analysis"],
        "critique": state["critique"],
    })

    return {
        "final_answer": response.content
    }