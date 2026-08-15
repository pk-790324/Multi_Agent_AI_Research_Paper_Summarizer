from langchain_core.prompts import ChatPromptTemplate
from state.state import ResearchState
from llm.llm import llm 


critic_prompt = ChatPromptTemplate.from_template("""
You are a critical reviewer of a research paper analysis.

User question:
{user_query}

Retrieved evidence:
{context}

Analyst's analysis:
{analysis}

Evaluate the analysis.

Check:

1. Is the answer supported by the retrieved evidence?
2. Are there unsupported claims?
3. Is important information missing?
4. Does the analysis actually answer the user's question?

Return your evaluation using this format:

VERDICT: APPROVED
REASON: <reason>

OR

VERDICT: NEED_MORE_RETRIEVAL
REASON: <what information is missing>
""")


def critic_agent(state: ResearchState):
    print("\n" + "=" * 60)
    print("CRITIC AGENT")
    print("=" * 60)

    # -------------------------------------------------------
    # Truncate context to stay within Groq's free-tier limit
    # (llama-3.1-8b-instant: 6,000 TPM).
    # Budget breakdown (chars ≈ tokens * 4):
    #   prompt template + analysis + query ≈ 2,000 chars
    #   evidence budget                   ≈ 2,000 chars
    # -------------------------------------------------------
    MAX_CONTEXT_DOCS  = 5
    MAX_CONTEXT_CHARS = 2000
    MAX_ANALYSIS_CHARS = 800

    docs = state["retrieved_docs"][:MAX_CONTEXT_DOCS]
    context_parts = []
    used = 0
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
        if used + len(content) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - used
            if remaining < 80:
                break
            content = content[:remaining]
        context_parts.append(content)
        used += len(content)

    context = "\n\n".join(context_parts)
    analysis = (state["analysis"] or "")[:MAX_ANALYSIS_CHARS]

    chain = critic_prompt | llm

    response = chain.invoke({
        "user_query": state["user_query"],
        "context": context,
        "analysis": analysis,
    })

    return {
        "critique": response.content
    }