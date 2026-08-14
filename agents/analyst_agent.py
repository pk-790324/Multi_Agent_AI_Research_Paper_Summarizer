from langchain_core.prompts import ChatPromptTemplate
from state.state import ResearchState
from llm.llm import llm 

analyst_prompt = ChatPromptTemplate.from_template("""
You are a research paper analysis agent.

Analyze the retrieved research-paper content and answer
the user's question.

User question:
{user_query}

Retrieved content:
{context}

Instructions:
1. Use only the retrieved content.
2. Do not invent information.
3. Identify important findings and methodology.
4. Explain the reasoning clearly.
5. If the evidence is insufficient, explicitly say so.
""")


def analyst_agent(state: ResearchState):
    print("\n" + "=" * 60)
    print("ANALYST AGENT")
    print("=" * 60)

    MAX_CONTEXT_DOCS = 5
    MAX_CONTEXT_CHARS = 3500

    docs = state["retrieved_docs"][:MAX_CONTEXT_DOCS]
    context_parts = []
    current_length = 0

    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
        if current_length + len(content) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - current_length
            if remaining <= 100:
                break
            content = content[:remaining]
        context_parts.append(content)
        current_length += len(content)

    context = "\n\n".join(context_parts)

    chain = analyst_prompt | llm

    response = chain.invoke({
        "user_query": state["user_query"],
        "context": context
    })

    return {
        "analysis": response.content
    }