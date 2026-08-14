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

    context = "\n\n".join(
        doc.page_content
        for doc in state["retrieved_docs"]
    )

    chain = analyst_prompt | llm

    response = chain.invoke({
        "user_query": state["user_query"],
        "context": context
    })

    return {
        "analysis": response.content
    }