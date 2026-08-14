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

    context = "\n\n".join(
        doc.page_content
        for doc in state["retrieved_docs"]
    )

    chain = critic_prompt | llm

    response = chain.invoke({
        "user_query": state["user_query"],
        "context": context,
        "analysis": state["analysis"]
    })

    return {
        "critique": response.content
    }