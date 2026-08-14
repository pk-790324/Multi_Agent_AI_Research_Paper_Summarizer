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
10. Do not mention the internal agents or orchestration process.


FINAL ANSWER:
""")


def synthesizer_agent(state:ResearchState):

    # -----------------------------------------
    # Build context from retrieved documents
    # -----------------------------------------
    
    print("\n" + "=" * 60)
    print("SYNTHESIZER AGENT")
    print("=" * 60)

    context_parts = []

    for doc in state["retrieved_docs"]:

        metadata = doc.metadata

        paper_title = metadata.get(
            "paper_title",
            "Unknown"
        )

        section = metadata.get(
            "section",
            "Unknown"
        )

        page = metadata.get(
            "page_number",
            "Unknown"
        )

        context_parts.append(
            f"""
Paper: {paper_title}
Section: {section}
Page: {page}

Content:
{doc.page_content}
"""
        )

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