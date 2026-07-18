from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from tools.retrievers_tools import retrieve_documents


llm = ChatOllama(
    model="minimax-m3:cloud",
    temperature=0,
)

retriever_agent = create_agent(
    model=llm,
    tools=[retrieve_documents],
    system_prompt="""
You are the Retriever Agent.

Responsibilities

1. Understand the user's question.
2. Retrieve the most relevant chunks.
3. Return ONLY the retrieved chunks.

Never summarize.

Never answer the question.

Never analyze.

Never rewrite the retrieved text.
""",
)