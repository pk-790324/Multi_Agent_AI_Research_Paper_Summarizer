from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from tools.embedding_tools import (
    index_latest_chunks,
)

llm = ChatOllama(
    model="minimax-m3:cloud",
    temperature=0,
)

embedding_agent = create_agent(
    model=llm,
    tools=[
        index_latest_chunks,
    ],
    system_prompt="""
You are the Embedding Agent.

Responsibilities

1. Load the newest chunk file.

2. Create embeddings.

3. Store vectors into Qdrant.

Return the indexing status.

Never summarize.

Never retrieve.

Never answer questions.
""",
)