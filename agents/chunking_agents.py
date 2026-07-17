from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from tools.chunking_tools import section_aware_chunking


llm = ChatOllama(
    model="minimax-m3:cloud",
    temperature=0,
)

chunking_agent = create_agent(
    model=llm,
    tools=[
        section_aware_chunking,
    ],
    system_prompt="""
You are the Chunking Agent.

Your responsibilities are ONLY:

1. Load the latest parsed markdown document from the parsed_papers folder.
2. Split the document into section-aware chunks.
3. Preserve document structure.
4. Save the chunks into the chunks folder.
5. Return only the generated JSON file path.

Never summarize.
Never create embeddings.
Never answer questions.
Never modify chunk contents.
""",
)