from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from tools.parser_tools import parse_pdf


llm = ChatOllama(
    model="minimax-m3:cloud",
    temperature=0,
)

parser_agent = create_agent(
    model=llm,
    tools=[parse_pdf],
    system_prompt="""
You are the PDF Parsing Agent.

Responsibilities:
- Read PDFs from the papers folder.
- Extract all readable text.
- Return only the extracted text.

Do NOT summarize.
Do NOT chunk.
Do NOT analyze.
Do NOT create embeddings.
""",
)