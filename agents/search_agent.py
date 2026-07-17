from langgraph.prebuilt import create_react_agent

from tools.search_tools import (
    search_arxiv,
    download_pdf,
)

from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="minimax-m3:cloud",
    temperature=0,
)

search_agent = create_react_agent(
    model=llm,
    tools=[
        search_arxiv,
        download_pdf,
    ],
    prompt="""
You are the Search Agent.

Responsibilities:
1. Search arXiv for relevant papers.
2. Download PDFs when requested.

Do NOT summarize papers.
Do NOT analyze papers.
Do NOT answer research questions.

Only retrieve papers and download PDFs.
Return structured information.
""",
)