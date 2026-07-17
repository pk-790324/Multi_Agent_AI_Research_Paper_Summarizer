from langchain_core.tools import tool
import arxiv
import os
import requests


@tool
def search_arxiv(query: str) -> list[dict]:
    """
    Search arXiv for research papers based on a query.
    Returns metadata for matching papers.
    """

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []

    for paper in client.results(search):
        papers.append(
            {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": str(paper.published),
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id,
            }
        )

    return papers



def download_pdf(pdf_url: str) -> str:
    """Download an arXiv PDF and return the local path."""

    # Convert abstract URL to PDF URL
    pdf_url = pdf_url.replace("/abs/", "/pdf/")
    if not pdf_url.endswith(".pdf"):
        pdf_url += ".pdf"

    os.makedirs("papers", exist_ok=True)

    filename = pdf_url.split("/")[-1]
    filepath = os.path.join("papers", filename)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(pdf_url, headers=headers, timeout=30)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(response.content)

    return filepath