import re
import arxiv


class ArxivService:

    @staticmethod
    def search_papers(
        query: str,
    ) -> list:
        """
        Retrieve the exact arXiv paper from its URL.

        Example:
        https://arxiv.org/abs/2607.16122
        """

        # Extract arXiv ID
        match = re.search(
            r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)(?:\.pdf)?",
            query,
        )

        if not match:
            raise ValueError("Invalid arXiv URL.")

        arxiv_id = match.group(1)

        client = arxiv.Client()

        search = arxiv.Search(
            id_list=[arxiv_id],
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