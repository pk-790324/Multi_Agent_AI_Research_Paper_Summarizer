import os
import requests


class PDFService:

    @staticmethod
    def download_pdf(pdf_url: str) -> str:
        """
        Download an arXiv PDF.

        Args:
            pdf_url: PDF or abstract URL.

        Returns:
            Local file path.
        """

        pdf_url = pdf_url.replace("/abs/", "/pdf/")

        if not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"

        os.makedirs("papers", exist_ok=True)

        filename = pdf_url.split("/")[-1]
        filepath = os.path.join("papers", filename)

        response = requests.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )

        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        return filepath