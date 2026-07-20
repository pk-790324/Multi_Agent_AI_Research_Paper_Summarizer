from state.state import ResearchPaperState
from services.pdf_service import PDFService


def download_node(state: ResearchPaperState) -> ResearchPaperState:

    pdf_path = PDFService.download_pdf(
        state["paper"][0]["pdf_url"]
    )

    return {
        **state,
        "artifacts": {
            **state.get("artifacts", {}),
            "pdf_path": pdf_path,
        },
    }