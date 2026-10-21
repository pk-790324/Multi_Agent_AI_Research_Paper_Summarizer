
from state.state import ResearchPaperState
from services.pdf_service import PDFService





def upload_node(state: ResearchPaperState) -> ResearchPaperState:
    """
    Validate and save the uploaded document.
    """

    print("=" * 60)
    print("UPLOAD NODE")
    print("=" * 60)

    # The initial state must contain the temporary uploaded file path.
    uploaded_path = state.get("artifacts", {}).get("uploaded_file_path")

    try:
        document_info = PDFService.save_uploaded_document(
            uploaded_path
        )

        saved_path = document_info["file_path"]

        print(f"Document saved: {saved_path}")

        return {
            **state,
            "artifacts": {
                **state.get("artifacts", {}),
                **document_info,
                "file_path": saved_path,
            },
            "error": "",
        }

    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"Upload failed: {exc}")

        return {
            **state,
            "error": f"Upload failed: {exc}",
        }
