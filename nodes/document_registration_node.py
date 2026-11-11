
from state.state import ResearchPaperState

from services.document_management_service import (
    DocumentManagementService
)


def document_registration_node(
    state: ResearchPaperState
) -> ResearchPaperState:

    print("=" * 60)
    print("DOCUMENT REGISTRATION NODE")
    print("=" * 60)

    artifacts = state.get("artifacts", {})

    service = DocumentManagementService()

    try:

        metadata = {
            "paper_title": artifacts.get("paper_title"),
            "doi": artifacts.get("doi"),

            "file_hash": artifacts["file_hash"],
            "original_filename": artifacts["original_filename"],
            "file_extension": artifacts["file_extension"],
            "file_size_bytes": artifacts["file_size_bytes"],
        }

        registration = service.register_document_version(
            metadata=metadata,
            owner_id=artifacts["owner_id"],
            ingestion_id=artifacts["ingestion_id"],
        )

        if registration["is_duplicate"]:

            return {
                **state,
                "is_duplicate": True,
                "error": "",
            }

        document_id = registration["document_id"]
        version = registration["version"]

        print(f"Document ID: {document_id}")
        print(f"Document Version: {version}")

        return {
            **state,

            "artifacts": {
                **artifacts,

                "document_id": document_id,
                "version": version,

                "identity_key": registration["identity_key"],

                "version_status": "pending",
            },

            "error": "",
        }

    except Exception as exc:

        print(f"Registration failed: {exc}")

        return {
            **state,
            "error": f"Document registration failed: {exc}",
        }

    finally:
        service.close()
