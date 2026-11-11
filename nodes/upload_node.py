
import uuid

from state.state import ResearchPaperState

from services.pdf_service import PDFService
from services.document_management_service import (
    DocumentManagementService
)


def upload_node(state: ResearchPaperState) -> ResearchPaperState:

    print("=" * 60)
    print("UPLOAD NODE")
    print("=" * 60)

    uploaded_path = (
        state.get("artifacts", {})
        .get("uploaded_file_path")
    )

    if not uploaded_path:
        return {
            **state,
            "error": "No uploaded file path found."
        }

    document_service = None

    try:

        # 1. Validate file and calculate SHA-256.
        metadata = PDFService.inspect_uploaded_document(
            uploaded_path
        )

        print(f"Filename: {metadata['original_filename']}")
        print(f"SHA-256: {metadata['file_hash']}")

        # 2. Get authenticated user.
        owner_id = state.get("user_id")

        if not owner_id:
            raise ValueError(
                "Authenticated user ID is missing."
            )

        # 3. Connect to MongoDB.
        document_service = DocumentManagementService()

        # 4. Check exact file duplicate.
        duplicate = document_service.find_duplicate_version(
            file_hash=metadata["file_hash"],
            owner_id=owner_id
        )

        if duplicate:

            print("Exact duplicate detected!")

            return {
                **state,
                "is_duplicate": True,
                "artifacts": {
                    **state.get("artifacts", {}),
                    **metadata,
                    **duplicate,
                    "owner_id": owner_id,
                },
                "error": "",
            }

        # 5. Generate unique ingestion ID.
        ingestion_id = str(uuid.uuid4())

        # 6. Save PDF using ingestion ID.
        saved_info = PDFService.save_uploaded_document(
            file_path=uploaded_path,
            document_id=ingestion_id,
            extension=metadata["file_extension"]
        )

        print(f"PDF saved: {saved_info['file_path']}")
        print(f"Ingestion ID: {ingestion_id}")

        # 7. Update graph state.
        return {
            **state,

            "is_duplicate": False,

            "artifacts": {
                **state.get("artifacts", {}),
                **metadata,
                **saved_info,

                "ingestion_id": ingestion_id,
                "owner_id": owner_id,
            },

            "error": "",
        }

    except Exception as exc:

        print(f"Upload failed: {exc}")

        return {
            **state,
            "error": f"Upload failed: {exc}"
        }

    finally:

        if document_service:
            document_service.close()
