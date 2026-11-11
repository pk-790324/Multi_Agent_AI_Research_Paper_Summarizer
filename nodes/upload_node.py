
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
    document_id = None

    try:

        # Step 1: Validate file and calculate SHA-256.
        metadata = PDFService.inspect_uploaded_document(
            uploaded_path
        )

        print(f"Filename: {metadata['original_filename']}")
        print(f"SHA-256: {metadata['file_hash']}")

        # Step 2: Connect to document registry.
        document_service = DocumentManagementService()

        # Temporary single-user identity.
        # Later, replace with authenticated backend user ID.
        owner_id = state.get("user_id")
        if not owner_id:
            raise ValueError(
                "Authenticated user ID is missing from ingestion state."
            )

        # Step 3: Register document and check duplicate.
        registration = document_service.register_document(
            metadata=metadata,
            owner_id=owner_id
        )

        document_id = registration["document_id"]

        # Step 4: Stop if document already exists.
        if registration["is_duplicate"]:

            print("Duplicate document detected!")

            print(
                f"Existing document ID: {document_id}"
            )

            return {
                **state,

                "is_duplicate": True,

                "artifacts": {
                    **state.get("artifacts", {}),
                    **metadata,
                    "document_id": document_id,
                    "duplicate_status": registration["status"],
                    "duplicate_filename": registration[
                        "original_filename"
                    ]
                },

                "error": ""
            }

        # Step 5: Save the new document.
        saved_info = PDFService.save_uploaded_document(
            file_path=uploaded_path,
            document_id=document_id,
            extension=metadata["file_extension"]
        )

        # Step 6: Update MongoDB status.
        document_service.update_status(
            document_id=document_id,
            status="uploaded",
            extra_fields={
                "file_name": saved_info["file_name"],
                "file_path": saved_info["file_path"]
            }
        )

        print(f"Document saved: {saved_info['file_path']}")
        print(f"Document ID: {document_id}")

        return {
            **state,

            "is_duplicate": False,

            "artifacts": {
                **state.get("artifacts", {}),
                **metadata,
                **saved_info,
                "document_id": document_id,
                "owner_id": owner_id
            },

            "error": ""
        }

    except Exception as exc:

        print(f"Upload failed: {exc}")

        # Mark failed registrations so they can be
        # identified and recovered in a later stage.
        if document_service and document_id:
            try:
                document_service.update_status(
                    document_id=document_id,
                    status="failed",
                    error=str(exc)
                )
            except Exception as status_error:
                print(
                    f"Could not update failure status: "
                    f"{status_error}"
                )

        return {
            **state,
            "error": f"Upload failed: {exc}"
        }

    finally:

        if document_service:
            document_service.close()
