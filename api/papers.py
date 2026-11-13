from fastapi import APIRouter, Depends

from api.auth_utils import get_current_user
from services.document_management_service import DocumentManagementService


router = APIRouter(
    prefix="/papers",
    tags=["Uploaded Papers"]
)

document_service = DocumentManagementService()


@router.get("/uploaded")
def list_uploaded_papers(
    current_user: dict = Depends(get_current_user)
):
    """List all uploaded papers with their active and inactive versions."""

    # 1. Get authenticated user's ID
    owner_id = str(current_user["_id"])

    # 2. Fetch only this user's documents
    documents = list(
        document_service.documents.find({
            "owner_id": owner_id
        })
    )

    if not documents:
        return {
            "message": "You haven't uploaded any research papers yet.",
            "total_files": 0,
            "files": []
        }

    # 3. Get all document IDs
    document_ids = [
        document.get("document_id")
        for document in documents
    ]

    # 4. Fetch ALL versions belonging to this user
    all_versions = document_service.versions.find({
        "owner_id": owner_id,
        "document_id": {"$in": document_ids}
    }).sort("version", 1)

    # 5. Group versions by document_id
    versions_by_document = {}

    for version_record in all_versions:

        document_id = version_record.get("document_id")

        versions_by_document.setdefault(
            document_id, []
        ).append(version_record)

    # 6. Prepare response
    papers = []

    for document in documents:

        document_id = document.get("document_id")
        active_version = document.get("active_version")

        version_records = versions_by_document.get(
            document_id, []
        )

        versions = []

        for record in version_records:

            version_number = record.get("version")

            versions.append({
                "version": version_number,
                "status": record.get("status"),
                "is_active": version_number == active_version,
                "file_hash": record.get("file_hash"),
                "created_at": record.get("created_at")
            })

        papers.append({
            "document_id": document_id,
            "paper_title": document.get("paper_title") or "Untitled Paper",
            "active_version": active_version,
            "total_versions": len(versions),
            "versions": versions
        })

    return {
        "message": "Your research papers retrieved successfully.",
        "total_files": len(papers),
        "files": papers
    }