
import os
import uuid

from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

load_dotenv()


class DocumentManagementService:
    """Manage research document metadata in MongoDB."""

    def __init__(self):

        mongodb_uri = os.getenv("MONGODB_URI")

        if not mongodb_uri:
            raise ValueError(
                "MONGODB_URI is missing from environment variables."
            )

        database_name = os.getenv(
            "MONGODB_DATABASE",
            "research_paper_summarizer"
        )

        collection_name = os.getenv(
            "MONGODB_COLLECTION",
            "documents"
        )

        self.client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000
        )

        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

        self._create_indexes()

    def _create_indexes(self):
        """Create indexes for document lookup and deduplication."""

        # Same file can belong to different users.
        # But one user cannot register identical content twice.
        self.collection.create_index(
            [
                ("owner_id", ASCENDING),
                ("file_hash", ASCENDING)
            ],
            unique=True,
            name="unique_user_document_hash"
        )

        self.collection.create_index(
            [("status", ASCENDING)],
            name="document_status_index"
        )

    def register_document(
        self,
        metadata: dict,
        owner_id: str = "default"
    ) -> dict:
        """Register a document or return existing duplicate."""

        document_id = str(uuid.uuid4())

        document = {
            "_id": document_id,
            "document_id": document_id,

            "owner_id": owner_id,

            "original_filename": metadata["original_filename"],
            "file_hash": metadata["file_hash"],
            "file_extension": metadata["file_extension"],
            "file_size_bytes": metadata["file_size_bytes"],

            "version": 1,

            "status": "pending",

            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),

            "error": None
        }

        try:

            self.collection.insert_one(document)

            return {
                "is_duplicate": False,
                "document_id": document_id,
                "status": "pending"
            }

        except DuplicateKeyError:

            existing = self.collection.find_one({
                "owner_id": owner_id,
                "file_hash": metadata["file_hash"]
            })

            if existing is None:
                raise RuntimeError(
                    "Document registration conflict. "
                    "Please retry the upload."
                )

            return {
                "is_duplicate": True,
                "document_id": existing["document_id"],
                "status": existing["status"],
                "original_filename": existing["original_filename"]
            }

    def update_status(
        self,
        document_id: str,
        status: str,
        error: str = None,
        extra_fields: dict = None
    ):
        """Update document processing status."""

        update_fields = {
            "status": status,
            "error": error,
            "updated_at": datetime.now(timezone.utc)
        }

        if extra_fields:
            update_fields.update(extra_fields)

        result = self.collection.update_one(
            {"document_id": document_id},
            {"$set": update_fields}
        )

        if result.matched_count == 0:
            raise ValueError(
                f"Document not found: {document_id}"
            )

    def get_document(self, document_id: str):
        """Retrieve document metadata."""

        return self.collection.find_one({
            "document_id": document_id
        })

    def close(self):
        self.client.close()
