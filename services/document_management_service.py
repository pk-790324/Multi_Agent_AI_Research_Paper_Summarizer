
import os
import re
import uuid

from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import (
    MongoClient,
    ASCENDING,
    ReturnDocument,
)
from pymongo.errors import DuplicateKeyError, OperationFailure

load_dotenv()


class DocumentManagementService:
    """Manage logical research papers and their uploaded versions."""

    @staticmethod
    def _transactions_unsupported(error: OperationFailure) -> bool:
        return (
            error.code == 20
            and "transaction numbers are only allowed"
            in str(error).lower()
        )

    def __init__(self):

        mongodb_uri = os.getenv("MONGODB_URI")

        if not mongodb_uri:
            raise ValueError("MONGODB_URI is missing.")

        database_name = os.getenv(
            "MONGODB_DATABASE",
            "research_paper_summarizer"
        )

        self.client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000
        )

        self.database = self.client[database_name]

        self.documents = self.database[
            os.getenv("MONGODB_DOCUMENTS_COLLECTION", "documents")
        ]

        self.versions = self.database[
            os.getenv(
                "MONGODB_VERSIONS_COLLECTION",
                "document_versions"
            )
        ]

        self._create_indexes()

    # --------------------------------------------------
    # 1. CREATE INDEXES
    # --------------------------------------------------

    def _create_indexes(self):

        # One logical paper per user and identity key.
        self.documents.create_index(
            [
                ("owner_id", ASCENDING),
                ("identity_key", ASCENDING),
            ],
            unique=True,
            name="unique_user_document_identity"
        )

        # Prevent identical content from being registered twice
        # for the same user.
        self.versions.create_index(
            [
                ("owner_id", ASCENDING),
                ("file_hash", ASCENDING),
            ],
            unique=True,
            name="unique_user_version_hash"
        )

        # One version number per logical document.
        self.versions.create_index(
            [
                ("document_id", ASCENDING),
                ("version", ASCENDING),
            ],
            unique=True,
            name="unique_document_version"
        )

        self.versions.create_index(
            [
                ("owner_id", ASCENDING),
                ("document_id", ASCENDING),
                ("status", ASCENDING),
            ],
            name="document_version_status"
        )

    # --------------------------------------------------
    # 2. NORMALIZE DOI
    # --------------------------------------------------

    @staticmethod
    def normalize_doi(doi: str) -> str:

        if not doi:
            return ""

        doi = doi.strip().lower()

        doi = re.sub(
            r"^https?://(dx\.)?doi\.org/",
            "",
            doi
        )

        doi = re.sub(r"^doi:\s*", "", doi)

        return doi.rstrip(".,; ")

    # --------------------------------------------------
    # 3. NORMALIZE PAPER TITLE
    # --------------------------------------------------

    @staticmethod
    def normalize_title(title: str) -> str:

        if not title:
            return ""

        title = title.lower().strip()

        title = re.sub(
            r"[^a-z0-9]+",
            " ",
            title
        )

        return " ".join(title.split())

    # --------------------------------------------------
    # 4. GENERATE DOCUMENT IDENTITY
    # --------------------------------------------------

    @classmethod
    def generate_identity_key(
        cls,
        paper_title: str,
        doi: str = None
    ) -> str:

        normalized_doi = cls.normalize_doi(doi)

        if normalized_doi:
            return f"doi:{normalized_doi}"

        normalized_title = cls.normalize_title(
            paper_title
        )

        if not normalized_title:
            raise ValueError(
                "A valid paper title or DOI is required."
            )

        return f"title:{normalized_title}"

    # --------------------------------------------------
    # 5. CHECK EXACT FILE DUPLICATE
    # --------------------------------------------------

    def find_duplicate_version(
        self,
        file_hash: str,
        owner_id: str
    ) -> dict | None:

        existing = self.versions.find_one({
            "owner_id": owner_id,
            "file_hash": file_hash
        })

        if not existing:
            return None

        return {
            "is_duplicate": True,
            "document_id": existing["document_id"],
            "version": existing["version"],
            "ingestion_id": existing["ingestion_id"],
            "status": existing["status"],
            "original_filename": existing["original_filename"],
        }

    # --------------------------------------------------
    # 6. REGISTER NEW DOCUMENT VERSION
    # --------------------------------------------------

    def register_document_version(
        self,
        metadata: dict,
        owner_id: str,
        ingestion_id: str = None,
    ) -> dict:
        """
        Resolve the logical document and register a new version.

        MongoDB transactions are used when supported. Standalone MongoDB
        falls back to atomic single-document writes.
        """

        if not owner_id:
            raise ValueError("owner_id is required.")

        paper_title = metadata.get("paper_title", "").strip()
        doi = metadata.get("doi")

        identity_key = self.generate_identity_key(
            paper_title=paper_title,
            doi=doi
        )

        file_hash = metadata["file_hash"]

        ingestion_id = ingestion_id or str(uuid.uuid4())

        now = datetime.now(timezone.utc)

        # Fast duplicate check.
        duplicate = self.find_duplicate_version(
            file_hash=file_hash,
            owner_id=owner_id
        )

        if duplicate:
            return duplicate

        document_id = str(uuid.uuid4())

        try:

            with self.client.start_session() as session:

                with session.start_transaction():

                    # Create the logical document if it doesn't exist.
                    document = self.documents.find_one_and_update(
                        {
                            "owner_id": owner_id,
                            "identity_key": identity_key,
                        },
                        {
                            "$setOnInsert": {
                                "document_id": document_id,
                                "owner_id": owner_id,
                                "identity_key": identity_key,
                                "paper_title": paper_title,
                                "doi": doi,
                                "active_version": None,
                                "next_version": 0,
                                "status": "pending",
                                "created_at": now,
                            },
                            "$set": {
                                "updated_at": now,
                            },
                        },
                        upsert=True,
                        return_document=ReturnDocument.AFTER,
                        session=session,
                    )

                    # Atomically allocate the next version.
                    document = self.documents.find_one_and_update(
                        {
                            "document_id": document["document_id"],
                            "owner_id": owner_id,
                        },
                        {
                            "$inc": {
                                "next_version": 1
                            },
                            "$set": {
                                "updated_at": now,
                            },
                        },
                        return_document=ReturnDocument.AFTER,
                        session=session,
                    )

                    version = document["next_version"]

                    version_record = {
                        "ingestion_id": ingestion_id,
                        "document_id": document["document_id"],
                        "owner_id": owner_id,

                        "version": version,

                        "paper_title": paper_title,
                        "doi": doi,
                        "identity_key": identity_key,

                        "file_hash": file_hash,
                        "original_filename": metadata[
                            "original_filename"
                        ],
                        "file_extension": metadata[
                            "file_extension"
                        ],
                        "file_size_bytes": metadata[
                            "file_size_bytes"
                        ],

                        "status": "pending",

                        "created_at": now,
                        "updated_at": now,

                        "error": None,
                    }

                    self.versions.insert_one(
                        version_record,
                        session=session
                    )

            return {
                "is_duplicate": False,
                "document_id": document["document_id"],
                "version": version,
                "ingestion_id": ingestion_id,
                "identity_key": identity_key,
                "status": "pending",
            }

        except DuplicateKeyError:

            # Another concurrent upload may have registered
            # identical content.
            duplicate = self.find_duplicate_version(
                file_hash=file_hash,
                owner_id=owner_id
            )

            if duplicate:
                return duplicate

            raise

        except OperationFailure as exc:
            if not self._transactions_unsupported(exc):
                raise

            return self._register_document_version_without_transaction(
                metadata=metadata,
                owner_id=owner_id,
                paper_title=paper_title,
                doi=doi,
                identity_key=identity_key,
                file_hash=file_hash,
                ingestion_id=ingestion_id,
                now=now,
            )

    def _register_document_version_without_transaction(
        self,
        metadata: dict,
        owner_id: str,
        paper_title: str,
        doi: str,
        identity_key: str,
        file_hash: str,
        ingestion_id: str,
        now: datetime,
    ) -> dict:
        """Register using atomic single-document writes on standalone MongoDB."""
        document_query = {
            "owner_id": owner_id,
            "identity_key": identity_key,
        }

        try:
            document = self.documents.find_one_and_update(
                document_query,
                {
                    "$setOnInsert": {
                        "document_id": str(uuid.uuid4()),
                        "owner_id": owner_id,
                        "identity_key": identity_key,
                        "paper_title": paper_title,
                        "doi": doi,
                        "active_version": None,
                        "next_version": 0,
                        "status": "pending",
                        "created_at": now,
                    },
                    "$set": {"updated_at": now},
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except DuplicateKeyError:
            # A concurrent upsert may have created this identity first.
            document = self.documents.find_one(document_query)
            if document is None:
                raise

        document = self.documents.find_one_and_update(
            {
                "document_id": document["document_id"],
                "owner_id": owner_id,
            },
            {
                "$inc": {"next_version": 1},
                "$set": {"updated_at": now},
            },
            return_document=ReturnDocument.AFTER,
        )

        version = document["next_version"]
        version_record = {
            "ingestion_id": ingestion_id,
            "document_id": document["document_id"],
            "owner_id": owner_id,
            "version": version,
            "paper_title": paper_title,
            "doi": doi,
            "identity_key": identity_key,
            "file_hash": file_hash,
            "original_filename": metadata["original_filename"],
            "file_extension": metadata["file_extension"],
            "file_size_bytes": metadata["file_size_bytes"],
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "error": None,
        }

        try:
            self.versions.insert_one(version_record)
        except DuplicateKeyError:
            duplicate = self.find_duplicate_version(
                file_hash=file_hash,
                owner_id=owner_id,
            )
            if duplicate:
                return duplicate
            raise

        return {
            "is_duplicate": False,
            "document_id": document["document_id"],
            "version": version,
            "ingestion_id": ingestion_id,
            "identity_key": identity_key,
            "status": "pending",
        }

    # --------------------------------------------------
    # 7. UPDATE VERSION STATUS
    # --------------------------------------------------

    def update_version_status(
        self,
        document_id: str,
        version: int,
        status: str,
        error: str = None,
        extra_fields: dict = None,
        owner_id: str = None,
    ):

        update_fields = {
            "status": status,
            "error": error,
            "updated_at": datetime.now(timezone.utc),
        }

        if extra_fields:
            update_fields.update(extra_fields)

        query = {
            "document_id": document_id,
            "version": version,
        }

        if owner_id:
            query["owner_id"] = owner_id

        result = self.versions.update_one(
            query,
            {"$set": update_fields}
        )

        if result.matched_count == 0:
            raise ValueError(
                f"Version not found: {document_id}, v{version}"
            )

    # --------------------------------------------------
    # 8. GET DOCUMENT
    # --------------------------------------------------

    def get_document(
        self,
        document_id: str,
        owner_id: str
    ):

        return self.documents.find_one({
            "document_id": document_id,
            "owner_id": owner_id,
        })

    # --------------------------------------------------
    # 9. GET VERSION HISTORY
    # --------------------------------------------------

    def get_document_versions(
        self,
        document_id: str,
        owner_id: str
    ):

        return list(
            self.versions.find({
                "document_id": document_id,
                "owner_id": owner_id,
            }).sort("version", ASCENDING)
        )

    # --------------------------------------------------
    # 10. CLOSE CONNECTION
    # --------------------------------------------------

    def close(self):
        self.client.close()
        
    
    def activate_version(
        self,
        document_id: str,
        owner_id: str,
        version: int,
    ):

        now = datetime.now(timezone.utc)

        try:
            with self.client.start_session() as session:

                with session.start_transaction():

                    # 1. Verify that this version was indexed.
                    version_record = self.versions.find_one(
                        {
                            "document_id": document_id,
                            "owner_id": owner_id,
                            "version": version,
                            "status": "indexed",
                        },
                        session=session,
                    )

                    if not version_record:
                        raise ValueError(
                            "Cannot activate a version "
                            "that has not been indexed."
                        )

                    # 2. Activate only if no newer version
                    # has already become active.
                    updated_document = (
                        self.documents.find_one_and_update(
                            {
                                "document_id": document_id,
                                "owner_id": owner_id,
                                "$or": [
                                    {"active_version": None},
                                    {"active_version": {"$lt": version}},
                                ],
                            },
                            {
                                "$set": {
                                    "active_version": version,
                                    "status": "completed",
                                    "updated_at": now,
                                }
                            },
                            return_document=ReturnDocument.BEFORE,
                            session=session,
                        )
                    )

                    if updated_document is None:

                        current = self.documents.find_one(
                            {
                                "document_id": document_id,
                                "owner_id": owner_id,
                            },
                            session=session,
                        )

                        raise ValueError(
                            "This version cannot replace the "
                            f"currently active version "
                            f"{current.get('active_version') if current else None}."
                        )

                    old_version = updated_document.get(
                        "active_version"
                    )

                    # 3. Mark new version active.
                    self.versions.update_one(
                        {
                            "document_id": document_id,
                            "owner_id": owner_id,
                            "version": version,
                        },
                        {
                            "$set": {
                                "status": "active",
                                "activated_at": now,
                                "updated_at": now,
                            }
                        },
                        session=session,
                    )

                    # 4. Mark previous version superseded.
                    if old_version is not None:

                        self.versions.update_one(
                            {
                                "document_id": document_id,
                                "owner_id": owner_id,
                                "version": old_version,
                            },
                            {
                                "$set": {
                                    "status": "superseded",
                                    "updated_at": now,
                                }
                            },
                            session=session,
                        )

        except OperationFailure as exc:
            if not self._transactions_unsupported(exc):
                raise

            return self._activate_version_without_transaction(
                document_id=document_id,
                owner_id=owner_id,
                version=version,
                now=now,
            )

        return {
            "activated": True,
            "document_id": document_id,
            "active_version": version,
            "previous_version": old_version,
        }

    def _activate_version_without_transaction(
        self,
        document_id: str,
        owner_id: str,
        version: int,
        now: datetime,
    ) -> dict:
        """Activate a version with atomic single-document writes."""
        version_record = self.versions.find_one(
            {
                "document_id": document_id,
                "owner_id": owner_id,
                "version": version,
                "status": "indexed",
            }
        )
        if not version_record:
            raise ValueError(
                "Cannot activate a version that has not been indexed."
            )

        updated_document = self.documents.find_one_and_update(
            {
                "document_id": document_id,
                "owner_id": owner_id,
                "$or": [
                    {"active_version": None},
                    {"active_version": {"$lt": version}},
                ],
            },
            {
                "$set": {
                    "active_version": version,
                    "status": "completed",
                    "updated_at": now,
                }
            },
            return_document=ReturnDocument.BEFORE,
        )
        if updated_document is None:
            current = self.documents.find_one(
                {
                    "document_id": document_id,
                    "owner_id": owner_id,
                }
            )
            raise ValueError(
                "This version cannot replace the "
                f"currently active version "
                f"{current.get('active_version') if current else None}."
            )

        old_version = updated_document.get("active_version")
        try:
            self.versions.update_one(
                {
                    "document_id": document_id,
                    "owner_id": owner_id,
                    "version": version,
                },
                {
                    "$set": {
                        "status": "active",
                        "activated_at": now,
                        "updated_at": now,
                    }
                },
            )

            if old_version is not None:
                self.versions.update_one(
                    {
                        "document_id": document_id,
                        "owner_id": owner_id,
                        "version": old_version,
                    },
                    {
                        "$set": {
                            "status": "superseded",
                            "updated_at": now,
                        }
                    },
                )
        except Exception:
            self.documents.update_one(
                {
                    "document_id": document_id,
                    "owner_id": owner_id,
                    "active_version": version,
                },
                {
                    "$set": {
                        "active_version": old_version,
                        "status": "completed" if old_version is not None else "pending",
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            raise

        return {
            "activated": True,
            "document_id": document_id,
            "active_version": version,
            "previous_version": old_version,
        }

    def get_active_document_versions(self, owner_id: str):
        return list(
            self.documents.find(
                {
                    "owner_id": owner_id,
                    "active_version": {"$ne": None},
                },
                {
                    "_id": 0,
                    "document_id": 1,
                    "active_version": 1,
                }
            )
        )