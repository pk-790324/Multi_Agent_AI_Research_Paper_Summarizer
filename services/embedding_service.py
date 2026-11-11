
from pathlib import Path
import json
import uuid
import os
import re

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from fastembed import SparseTextEmbedding

from qdrant_client import QdrantClient, models

load_dotenv()


def normalize_paper_slug(title: str) -> str:
    """Create a stable slug while preserving the subtitle."""
    if not title:
        return ""

    base_title = title.split("(", 1)[0].strip()
    return re.sub(r"[^a-z0-9]+", "-", base_title.lower()).strip("-")


class EmbeddingService:
    
    @staticmethod
    def create_payload_indexes(
        client: QdrantClient,
        collection_name: str,
    ) -> None:
        """
        Create payload indexes required for metadata filtering.
        """

        payload_indexes = {
            "section": models.PayloadSchemaType.KEYWORD,
            "paper_title": models.PayloadSchemaType.KEYWORD,
            "paper_slug": models.PayloadSchemaType.KEYWORD,
            "source_file": models.PayloadSchemaType.KEYWORD,

            # Document ownership and version management
            "document_id": models.PayloadSchemaType.KEYWORD,
            "owner_id": models.PayloadSchemaType.KEYWORD,

            "version": models.PayloadSchemaType.INTEGER,

            # Chunk and page metadata
            "chunk_id": models.PayloadSchemaType.INTEGER,
            "page_start": models.PayloadSchemaType.INTEGER,
            "page_end": models.PayloadSchemaType.INTEGER,
        }

        for field_name, field_type in payload_indexes.items():

            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type,
                    wait=True,
                )

                print(f"Payload index created/verified: {field_name}")

            except Exception as exc:
                print(
                    f"Payload index check failed for "
                    f"{field_name}: {exc}"
                )

    @staticmethod
    def index_chunks(
        chunk_file: str,
        document_id: str,
        owner_id: str,
        version: int,
        ingestion_id: str,
        collection_name: str = "research_papers",
        batch_size: int = 16,) -> dict:
        """
        Generate dense and sparse embeddings for each chunk
        and upload both vectors to an existing Qdrant collection.
        """

        # -----------------------------------------
        # 1. Load chunks
        # -----------------------------------------

        chunk_file = Path(chunk_file)

        if not chunk_file.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {chunk_file}"
            )

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Support either a JSON list or {"chunks": [...]}
        if isinstance(chunks, dict):
            chunks = chunks.get("chunks", [])

        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Chunk file is empty or invalid.")

        chunks = [
            chunk for chunk in chunks
            if isinstance(chunk, dict)
            and isinstance(chunk.get("text"), str)
            and chunk["text"].strip()
        ]

        if not chunks:
            raise ValueError("No valid text chunks found.")

        # -----------------------------------------
        # 2. Qdrant Cloud configuration
        # -----------------------------------------

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise ValueError("QDRANT_URL is not configured.")

        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY is not configured.")

        # -----------------------------------------
        # 3. Connect to Qdrant Cloud
        # -----------------------------------------

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        if not client.collection_exists(collection_name):
            raise ValueError(
                f"Qdrant collection '{collection_name}' does not exist."
            )

        # -----------------------------------------
        # 4. Validate vector configuration
        # -----------------------------------------

        collection = client.get_collection(collection_name)

        dense_config = collection.config.params.vectors
        sparse_config = collection.config.params.sparse_vectors

        if not isinstance(dense_config, dict) or "dense" not in dense_config:
            raise ValueError(
                "Collection must have a named dense vector called 'dense'."
            )

        if not sparse_config or "sparse" not in sparse_config:
            raise ValueError(
                "Collection must have a sparse vector called 'sparse'."
            )

       
        # -----------------------------------------
        # 5. Initialize embedding models
        # -----------------------------------------

        # Dense semantic embeddings: Ollama
        dense_model = OllamaEmbeddings(
            model="mxbai-embed-large:latest"
        )

        # Sparse lexical embeddings: BM25
        sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

        # -----------------------------------------
        # 6. Create metadata indexes
        # -----------------------------------------

        EmbeddingService.create_payload_indexes(
            
            client=client,
            collection_name=collection_name,)

        # -----------------------------------------
        # 7. Generate and upload vectors in batches
        # -----------------------------------------



        total_chunks = len(chunks)
        indexed_chunks = 0
        vector_dimension = None

        try:

            for start in range(0, total_chunks, batch_size):

                batch = chunks[start:start + batch_size]

                texts = [chunk["text"] for chunk in batch]

                # Generate dense embeddings
                dense_vectors = dense_model.embed_documents(texts)

                # Generate sparse embeddings
                sparse_vectors = list(sparse_model.embed(texts))

                if len(dense_vectors) != len(batch):
                    raise ValueError("Dense embedding count mismatch.")

                if len(sparse_vectors) != len(batch):
                    raise ValueError("Sparse embedding count mismatch.")

                points = []

                for chunk, dense, sparse in zip(
                    batch, dense_vectors, sparse_vectors
                ):

                    if len(dense) != 1024:
                        raise ValueError(
                            f"Expected 1024 dimensions, got {len(dense)}."
                        )

                    vector_dimension = len(dense)

                    paper_title = chunk.get("paper_title", "")
                    paper_slug = normalize_paper_slug(paper_title)

                    payload = {
                        "page_content": chunk["text"],

                        # Document identity
                        "document_id": document_id,
                        "owner_id": owner_id,
                        "version": version,

                        # Paper metadata
                        "paper_title": paper_title,
                        "paper_slug": paper_slug,
                        "source_file": chunk.get("source_file", ""),

                        # Section and chunk metadata
                        "section": chunk.get("section", ""),
                        "chunk_id": chunk.get("chunk_id"),
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),

                        # Additional metadata
                        "char_count": chunk.get(
                            "char_count", len(chunk["text"])
                        ),
                        "created_at": chunk.get("created_at", ""),
                    }

                    point = models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": dense,
                            "sparse": models.SparseVector(
                                indices=sparse.indices.tolist(),
                                values=sparse.values.tolist(),
                            ),
                        },
                        payload=payload,
                    )

                    points.append(point)

                # Upload batch
                client.upsert(
                    collection_name=collection_name,
                    points=points,
                    wait=True,
                )

                indexed_chunks += len(points)

                print(
                    f"Indexed {indexed_chunks}/{total_chunks} chunks"
                )
                verification = EmbeddingService.verify_version(
                    document_id=document_id,
                    owner_id=owner_id,
                    version=version,
                    expected_chunks=indexed_chunks,
                    collection_name=collection_name,
                )

                if not verification["verified"]:
                    raise RuntimeError(
                        f"Qdrant verification failed: {verification}"
                    )

        except Exception:

            print(
                f"Indexing failed for document {document_id}, "
                f"version {version}. Cleaning partial Qdrant points."
            )

            EmbeddingService.cleanup_failed_version(
                document_id=document_id,
                owner_id=owner_id,
                version=version,
                collection_name=collection_name,
            )

            raise

        # -----------------------------------------
        # 10. Verify final point count
        # -----------------------------------------

        final_count = client.count(
            collection_name=collection_name,
            exact=True,
        ).count

        return {
            "status": "success",
            "collection_name": collection_name,
            "dense_model": "mxbai-embed-large:latest",
            "sparse_model": "Qdrant/bm25",
            "vector_names": ["dense", "sparse"],
            "vector_dimension": vector_dimension,
            "indexed_chunks": indexed_chunks,
            "verified_chunks": indexed_chunks,
            "points_in_collection": final_count,
        }

    @staticmethod
    def _get_qdrant_client():

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "Qdrant credentials are missing."
            )

        return QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    @staticmethod
    def build_version_filter(
        document_id: str,
        owner_id: str,
        version: int,
    ):

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id
                    ),
                ),
                models.FieldCondition(
                    key="owner_id",
                    match=models.MatchValue(
                        value=owner_id
                    ),
                ),
                models.FieldCondition(
                    key="version",
                    match=models.MatchValue(
                        value=version
                    ),
                ),
            ]
        )

    @staticmethod
    def verify_version(
        document_id: str,
        owner_id: str,
        version: int,
        expected_chunks: int,
        collection_name: str = "research_papers",
    ) -> dict:

        client = EmbeddingService._get_qdrant_client()

        try:

            version_filter = (
                EmbeddingService.build_version_filter(
                    document_id,
                    owner_id,
                    version,
                )
            )

            result = client.count(
                collection_name=collection_name,
                count_filter=version_filter,
                exact=True,
            )

            actual_chunks = result.count

            return {
                "verified": actual_chunks == expected_chunks,
                "expected_chunks": expected_chunks,
                "actual_chunks": actual_chunks,
            }

        finally:
            client.close()

    @staticmethod
    def delete_version(
        document_id: str,
        owner_id: str,
        version: int,
        collection_name: str = "research_papers",
    ) -> dict:

        client = EmbeddingService._get_qdrant_client()

        try:

            version_filter = (
                EmbeddingService.build_version_filter(
                    document_id,
                    owner_id,
                    version,
                )
            )

            # Delete only the specified document version.
            client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=version_filter
                ),
                wait=True,
            )

            remaining = client.count(
                collection_name=collection_name,
                count_filter=version_filter,
                exact=True,
            ).count

            if remaining != 0:
                raise RuntimeError(
                    f"Version deletion incomplete: "
                    f"{remaining} points remain."
                )

            return {
                "status": "deleted",
                "document_id": document_id,
                "version": version,
                "remaining_points": remaining,
            }

        finally:
            client.close()

    @staticmethod
    def cleanup_failed_version(
        document_id: str,
        owner_id: str,
        version: int,
        collection_name: str = "research_papers",
    ):

        return EmbeddingService.delete_version(
            document_id=document_id,
            owner_id=owner_id,
            version=version,
            collection_name=collection_name,
        )

    @staticmethod
    def build_active_versions_filter(
        owner_id: str,
        active_documents: list[dict],
    ):

        conditions = []

        for document in active_documents:

            conditions.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(
                                value=document["document_id"]
                            ),
                        ),
                        models.FieldCondition(
                            key="version",
                            match=models.MatchValue(
                                value=document["active_version"]
                            ),
                        ),
                    ]
                )
            )

        if not conditions:
            return None

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="owner_id",
                    match=models.MatchValue(value=owner_id),
                )
            ],
            should=conditions,
        )

