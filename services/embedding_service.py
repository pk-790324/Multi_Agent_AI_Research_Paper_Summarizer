
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
    def index_chunks(
        chunk_file: str,
        collection_name: str = "research_papers",
        batch_size: int = 16,
    ) -> dict:
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

        for field_name in [
            "section",
            "paper_title",
            "page_number",
            "paper_slug",
            "chunk_id",
        ]:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema="keyword",
                    wait=True,
                )
            except Exception:
                # An index may already exist.
                pass

        # -----------------------------------------
        # 7. Generate and upload vectors in batches
        # -----------------------------------------

        total_chunks = len(chunks)
        indexed_chunks = 0
        vector_dimension = None

        for start in range(0, total_chunks, batch_size):

            batch = chunks[start:start + batch_size]

            texts = [chunk["text"] for chunk in batch]

            # Dense embeddings for all texts in this batch
            dense_vectors = dense_model.embed_documents(texts)

            # Sparse embeddings for all texts in this batch
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
                        f"Expected 1024 dense dimensions, got {len(dense)}."
                    )

                vector_dimension = len(dense)

                paper_title = chunk.get("paper_title", "")
                paper_slug = normalize_paper_slug(paper_title)

                # ---------------------------------
                # 8. Flat metadata payload
                # ---------------------------------

                payload = {
                    "page_content": chunk["text"],
                    "paper_title": paper_title,
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "paper_slug": paper_slug,
                    "section": chunk.get("section", ""),
                    "chunk_id": chunk.get("chunk_id"),
                    "char_count": chunk.get(
                        "char_count", len(chunk["text"])
                    ),
                    "source_file": chunk.get("source_file", ""),
                    "created_at": chunk.get("created_at", ""),
                }

                # ---------------------------------
                # 9. Store dense + sparse vectors
                # ---------------------------------

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

            # Upload the current batch
            client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )

            indexed_chunks += len(points)

            print(
                f"Indexed {indexed_chunks}/{total_chunks} chunks"
            )

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
            "points_in_collection": final_count,
        }




