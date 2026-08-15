from pathlib import Path
import json
import uuid
import os
import re

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


load_dotenv()


def normalize_paper_slug(title: str) -> str:
    """Normalize the full paper title into a stable slug used for metadata filtering.

    NOTE: We intentionally do NOT strip the subtitle (after ':') so that slugs
    are consistent across embedding and retrieval.  Parenthetical suffixes are
    still stripped because they are version-specific (e.g. '(v2)').
    """
    if not title:
        return ""
    # Remove only parenthetical suffixes; keep subtitles to preserve uniqueness
    base_title = title.split("(", 1)[0].strip()
    return re.sub(r"[^a-z0-9]+", "-", base_title.lower()).strip("-")


class EmbeddingService:

    @staticmethod
    def index_chunks(
        chunk_file: str,
        collection_name: str = "research_papers",
    ) -> dict:
        """
        Generate embeddings from chunk JSON and upload them
        to an existing Qdrant Cloud collection.

        The collection must already exist.
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

        if not chunks:
            raise ValueError("Chunk file is empty.")

        # -----------------------------------------
        # 2. Qdrant Cloud configuration
        # -----------------------------------------

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise ValueError(
                "QDRANT_URL is not configured."
            )

        if not qdrant_api_key:
            raise ValueError(
                "QDRANT_API_KEY is not configured."
            )

        # -----------------------------------------
        # 3. Connect to Qdrant Cloud
        # -----------------------------------------

        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # -----------------------------------------
        # 4. Collection MUST already exist
        # -----------------------------------------

        if not client.collection_exists(collection_name):
            raise ValueError(
                f"Qdrant collection '{collection_name}' "
                "does not exist. "
                "Create it manually in Qdrant Cloud."
            )

        # -----------------------------------------
        # 5. Embedding model
        # -----------------------------------------

        embedding_model = OllamaEmbeddings(
            model="mxbai-embed-large:latest"
        )

        # -----------------------------------------
        # 6. Ensure section-based indexing exists
        # -----------------------------------------

        for field_name in ["section", "paper_title", "paper_slug", "chunk_id"]:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema="keyword",
                )
            except Exception:
                # Index may already exist; ignore duplicate-index errors.
                pass

        # -----------------------------------------
        # 7. Generate embeddings
        # -----------------------------------------

        points = []

        for chunk in chunks:

            text = chunk["text"]
            paper_title = chunk["paper_title"]
            paper_slug = normalize_paper_slug(paper_title)

            embedding = embedding_model.embed_query(text)

            # -------------------------------------
            # 8. Flat payload for filterable metadata
            # -------------------------------------

            payload = {
                "page_content": text,
                "paper_title": paper_title,
                "paper_slug": paper_slug,
                "section": chunk["section"],
                "chunk_id": chunk["chunk_id"],
                "char_count": chunk["char_count"],
                "source_file": chunk["source_file"],
                "created_at": chunk["created_at"],
            }

            # -------------------------------------
            # 9. Create Qdrant point
            # -------------------------------------

            point = PointStruct(
                id=str(uuid.uuid4()),

                # Qdrant Cloud vector name = dense
                vector={
                    "dense": embedding
                },

                payload=payload,
            )

            points.append(point)

        # -----------------------------------------
        # 9. Upload to existing Qdrant collection
        # -----------------------------------------

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        # -----------------------------------------
        # 10. Return metadata
        # -----------------------------------------

        return {
            "collection_name": collection_name,
            "embedding_model": "mxbai-embed-large:latest",
            "vector_name": "dense",
            "vector_dimension": len(points[0].vector["dense"]),
            "indexed_chunks": len(points),
        }