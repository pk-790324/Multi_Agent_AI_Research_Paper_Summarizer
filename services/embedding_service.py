from pathlib import Path
import json
import uuid

from langchain_ollama import OllamaEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
)


class EmbeddingService:

    @staticmethod
    def index_chunks(
        chunk_file: str,
        collection_name: str = "research_papers",
    ) -> dict:
        """
        Load chunks, generate embeddings, and store them in Qdrant.

        Returns indexing metadata.
        """

        chunk_file = Path(chunk_file)

        if not chunk_file.exists():
            raise FileNotFoundError(chunk_file)

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            raise ValueError("Chunk file is empty.")

        embedding_model = OllamaEmbeddings(
            model="mxbai-embed-large:latest"
        )

        client = QdrantClient(path="qdrant_db")

        first_embedding = embedding_model.embed_query(
            chunks[0]["text"]
        )

        if not client.collection_exists(collection_name):

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=len(first_embedding),
                    distance=Distance.COSINE,
                ),
            )

        points = []

        for index, chunk in enumerate(chunks):

            if index == 0:
                embedding = first_embedding
            else:
                embedding = embedding_model.embed_query(
                    chunk["text"]
                )

            payload = {
                "paper_title": chunk["paper_title"],
                "section": chunk["section"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "char_count": chunk["char_count"],
                "source_file": chunk["source_file"],
                "created_at": chunk["created_at"],
            }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        return {
            "collection_name": collection_name,
            "embedding_model": "mxbai-embed-large:latest",
            "vector_dimension": len(first_embedding),
            "indexed_chunks": len(points),
        }