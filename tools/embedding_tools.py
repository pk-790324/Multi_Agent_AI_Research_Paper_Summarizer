from pathlib import Path
import json
import uuid

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
)



from state.state import ResearchPaperState


@tool
def index_chunks(
    state: ResearchPaperState,
) -> ResearchPaperState:
    """
    Load the chunk JSON from the shared state,
    generate embeddings using Ollama,
    and store them in Qdrant.
    """

    chunk_file = Path(
        state["artifacts"]["chunk_file"]
    )

    if not chunk_file.exists():
        raise FileNotFoundError(
            f"{chunk_file} does not exist."
        )

    with open(
        chunk_file,
        "r",
        encoding="utf-8",
    ) as f:
        chunks = json.load(f)

    embedding_model = OllamaEmbeddings(
        model="mxbai-embed-large:latest"
    )

    db_path = Path("qdrant_db")

    client = QdrantClient(
        path=str(db_path)
    )

    collection_name = "research_papers"

    # Determine embedding dimension
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
        **state,
        "collection_name": collection_name,
        "embedding": {
            "embedding_model": "mxbai-embed-large:latest",
            "vector_dimension": len(first_embedding),
            "indexed_chunks": len(points),
        },
    }
    
    