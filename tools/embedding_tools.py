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

def load_latest_chunk_file():

    chunk_dir = Path("chunks")

    if not chunk_dir.exists():
        raise FileNotFoundError("chunks folder not found.")

    json_files = list(chunk_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError("No chunk json found.")

    latest = max(
        json_files,
        key=lambda f: f.stat().st_mtime,
    )

    with open(latest, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return latest, chunks


@tool
def index_latest_chunks() -> str:
    """
    Load the latest chunk JSON, create embeddings using Ollama,
    and store them in Qdrant.
    """

    latest_file, chunks = load_latest_chunk_file()

    embedding_model = OllamaEmbeddings(
        model="mxbai-embed-large:latest"
    )
    db_path = Path("qdrant_db")
    client = QdrantClient(
    path=str(db_path)
)

    collection_name = "research_papers"

    # ---------- Create one embedding ----------

    sample = embedding_model.embed_query("hello")

    if not client.collection_exists(collection_name):

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=len(sample),
                distance=Distance.COSINE,
            ),
        )

    points = []

    for chunk in chunks:

        embedding_text = f"""
Paper Title:
{chunk['paper_title']}

Section:
{chunk['section']}

Content:
{chunk['text']}
"""

        embedding = embedding_model.embed_query(
            embedding_text
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

    return (
        f"Indexed {len(points)} chunks "
        f"from {latest_file.name}."
    )
    
    