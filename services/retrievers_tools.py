from pathlib import Path

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient


COLLECTION_NAME = "research_papers"


def get_qdrant_client():

    db_path = Path("qdrant_db")
    db_path.mkdir(exist_ok=True)

    return QdrantClient(path=str(db_path))




from state.state import ResearchPaperState,QAState


@tool
def retrieve_documents(
    state: QAState,
) -> QAState:
    """
    Retrieve the most relevant document chunks from Qdrant
    using the user's query and update the shared state.
    """

    db_path = Path("qdrant_db")

    client = QdrantClient(
        path=str(db_path)
    )

    embedding_model = OllamaEmbeddings(
        model="mxbai-embed-large:latest"
    )

    query_vector = embedding_model.embed_query(
        state["question"]
    )

    collection_name = state["collection_name"]

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5,
    ).points

    retrieved_chunks = []

    for point in results:

        payload = point.payload

        retrieved_chunks.append(
            {
                "paper_title": payload["paper_title"],
                "section": payload["section"],
                "chunk_id": payload["chunk_id"],
                "score": float(point.score),
                "text": payload["text"],
                "source_file": payload["source_file"],
            }
        )

    return {
        **state,
        "retrieved_context": retrieved_chunks
    }