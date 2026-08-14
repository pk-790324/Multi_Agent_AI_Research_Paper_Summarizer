import os

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


load_dotenv()


COLLECTION_NAME = "research_papers"


client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


embedding_model = OllamaEmbeddings(
    model="mxbai-embed-large:latest"
)


def detect_section(query: str):

    query = query.lower().strip()

    section_keywords = {
        "abstract": "Abstract",

        "introduction": "Introduction",

        "methodology": "Methodology",
        "method": "Methodology",
        "methods": "Methodology",

        "experiment": "Experiments",
        "experiments": "Experiments",
        "experimental": "Experiments",

        "evaluation": "Evaluation",

        "result": "Results",
        "results": "Results",

        "discussion": "Discussion",

        "conclusion": "Conclusion",
        "conclusions": "Conclusion",

        "background": "Background",

        "related work": "Related Work",

        "literature review": "Literature Review",

        "limitations": "Limitations",

        "future work": "Future Work",

        "references": "References",
    }

    # Check longer phrases first
    # e.g. "related work" before "work"
    for keyword, section in sorted(
        section_keywords.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        if keyword in query:
            return section

    return None


def retriever_agent(state):

    query = state["user_query"]

    print("\n" + "=" * 60)
    print("RETRIEVER AGENT")
    print("=" * 60)

    print("Query:", query)

    section = detect_section(query)

    print("Detected section:", section)

    # ---------------------------------------
    # Create query embedding
    # ---------------------------------------

    query_vector = embedding_model.embed_query(query)

    # ---------------------------------------
    # Build filter
    # ---------------------------------------

    qdrant_filter = None

    if section:

        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="section",
                    match=MatchValue(
                        value=section
                    ),
                )
            ]
        )

    # ---------------------------------------
    # Qdrant search
    # ---------------------------------------

    search_result = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_vector,

        using="dense",

        query_filter=qdrant_filter,

        limit=3,

        with_payload=True,
    )

    # ---------------------------------------
    # Convert to LangChain Documents
    # ---------------------------------------

    documents = []

    for point in search_result.points:

        payload = point.payload

        documents.append(
            Document(
                page_content=payload["text"],

                metadata={
                    "paper_title":
                        payload["paper_title"],

                    "section":
                        payload["section"],

                    "chunk_id":
                        payload["chunk_id"],

                    "source_file":
                        payload["source_file"],

                    "created_at":
                        payload["created_at"],
                },
            )
        )

    # ---------------------------------------
    # Debug
    # ---------------------------------------

    print(
        "Documents retrieved:",
        len(documents)
    )

    for i, doc in enumerate(documents):

        print(
            f"\n--- Document {i + 1} ---"
        )

        print(
            "Section:",
            doc.metadata["section"]
        )

        print(
            "Chunk:",
            doc.metadata["chunk_id"]
        )

        print(
            "Content:",
            doc.page_content[:500]
        )

    return {
        "retrieved_docs": documents,

        "retrieval_count":
            state.get(
                "retrieval_count",
                0
            ) + 1,
    }