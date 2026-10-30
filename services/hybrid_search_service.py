
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

import os

load_dotenv()


class HybridSearchService:

    def __init__(
        self,
        collection_name: str = "research_papers",
    ):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be configured."
            )

        self.collection_name = collection_name

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # Must match the models used during indexing.
        self.dense_model = OllamaEmbeddings(
            model="mxbai-embed-large:latest"
        )

        self.sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        paper_slug: str | None = None,
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1 or candidate_k < top_k:
            raise ValueError(
                "Use top_k >= 1 and candidate_k >= top_k."
            )

        # 1. Dense query embedding
        dense_vector = self.dense_model.embed_query(query)

        # 2. Sparse query embedding
        sparse_vector = next(
            self.sparse_model.query_embed([query])
        )

        sparse_query = models.SparseVector(
            indices=sparse_vector.indices.tolist(),
            values=sparse_vector.values.tolist(),
        )

        # 3. Optional metadata filter
        query_filter = None

        if paper_slug:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="paper_slug",
                        match=models.MatchValue(value=paper_slug),
                    )
                ]
            )

        # 4. Dense + sparse candidate retrieval, then RRF
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=candidate_k,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    limit=candidate_k,
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        # 5. Return a clean list for LangGraph
        results = []

        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}

            results.append(
                {
                    "rank": rank,
                    "score": point.score,
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "page_content": payload.get("page_content", ""),
                    "paper_title": payload.get("paper_title", ""),
                    "paper_slug": payload.get("paper_slug", ""),
                    "section": payload.get("section", ""),
                    "chunk_id": payload.get("chunk_id"),
                    "source_file": payload.get("source_file", ""),
                }
            )

        return results
