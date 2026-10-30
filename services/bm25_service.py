
import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Service:
    """
    Keyword-based retrieval over existing research paper chunks.

    The BM25 index is persisted as JSON. No dense embeddings
    or existing Qdrant vectors are modified.
    """

    def __init__(self, index_dir: str = "data/bm25_index"):
        self.index_dir = Path(index_dir)
        self.index_file = self.index_dir / "chunks.json"

        self.chunks: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Normalize text while preserving technical terms."""
        return re.findall(r"[\w.-]+", text.lower())

    @staticmethod
    def load_chunks(chunk_file: str) -> list[dict[str, Any]]:
        """Load chunks from an existing JSON chunk file."""
        with open(chunk_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Support either a list of chunks or {"chunks": [...]}.
        if isinstance(data, dict):
            data = data.get("chunks", [])

        if not isinstance(data, list):
            raise ValueError(
                "Expected a list of chunks or a JSON object "
                "containing a 'chunks' list."
            )

        chunks = [
            chunk
            for chunk in data
            if isinstance(chunk, dict)
            and isinstance(chunk.get("text"), str)
            and chunk["text"].strip()
        ]

        if not chunks:
            raise ValueError("No valid text chunks found.")

        return chunks

    def build_index(self, chunk_file: str) -> dict[str, Any]:
        """Build and persist a BM25 index from existing chunks."""
        self.chunks = self.load_chunks(chunk_file)

        self.tokenized_corpus = [
            self.tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save chunk text and metadata, not Python pickle objects.
        with open(self.index_file, "w", encoding="utf-8") as file:
            json.dump(
                self.chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "status": "success",
            "indexed_chunks": len(self.chunks),
            "index_path": str(self.index_file),
        }

    def load_index(self) -> None:
        """Rebuild the in-memory BM25 index from saved chunks."""
        if not self.index_file.exists():
            raise FileNotFoundError(
                f"BM25 index not found: {self.index_file}. "
                "Run build_index() first."
            )

        with open(self.index_file, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        self.tokenized_corpus = [
            self.tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top BM25-ranked chunks for a query."""
        if self.bm25 is None:
            self.load_index()

        if not query.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Highest BM25 score first.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            # Exclude chunks with no lexical match.
            if score <= 0:
                continue

            results.append({
                **self.chunks[index],
                "bm25_score": score,
            })

            if len(results) >= top_k:
                break

        return results
