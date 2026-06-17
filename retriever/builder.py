from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings

from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RetrieverBuilder:
    def __init__(self):
        """Initialize embedding model."""

        self.embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:latest",
            base_url="http://localhost:11434"
        )

    def build_hybrid_retriever(self, docs):
        """
        Build a hybrid retriever using BM25 and Chroma vector search.
        """
        try:
            # Create Chroma vector store
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=settings.CHROMA_DB_PATH
            )

            logger.info("Vector store created successfully.")

            # BM25 retriever
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = settings.BM25_SEARCH_K

            logger.info("BM25 retriever created successfully.")

            # Vector retriever
            vector_retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": settings.VECTOR_SEARCH_K
                }
            )

            logger.info("Vector retriever created successfully.")

            # Hybrid retriever
            hybrid_retriever = EnsembleRetriever(
                retrievers=[
                    bm25,
                    vector_retriever
                ],
                weights=settings.HYBRID_RETRIEVER_WEIGHTS
            )

            logger.info("Hybrid retriever created successfully.")

            return hybrid_retriever

        except Exception as e:
            logger.error(f"Failed to build hybrid retriever: {e}")
            raise