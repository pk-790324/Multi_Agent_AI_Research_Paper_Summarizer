from services.embedding_service import EmbeddingService

result = EmbeddingService.index_chunks(
    chunk_file="chunks/rules_of_ml_chunks.json",
    collection_name="research_papers",
    batch_size=16,
)

print(result)