
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv


load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not qdrant_url:
    raise ValueError("QDRANT_URL is not configured.")

if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY is not configured.")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

collection_name = "research_papers"

info = client.get_collection(collection_name)

print("Collection:", collection_name)
print("Status:", info.status)
print("Points:", info.points_count)
print("Vectors configuration:")
print(info.config.params.vectors)

print("Sparse vectors configuration:")
print(info.config.params.sparse_vectors)
