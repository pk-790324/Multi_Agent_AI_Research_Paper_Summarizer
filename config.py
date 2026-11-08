from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


#===========================================
# config for data ingestin
#===========================================

MAX_FILE_SIZE_MB=10
UPLOAD_DIR="./data/uploaded_papers"
PROJECT_DIR = Path(__file__).resolve().parent
UPLOADED_DIR = PROJECT_DIR / "data" / "uploaded_papers"
STAGES = ["Upload", "Parsing", "Chunking", "Embeddings"]
SUPPORTED_FILES = [
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".txt",
    ".md",
    ".rtf",
]

PREVIEW_CHAR_LIMIT = 20000
CHUNK_PREVIEW_LIMIT = 20


#===========================================
# config for multi agent 
#===========================================

MAX_AGENT_ATTEMPTS = 5

AGENT_NAMES = [
    "Orchestrator",
    "Retriever",
    "Research Analyst",
    "Critical Analyst",
    "Citation Agent",
    "Synthesis Writer",
]





QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")

EMBEDDING_MODEL = "mxbai-embed-large:latest"