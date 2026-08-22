import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "docs"
DB_DIR = BASE_DIR / "chromadb_storage"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# Model Configurations
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking Configurations
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
