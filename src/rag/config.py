"""
Centralized configuration for the RAG engine.

All settings can be overridden via environment variables or a .env file.
The backend teammate can also import and modify these before calling the RAG API.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICIES_DIR = PROJECT_ROOT / "policies"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
DATA_DIR = PROJECT_ROOT / "data"

# --- Embedding Model ---
# all-MiniLM-L6-v2: local, free, 384 dims, good quality for prototyping
# Switch to "BAAI/bge-small-en-v1.5" for better quality (still local & free)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- LLM ---
# Supported providers: "google", "openai", "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))

# --- ChromaDB ---
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "bank_policies")

# --- Response Cache ---
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "100"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour

# --- Document Upload Limits ---
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
