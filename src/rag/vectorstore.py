"""
Vector store wrapper around ChromaDB with sentence-transformers embeddings.

Handles:
- Chunking documents using LangChain text splitters
- Embedding with all-MiniLM-L6-v2 (local, no API key needed)
- Storing and querying in ChromaDB (persisted to disk)
- Policy management (add, delete, list)
"""

from pathlib import Path
from typing import Optional, Union

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from . import config


class VectorStore:
    """
    ChromaDB-backed vector store for bank policy documents.

    Usage:
        vs = VectorStore()
        vs.add_documents(documents)       # from pdf_loader
        results = vs.query("DPD 45 days collection strategy")
        vs.list_policies()                # see what's loaded
        vs.delete_policy("old_policy.pdf") # remove a policy
    """

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.persist_dir = str(persist_dir or config.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION_NAME
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL

        # Initialize embedding model (downloads on first use, ~80MB)
        self._embedder = SentenceTransformer(self.embedding_model_name)

        # Initialize ChromaDB with disk persistence
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Text splitter for chunking documents
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
        )

    def add_documents(self, documents: list[dict]) -> int:
        """
        Chunk, embed, and store documents in ChromaDB.

        Args:
            documents: List of dicts from pdf_loader.extract_text_from_pdf().
                Each dict has "text" and "metadata" keys.

        Returns:
            Number of chunks added to the store.
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc in documents:
            chunks = self._splitter.split_text(doc["text"])
            for i, chunk in enumerate(chunks):
                # Create a unique ID for each chunk
                source = doc["metadata"]["source"]
                page = doc["metadata"]["page_number"]
                chunk_id = f"{source}_p{page}_c{i}"

                all_chunks.append(chunk)
                all_metadatas.append(
                    {
                        **doc["metadata"],
                        "chunk_index": i,
                    }
                )
                all_ids.append(chunk_id)

        if not all_chunks:
            return 0

        # Embed all chunks at once (batch is faster)
        embeddings = self._embedder.encode(all_chunks).tolist()

        # Upsert so re-ingesting the same PDF overwrites instead of duplicating
        self._collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
        )

        return len(all_chunks)

    def query(self, query_text: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Query the vector store for relevant policy chunks.

        Args:
            query_text: Natural language query
            top_k: Number of results to return (default: config.TOP_K)

        Returns:
            List of dicts, each with:
                - "text": the chunk content
                - "metadata": source info (source, page_number, etc.)
                - "score": cosine similarity (0 to 1, higher = more relevant)
        """
        top_k = top_k or config.TOP_K

        # Guard against querying an empty store
        current_count = self._collection.count()
        if current_count == 0:
            return []

        query_embedding = self._embedder.encode([query_text]).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, current_count),
        )

        if not results["documents"][0]:
            return []

        return [
            {
                "text": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),  # Convert cosine distance to similarity
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def delete_policy(self, source_name: str) -> int:
        """
        Delete all chunks belonging to a specific policy document.

        Useful when the admin uploads a new version of an existing policy.
        Call this before re-ingesting the updated PDF.

        Args:
            source_name: The PDF filename (e.g., "collection_strategy_policy.pdf")

        Returns:
            Number of chunks deleted.
        """
        results = self._collection.get(where={"source": source_name})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def list_policies(self) -> list[str]:
        """List all unique policy document names currently in the store."""
        all_data = self._collection.get()
        if not all_data["metadatas"]:
            return []
        sources = set(m["source"] for m in all_data["metadatas"])
        return sorted(sources)

    def clear(self) -> None:
        """Delete the entire collection and recreate it empty."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Total number of chunks in the vector store."""
        return self._collection.count()
