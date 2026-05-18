"""ChromaDB vector store for policy RAG retrieval."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "moderation_policies"


@lru_cache(maxsize=1)
def _get_embedding_function(model_name: str):
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


class PolicyRAG:
    """Retrieve relevant policy chunks from ChromaDB."""

    def __init__(self) -> None:
        settings = get_settings()
        self.persist_dir = settings.chroma_persist_dir
        self.policy_path = settings.policy_data_path
        self.top_k = settings.rag_top_k
        self._client = None
        self._collection = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import chromadb

        settings = get_settings()
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        embed_fn = _get_embedding_function(settings.embedding_model_name)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_policies(self, force: bool = False) -> int:
        """Load policy chunks from JSON into ChromaDB."""
        self._ensure_client()
        if self._collection.count() > 0 and not force:
            logger.info("policies_already_ingested", count=self._collection.count())
            return self._collection.count()

        path = Path(self.policy_path)
        if not path.exists():
            raise FileNotFoundError(f"Policy data not found: {path}")

        with path.open() as f:
            chunks = json.load(f)

        ids = [c["id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "category": c["category"],
                "risk_level": c.get("risk_level", "MEDIUM"),
                "title": c.get("title", ""),
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("policies_ingested", count=len(chunks))
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """Return top-k policy chunks relevant to the query."""
        self._ensure_client()
        if self._collection.count() == 0:
            self.ingest_policies()

        k = top_k or self.top_k
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
        )

        chunks: list[dict] = []
        if not results["ids"] or not results["ids"][0]:
            return chunks

        for i, doc_id in enumerate(results["ids"][0]):
            chunks.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i]
                    if results.get("distances")
                    else None,
                }
            )
        return chunks
