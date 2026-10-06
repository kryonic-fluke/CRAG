from pathlib import Path
from typing import List, Optional, Union
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.retrieval.embeddings import get_embeddings_model
from app.schemas.document import DocumentChunk, DocumentMetadata
from app.schemas.retrieval import SearchResult


class ChromaVectorStore:
    """
    Production ChromaDB vector store wrapper for Dense Semantic Retrieval.

    Responsibilities:
    - Stores DocumentChunks with deterministic IDs and rich metadata.
    - Computes dense vector embeddings using the configured Embeddings provider.
    - Executes similarity search returning validated SearchResult Pydantic objects.
    """

    def __init__(
        self,
        collection_name: str = "crag_knowledge_base",
        persist_dir: Optional[Union[str, Path]] = None,
        embeddings: Optional[Embeddings] = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir or settings.chroma_persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = embeddings or get_embeddings_model()

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Indexes a batch of Pydantic DocumentChunk objects into ChromaDB.
        """
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.content)

            meta_dict = {
                "source": chunk.metadata.source,
                "chunk_index": chunk.metadata.chunk_index,
                "char_count": chunk.metadata.char_count,
                "created_at": chunk.metadata.created_at.isoformat(),
            }

            for k, v in chunk.metadata.extra.items():
                meta_dict[f"extra_{k}"] = str(v)
            metadatas.append(meta_dict)

        embeddings_vectors = self.embeddings.embed_documents(documents)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings_vectors,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def similarity_search(self, query: str, k: int = 4) -> List[SearchResult]:
        """
        Executes dense semantic similarity search against the indexed collection.
        Returns top-k results formatted as SearchResult Pydantic models.
        """
        if self.collection.count() == 0:
            return []

        query_vector = self.embeddings.embed_query(query)

        raw_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        search_results: List[SearchResult] = []
        doc_ids = raw_results["ids"][0]
        docs = raw_results["documents"][0]
        metas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        for rank_idx, (cid, doc_text, meta, dist) in enumerate(
            zip(doc_ids, docs, metas, distances), start=1
        ):
            similarity_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            metadata_obj = DocumentMetadata(
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", 0),
                char_count=meta.get("char_count", len(doc_text)),
                extra={
                    k.replace("extra_", ""): str(v)
                    for k, v in meta.items()
                    if k.startswith("extra_")
                },
            )

            result = SearchResult(
                chunk_id=cid,
                content=doc_text,
                metadata=metadata_obj,
                score=round(similarity_score, 4),
                retrieval_method="dense",
                rank=rank_idx,
            )
            search_results.append(result)

        return search_results

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """Deletes and recreates the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
