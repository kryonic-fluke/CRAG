from typing import List
import hashlib
import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.config import settings


class DeterministicHashEmbeddings(Embeddings):
    """
    Lightweight, deterministic embedding generator using SHA-256 + pseudo-random projection.

    Why this exists:
    - Guarantees 100% offline, lightning-fast testing and CI execution with 0 network calls.
    - Produces fixed 384-dimensional unit-norm float vectors.
    - Identical texts always produce identical vectors.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed(self, text: str) -> List[float]:

        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class ChromaDefaultEmbeddingsWrapper(Embeddings):
    """
    Wraps Chroma's built-in ONNX all-MiniLM-L6-v2 embedding model into LangChain Embeddings.
    Free, local, runs on CPU/GPU without external API keys.
    """

    def __init__(self) -> None:
        import chromadb.utils.embedding_functions as ef

        self._ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._ef(texts)

        return [[float(x) for x in e] for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._ef([text])
        return [float(x) for x in embeddings[0]]


def get_embeddings_model() -> Embeddings:
    """
    Factory that selects the best available embedding model:
    1. If a valid OpenAI API key is present -> OpenAIEmbeddings
    2. Else -> Chroma's local all-MiniLM-L6-v2 ONNX embeddings
    3. Fallback -> DeterministicHashEmbeddings
    """
    key = settings.openai_api_key
    if key and key.startswith("sk-") and "mock" not in key.lower():
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(api_key=key, model="text-embedding-3-small")
        except Exception:
            pass

    try:
        return ChromaDefaultEmbeddingsWrapper()
    except Exception:
        return DeterministicHashEmbeddings()
