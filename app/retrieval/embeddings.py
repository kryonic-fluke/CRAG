from typing import List
import hashlib
import os
import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.config import settings


class DeterministicHashEmbeddings(Embeddings):
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
    provider = settings.llm_provider.lower().strip()

    if provider == "gemini":
        gemini_key = (
            settings.gemini_api_key
            or settings.google_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if gemini_key and "mock" not in gemini_key.lower():
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                return GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=gemini_key,
                )
            except Exception:
                pass

    if provider == "openai":
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
