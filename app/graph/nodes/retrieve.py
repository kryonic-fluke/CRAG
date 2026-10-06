from typing import Any, Dict, Optional
from app.graph.state import GraphState
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.bm25_retriever import BM25KeywordRetriever


_default_retriever: Optional[HybridRetriever] = None


def get_default_retriever() -> HybridRetriever:
    global _default_retriever
    if _default_retriever is None:
        vs = ChromaVectorStore()
        bm = BM25KeywordRetriever()
        _default_retriever = HybridRetriever(vector_store=vs, bm25_retriever=bm)
    return _default_retriever


def set_default_retriever(retriever: HybridRetriever) -> None:
    global _default_retriever
    _default_retriever = retriever


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    retriever = get_default_retriever()
    response = retriever.retrieve(query=question)
    return {"documents": response.results}
