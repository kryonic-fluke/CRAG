import pytest

from app.schemas.document import DocumentChunk, DocumentMetadata
from app.schemas.retrieval import SearchResult
from app.graph.state import GraphState
from app.retrieval.web_search import TavilySearchWrapper
from app.graph.chains.generation import get_generation_chain
from app.graph.nodes.web_search import web_search_node
from app.graph.nodes.generate import generate_node
from app.graph.nodes.retrieve import set_default_retriever
from app.graph.workflow import create_crag_graph
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.bm25_retriever import BM25KeywordRetriever
from app.retrieval.embeddings import DeterministicHashEmbeddings


@pytest.fixture
def policy_chunks():
    return [
        DocumentChunk(
            id="c1",
            content="Acme Corp gives an equipment stipend of $1,500 USD for home office setup.",
            metadata=DocumentMetadata(
                source="policy.txt", chunk_index=0, char_count=73
            ),
        ),
        DocumentChunk(
            id="c2",
            content="Employees have an unlimited PTO policy with a mandatory minimum of 20 days off.",
            metadata=DocumentMetadata(
                source="policy.txt", chunk_index=1, char_count=80
            ),
        ),
    ]


def test_tavily_search_wrapper():
    searcher = TavilySearchWrapper()
    results = searcher.search(query="Python 3.12 release date", max_results=2)

    assert len(results) > 0
    assert results[0].retrieval_method == "web"
    assert len(results[0].content) > 0
    assert results[0].metadata.source.startswith("http")


def test_generation_chain():
    chain = get_generation_chain()
    answer = chain.invoke(
        {
            "context": "Acme Corp provides $1,500 equipment stipend for new hires.",
            "question": "How much is the equipment stipend?",
        }
    )

    assert isinstance(answer, str)
    assert len(answer) > 0


def test_web_search_node():
    state: GraphState = {
        "question": "Current weather in Tokyo",
        "documents": [],
        "web_search_needed": True,
        "generation": None,
        "retry_count": 0,
        "search_query": "Current weather in Tokyo",
    }

    update = web_search_node(state)
    assert len(update["documents"]) > 0
    assert update["web_search_needed"] is False
    assert update["documents"][0].retrieval_method == "web"


def test_generate_node():
    doc = SearchResult(
        chunk_id="d1",
        content="Acme Corp provides $1,500 equipment stipend.",
        metadata=DocumentMetadata(source="policy.txt", chunk_index=0, char_count=44),
        score=0.95,
        retrieval_method="hybrid",
        rank=1,
    )

    state: GraphState = {
        "question": "What is the equipment stipend?",
        "documents": [doc],
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }

    update = generate_node(state)
    assert "generation" in update
    assert isinstance(update["generation"], str)
    assert len(update["generation"]) > 0


def test_crag_end_to_end_in_domain(tmp_path, policy_chunks):
    embeddings = DeterministicHashEmbeddings()
    vs = ChromaVectorStore(
        collection_name="day4_indomain_chroma",
        persist_dir=tmp_path / "chroma_day4_in",
        embeddings=embeddings,
    )
    vs.add_chunks(policy_chunks)
    bm = BM25KeywordRetriever(chunks=policy_chunks)
    retriever = HybridRetriever(vector_store=vs, bm25_retriever=bm)
    set_default_retriever(retriever)

    graph = create_crag_graph()

    input_state: GraphState = {
        "question": "equipment stipend home office",
        "documents": [],
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }

    final_state = graph.invoke(input_state)
    assert final_state["generation"] is not None
    assert final_state["web_search_needed"] is False
    assert final_state["documents"][0].retrieval_method == "hybrid"


def test_crag_end_to_end_out_of_domain_web_fallback(tmp_path, policy_chunks):
    embeddings = DeterministicHashEmbeddings()
    vs = ChromaVectorStore(
        collection_name="day4_fallback_chroma",
        persist_dir=tmp_path / "chroma_day4_fall",
        embeddings=embeddings,
    )
    vs.add_chunks(policy_chunks)
    bm = BM25KeywordRetriever(chunks=policy_chunks)
    retriever = HybridRetriever(vector_store=vs, bm25_retriever=bm)
    set_default_retriever(retriever)

    graph = create_crag_graph()

    input_state: GraphState = {
        "question": "orbital velocity of James Webb Space Telescope",
        "documents": [],
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }

    final_state = graph.invoke(input_state)
    assert final_state["generation"] is not None
    assert final_state["documents"][0].retrieval_method == "web"
