import pytest
from pydantic import ValidationError

from app.schemas.grader import GradeDocuments
from app.schemas.document import DocumentChunk, DocumentMetadata
from app.schemas.retrieval import SearchResult
from app.graph.state import GraphState
from app.graph.chains.doc_grader import get_doc_grader_chain
from app.graph.nodes.grade_documents import grade_documents_node
from app.graph.nodes.retrieve import set_default_retriever
from app.graph.edges import route_after_grading
from app.graph.workflow import create_crag_graph
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.bm25_retriever import BM25KeywordRetriever
from app.retrieval.embeddings import DeterministicHashEmbeddings


@pytest.fixture
def test_chunks():
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
            content="The solar system consists of eight planets orbiting the Sun.",
            metadata=DocumentMetadata(source="space.txt", chunk_index=0, char_count=60),
        ),
    ]


def test_grade_documents_schema():
    valid = GradeDocuments(
        binary_score="yes", reasoning="Document mentions equipment stipend."
    )
    assert valid.binary_score == "yes"

    with pytest.raises(ValidationError):
        GradeDocuments(binary_score="maybe", reasoning="invalid score")


def test_doc_grader_chain_evaluation(test_chunks):
    chain = get_doc_grader_chain()

    res_rel = chain.invoke(
        {
            "question": "How much is the equipment stipend?",
            "document": test_chunks[0].content,
        }
    )
    assert res_rel.binary_score == "yes"

    res_irrel = chain.invoke(
        {
            "question": "How much is the equipment stipend?",
            "document": test_chunks[1].content,
        }
    )
    assert res_irrel.binary_score == "no"


def test_grade_documents_node_filtering(test_chunks):
    search_results = [
        SearchResult.from_chunk(test_chunks[0], score=0.9, method="hybrid", rank=1),
        SearchResult.from_chunk(test_chunks[1], score=0.2, method="hybrid", rank=2),
    ]

    state: GraphState = {
        "question": "What is the equipment stipend amount?",
        "documents": search_results,
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }

    output = grade_documents_node(state)
    assert len(output["documents"]) == 1
    assert output["documents"][0].chunk_id == "c1"
    assert output["web_search_needed"] is False


def test_grade_documents_node_triggers_fallback(test_chunks):
    search_results = [
        SearchResult.from_chunk(test_chunks[1], score=0.2, method="hybrid", rank=1),
    ]

    state: GraphState = {
        "question": "What is the equipment stipend amount?",
        "documents": search_results,
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }

    output = grade_documents_node(state)
    assert len(output["documents"]) == 0
    assert output["web_search_needed"] is True
    assert output["search_query"] == "What is the equipment stipend amount?"


def test_route_after_grading_edge():
    state_gen: GraphState = {
        "question": "q",
        "documents": [],
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }
    assert route_after_grading(state_gen) == "generate"

    state_search: GraphState = {
        "question": "q",
        "documents": [],
        "web_search_needed": True,
        "generation": None,
        "retry_count": 0,
        "search_query": "q",
    }
    assert route_after_grading(state_search) == "web_search"


def test_compiled_graph_execution(tmp_path, test_chunks):
    embeddings = DeterministicHashEmbeddings()
    vs = ChromaVectorStore(
        collection_name="test_graph_collection",
        persist_dir=tmp_path / "chroma_graph",
        embeddings=embeddings,
    )
    vs.add_chunks(test_chunks)
    bm = BM25KeywordRetriever(chunks=test_chunks)
    retriever = HybridRetriever(vector_store=vs, bm25_retriever=bm)
    set_default_retriever(retriever)

    graph = create_crag_graph()

    input_state: GraphState = {
        "question": "equipment stipend amount",
        "documents": [],
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }

    final_state = graph.invoke(input_state)
    assert len(final_state["documents"]) > 0
    assert final_state["web_search_needed"] is False
    assert isinstance(final_state["generation"], str)
    assert len(final_state["generation"]) > 0

