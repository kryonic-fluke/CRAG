import pytest

from app.schemas.document import DocumentChunk, DocumentMetadata
from app.schemas.retrieval import SearchResult
from app.graph.state import GraphState
from app.graph.chains.hallucination_grader import get_hallucination_grader_chain
from app.graph.chains.answer_grader import get_answer_grader_chain
from app.graph.edges import check_hallucination_and_relevance
from app.graph.nodes.retrieve import set_default_retriever
from app.graph.workflow import create_crag_graph
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.bm25_retriever import BM25KeywordRetriever
from app.retrieval.embeddings import DeterministicHashEmbeddings


@pytest.fixture
def policy_context():
    return "Acme Corp operates an unlimited PTO policy with a mandatory minimum of 20 days off per calendar year to prevent burnout."


def test_hallucination_grader_grounded_vs_hallucinated(policy_context):
    chain = get_hallucination_grader_chain()

    grounded_ans = "Acme Corp provides an unlimited PTO policy with a mandatory minimum of 20 days off."
    res_grounded = chain.invoke(
        {"documents": policy_context, "generation": grounded_ans}
    )
    assert res_grounded.binary_score == "yes"

    hallucinated_ans = "Acme Corp grants 500 stock options and free lamborghinis (completely unverified claim)."
    res_hallucinated = chain.invoke(
        {"documents": policy_context, "generation": hallucinated_ans}
    )
    assert res_hallucinated.binary_score == "no"


def test_answer_grader_relevant_vs_irrelevant():
    chain = get_answer_grader_chain()

    q = "What is the mandatory minimum PTO?"
    ans_relevant = "The mandatory minimum is 20 days off per calendar year."
    res_rel = chain.invoke({"question": q, "generation": ans_relevant})
    assert res_rel.binary_score == "yes"

    ans_irrel = "I do not have enough context to answer this question."
    res_irrel = chain.invoke({"question": q, "generation": ans_irrel})
    assert res_irrel.binary_score == "no"


def test_guardrail_edge_useful(policy_context):
    doc = SearchResult(
        chunk_id="c1",
        content=policy_context,
        metadata=DocumentMetadata(
            source="p.txt", chunk_index=0, char_count=len(policy_context)
        ),
        score=0.9,
        retrieval_method="hybrid",
        rank=1,
    )

    state: GraphState = {
        "question": "What is the mandatory minimum PTO policy?",
        "documents": [doc],
        "web_search_needed": False,
        "generation": "Acme Corp offers unlimited PTO with a mandatory minimum of 20 days off per calendar year.",
        "retry_count": 0,
        "search_query": None,
    }

    decision = check_hallucination_and_relevance(state)
    assert decision == "useful"


def test_guardrail_edge_not_grounded_retry_loop(policy_context):
    doc = SearchResult(
        chunk_id="c1",
        content=policy_context,
        metadata=DocumentMetadata(
            source="p.txt", chunk_index=0, char_count=len(policy_context)
        ),
        score=0.9,
        retrieval_method="hybrid",
        rank=1,
    )

    state: GraphState = {
        "question": "What is the mandatory minimum PTO policy?",
        "documents": [doc],
        "web_search_needed": False,
        "generation": "Acme Corp gives free cars to employees (completely unverified claim).",
        "retry_count": 0,
        "search_query": None,
    }

    decision = check_hallucination_and_relevance(state)
    assert decision == "not_grounded"


def test_guardrail_edge_exhausted_retries(policy_context):
    doc = SearchResult(
        chunk_id="c1",
        content=policy_context,
        metadata=DocumentMetadata(
            source="p.txt", chunk_index=0, char_count=len(policy_context)
        ),
        score=0.9,
        retrieval_method="hybrid",
        rank=1,
    )

    state: GraphState = {
        "question": "What is the mandatory minimum PTO policy?",
        "documents": [doc],
        "web_search_needed": False,
        "generation": "Acme Corp gives free cars to employees (completely unverified claim).",
        "retry_count": 2,
        "search_query": None,
    }

    decision = check_hallucination_and_relevance(state)
    assert decision == "useful"


def test_crag_end_to_end_with_guardrails(tmp_path):
    chunks = [
        DocumentChunk(
            id="c1",
            content="Acme Corp gives an equipment stipend of $1,500 USD for home office setup.",
            metadata=DocumentMetadata(
                source="policy.txt", chunk_index=0, char_count=73
            ),
        )
    ]

    embeddings = DeterministicHashEmbeddings()
    vs = ChromaVectorStore(
        collection_name="day5_chroma",
        persist_dir=tmp_path / "chroma_day5",
        embeddings=embeddings,
    )
    vs.add_chunks(chunks)
    bm = BM25KeywordRetriever(chunks=chunks)
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
    assert final_state["generation"] is not None
    assert final_state["retry_count"] >= 1
    assert len(final_state["documents"]) > 0
