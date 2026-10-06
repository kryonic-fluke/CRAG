import pytest
from pydantic import ValidationError

from app.schemas.document import DocumentChunk, DocumentMetadata
from app.schemas.retrieval import SearchResult, HybridRetrieverConfig
from app.retrieval.embeddings import DeterministicHashEmbeddings
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.bm25_retriever import BM25KeywordRetriever
from app.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_chunks():
    """Provides a controlled set of document chunks for testing."""
    return [
        DocumentChunk(
            id="chunk-1",
            content="Acme Corp gives an equipment stipend of $1,500 USD for home office setup.",
            metadata=DocumentMetadata(source="policy.txt", chunk_index=0, char_count=73)
        ),
        DocumentChunk(
            id="chunk-2",
            content="Employees have an unlimited PTO policy with mandatory minimum 20 days off.",
            metadata=DocumentMetadata(source="policy.txt", chunk_index=1, char_count=74)
        ),
        DocumentChunk(
            id="chunk-3",
            content="Corrective RAG (CRAG) evaluates document relevance and falls back to Tavily web search.",
            metadata=DocumentMetadata(source="crag.txt", chunk_index=0, char_count=87)
        ),
        DocumentChunk(
            id="chunk-4",
            content="Production deployments are frozen on Friday after 2:00 PM to ensure weekend stability.",
            metadata=DocumentMetadata(source="policy.txt", chunk_index=2, char_count=87)
        )
    ]


def test_search_result_schema(sample_chunks):
    """Verify that SearchResult validates scores and formats correctly."""
    result = SearchResult.from_chunk(
        chunk=sample_chunks[0],
        score=0.92,
        method="dense",
        rank=1
    )
    assert result.chunk_id == "chunk-1"
    assert result.score == 0.92
    assert result.retrieval_method == "dense"
    assert result.rank == 1


def test_retriever_config_validation():
    """Verify that configuration prevents invalid weights (both 0.0)."""
    with pytest.raises(ValidationError):
        HybridRetrieverConfig(dense_weight=0.0, bm25_weight=0.0)

    # Valid config
    cfg = HybridRetrieverConfig(dense_weight=0.7, bm25_weight=0.3, top_k=5)
    assert cfg.dense_weight == 0.7
    assert cfg.top_k == 5


def test_bm25_keyword_search(sample_chunks):
    """Verify that BM25 accurately matches exact keywords like '$1,500' and 'PTO'."""
    bm25 = BM25KeywordRetriever(chunks=sample_chunks)
    assert bm25.count() == 4

    # Search for dollar amount and stipend
    results = bm25.similarity_search(query="$1,500 equipment stipend", k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score > 0.0
    assert results[0].retrieval_method == "bm25"

    # Search for PTO
    results_pto = bm25.similarity_search(query="mandatory PTO days", k=1)
    assert len(results_pto) == 1
    assert results_pto[0].chunk_id == "chunk-2"


def test_chroma_vector_store_deterministic(tmp_path, sample_chunks):
    """Verify that ChromaVectorStore indexes and searches correctly."""
    embeddings = DeterministicHashEmbeddings()
    store = ChromaVectorStore(
        collection_name="test_collection",
        persist_dir=tmp_path / "test_chroma",
        embeddings=embeddings
    )

    indexed_count = store.add_chunks(sample_chunks)
    assert indexed_count == 4
    assert store.count() == 4

    results = store.similarity_search(query="stipend equipment", k=2)
    assert len(results) == 2
    assert results[0].retrieval_method == "dense"
    assert 0.0 <= results[0].score <= 1.0


def test_hybrid_rrf_fusion(tmp_path, sample_chunks):
    """Verify that HybridRetriever fuses dense and BM25 using Reciprocal Rank Fusion."""
    embeddings = DeterministicHashEmbeddings()
    vector_store = ChromaVectorStore(
        collection_name="test_hybrid_collection",
        persist_dir=tmp_path / "test_hybrid",
        embeddings=embeddings
    )
    vector_store.add_chunks(sample_chunks)

    bm25 = BM25KeywordRetriever(chunks=sample_chunks)

    config = HybridRetrieverConfig(
        top_k=3,
        dense_weight=0.5,
        bm25_weight=0.5,
        fusion_strategy="rrf"
    )
    hybrid = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        config=config
    )

    response = hybrid.retrieve(query="Tavily web search CRAG")
    assert response.total_results <= 3
    assert len(response.results) > 0
    assert response.results[0].retrieval_method == "hybrid"
    assert "Hybrid RRF" in response.strategy_used


def test_hybrid_weighted_score_fusion(tmp_path, sample_chunks):
    """Verify that HybridRetriever supports weighted score fusion."""
    embeddings = DeterministicHashEmbeddings()
    vector_store = ChromaVectorStore(
        collection_name="test_weighted_collection",
        persist_dir=tmp_path / "test_weighted",
        embeddings=embeddings
    )
    vector_store.add_chunks(sample_chunks)

    bm25 = BM25KeywordRetriever(chunks=sample_chunks)

    config = HybridRetrieverConfig(
        top_k=2,
        dense_weight=0.6,
        bm25_weight=0.4,
        fusion_strategy="weighted_score"
    )
    hybrid = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25,
        config=config
    )

    response = hybrid.retrieve(query="Friday deployment freeze")
    assert response.total_results <= 2
    assert "Hybrid Weighted Score" in response.strategy_used
    assert response.results[0].chunk_id == "chunk-4"

