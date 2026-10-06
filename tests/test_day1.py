from pathlib import Path
import pytest
from pydantic import ValidationError

from app.schemas.document import DocumentChunk, DocumentMetadata, IngestConfig
from app.retrieval.document_loader import DocumentIngestionPipeline


def test_valid_document_chunk_creation():
    """Verify that a properly formed chunk passes validation and auto-generates UUID and timestamp."""
    meta = DocumentMetadata(source="doc1.txt", chunk_index=0, char_count=20)
    chunk = DocumentChunk(content="This is sample text.", metadata=meta)

    assert chunk.content == "This is sample text."
    assert chunk.metadata.source == "doc1.txt"
    assert len(chunk.id) > 0
    assert chunk.metadata.created_at is not None


def test_empty_content_rejected():
    """Verify that empty or pure whitespace content raises a Pydantic ValidationError."""
    meta = DocumentMetadata(source="doc1.txt", chunk_index=0, char_count=0)

    with pytest.raises(ValidationError) as exc_info:
        DocumentChunk(content="   ", metadata=meta)

    assert "DocumentChunk content cannot be empty or pure whitespace" in str(exc_info.value)


def test_ingest_config_overlap_validation():
    """Verify that chunk_overlap >= chunk_size raises a validation error."""
    with pytest.raises(ValidationError) as exc_info:
        IngestConfig(chunk_size=200, chunk_overlap=250)

    assert "chunk_overlap" in str(exc_info.value)


def test_ingestion_pipeline_processes_text():
    """Verify that DocumentIngestionPipeline splits text into validated Pydantic chunks."""
    pipeline = DocumentIngestionPipeline(config=IngestConfig(chunk_size=50, chunk_overlap=10))
    sample_text = "Paragraph one with some meaningful text.\n\nParagraph two with another sentence."

    chunks = pipeline.process_text(text=sample_text, source="memory_doc")

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.metadata.source == "memory_doc"
        assert len(chunk.content) > 0


def test_ingestion_pipeline_sample_docs():
    """Verify that sample docs directory is ingested without errors."""
    data_dir = Path("data/sample_docs")
    assert data_dir.exists(), "Sample docs directory must exist"

    pipeline = DocumentIngestionPipeline(config=IngestConfig(chunk_size=400, chunk_overlap=80))
    result = pipeline.ingest_directory(data_dir)

    assert result.total_documents_loaded >= 2
    assert result.total_chunks_created > 0
    assert len(result.chunks) == result.total_chunks_created

