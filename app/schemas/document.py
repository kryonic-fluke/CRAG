from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator


class DocumentMetadata(BaseModel):
    """
    Metadata schema associated with an individual document chunk.

    Equivalent to a strict TypeScript interface:
    interface DocumentMetadata {
      source: string;
      chunk_index: number;
      char_count: number;
      created_at: string;
      extra?: Record<string, any>;
    }
    """

    source: str = Field(
        ...,
        description="Source file path, URL, or identifier from which the document originated",
    )
    chunk_index: int = Field(
        default=0,
        ge=0,
        description="0-based sequential index of this chunk within the source document",
    )
    char_count: int = Field(
        default=0, ge=0, description="Total character length of the chunk content"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp (UTC) when this chunk was generated",
    )
    extra: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary additional key-value tags (e.g. department, author, date)",
    )


class DocumentChunk(BaseModel):
    """
    Canonical representation of a chunked document unit in the CRAG pipeline.

    Guarantees:
    - Content is never empty or pure whitespace.
    - An immutable UUID is generated if an ID is not provided.
    - Metadata conforms to the DocumentMetadata schema.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the chunk",
    )
    content: str = Field(
        ..., min_length=1, description="Raw text content of the document chunk"
    )
    metadata: DocumentMetadata = Field(
        ..., description="Structured metadata accompanying the text chunk"
    )

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, value: str) -> str:
        """Ensure content isn't just whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "DocumentChunk content cannot be empty or pure whitespace."
            )
        return stripped


class IngestConfig(BaseModel):
    """
    Configuration parameters for chunking and ingestion.
    Demonstrates Pydantic model validation (validating relationships between fields).
    """

    chunk_size: int = Field(
        default=500,
        ge=50,
        le=4000,
        description="Target character size of each text chunk",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Number of overlapping characters between adjacent chunks",
    )

    @model_validator(mode="after")
    def validate_overlap_less_than_size(self) -> "IngestConfig":
        """Cross-field validation: overlap must be strictly less than chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})."
            )
        return self


class IngestResult(BaseModel):
    """
    Summary result returned after ingesting documents.
    """

    total_documents_loaded: int = Field(..., ge=0)
    total_chunks_created: int = Field(..., ge=0)
    sources: List[str] = Field(default_factory=list)
    chunks: List[DocumentChunk] = Field(default_factory=list)
