from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

from app.schemas.document import DocumentChunk, DocumentMetadata


class SearchResult(BaseModel):
    """
    Standardized, validated result from any retrieval strategy (Dense, BM25, or Hybrid).
    
    Guarantees:
    - Every retrieved document chunk has an associated normalized score [0.0, 1.0].
    - Originating retrieval method is explicitly tracked.
    - Rank position in the final fusion list is preserved.
    """
    chunk_id: str = Field(..., description="Unique ID of the document chunk")
    content: str = Field(..., description="Text content of the retrieved chunk")
    metadata: DocumentMetadata = Field(..., description="Source metadata for attribution")
    score: float = Field(
        ...,
        ge=0.0,
        description="Relevance or fusion score (higher is more relevant)"
    )
    retrieval_method: Literal["dense", "bm25", "hybrid"] = Field(
        ...,
        description="The retrieval engine that fetched or synthesized this result"
    )
    rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="1-based rank position in the final result list"
    )

    @classmethod
    def from_chunk(
        cls,
        chunk: DocumentChunk,
        score: float,
        method: Literal["dense", "bm25", "hybrid"],
        rank: Optional[int] = None
    ) -> "SearchResult":
        return cls(
            chunk_id=chunk.id,
            content=chunk.content,
            metadata=chunk.metadata,
            score=score,
            retrieval_method=method,
            rank=rank
        )


class HybridRetrieverConfig(BaseModel):
    """
    Configuration for the Hybrid Retriever.
    
    Controls the blend between semantic similarity (Dense) and keyword precision (BM25).
    """
    top_k: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Number of final documents to return"
    )
    dense_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight for dense vector embeddings (0.0 to 1.0)"
    )
    bm25_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative weight for BM25 lexical keyword matching (0.0 to 1.0)"
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description="Smoothing constant k for Reciprocal Rank Fusion formula: 1 / (k + rank)"
    )
    fusion_strategy: Literal["rrf", "weighted_score"] = Field(
        default="rrf",
        description="Fusion algorithm to combine results: 'rrf' (rank-based) or 'weighted_score' (score-based)"
    )

    @model_validator(mode="after")
    def validate_weights(self) -> "HybridRetrieverConfig":
        if self.dense_weight == 0.0 and self.bm25_weight == 0.0:
            raise ValueError("At least one of dense_weight or bm25_weight must be greater than 0.")
        return self


class RetrievalResponse(BaseModel):
    """
    Payload containing the retrieved candidates and query profiling metadata.
    """
    query: str = Field(..., description="Original user query")
    total_results: int = Field(..., ge=0)
    results: List[SearchResult] = Field(default_factory=list)
    strategy_used: str = Field(..., description="Summary of retrieval strategy applied")

