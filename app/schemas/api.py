from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        description="User question to be answered by the CRAG pipeline",
        examples=["What is Acme Corp's remote work policy?"],
    )
    enable_web_fallback: bool = Field(
        default=True,
        description="Whether to allow falling back to Tavily web search if internal docs fail",
    )


class SourceDocument(BaseModel):
    chunk_id: str
    source: str
    snippet: str
    retrieval_method: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]
    web_search_triggered: bool
    retry_count: int


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    model_name: str
    version: str

