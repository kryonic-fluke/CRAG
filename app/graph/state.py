from typing import List, Optional
from typing_extensions import TypedDict
from app.schemas.retrieval import SearchResult


class GraphState(TypedDict):
    question: str
    documents: List[SearchResult]
    web_search_needed: bool
    generation: Optional[str]
    retry_count: int
    search_query: Optional[str]
