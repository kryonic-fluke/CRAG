from typing import Any, Dict, Optional
from app.graph.state import GraphState
from app.retrieval.web_search import TavilySearchWrapper


def web_search_node(
    state: GraphState, searcher: Optional[TavilySearchWrapper] = None
) -> Dict[str, Any]:
    query = state.get("search_query") or state.get("question")
    wrapper = searcher or TavilySearchWrapper()
    results = wrapper.search(query=query, max_results=3)

    return {
        "documents": results,
        "web_search_needed": False,
    }

