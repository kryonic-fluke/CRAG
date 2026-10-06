from typing import Literal
from app.graph.state import GraphState


def route_after_grading(state: GraphState) -> Literal["generate", "web_search"]:
    if state.get("web_search_needed", False):
        return "web_search"
    return "generate"
