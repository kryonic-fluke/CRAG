from typing import Any, Dict
from langgraph.graph import StateGraph, START, END

from app.graph.state import GraphState
from app.graph.nodes.retrieve import retrieve_node
from app.graph.nodes.grade_documents import grade_documents_node
from app.graph.edges import route_after_grading


def placeholder_generate_node(state: GraphState) -> Dict[str, Any]:
    docs_count = len(state.get("documents", []))
    return {"generation": f"Generated answer based on {docs_count} relevant documents."}


def placeholder_web_search_node(state: GraphState) -> Dict[str, Any]:
    query = state.get("search_query") or state.get("question")
    return {"generation": f"Fallback to web search for: {query}."}


def create_crag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", placeholder_generate_node)
    workflow.add_node("web_search", placeholder_web_search_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "web_search": "web_search",
        },
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("web_search", END)

    return workflow.compile()
