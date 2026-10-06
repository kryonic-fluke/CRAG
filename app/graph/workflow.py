from langgraph.graph import StateGraph, START, END

from app.graph.state import GraphState
from app.graph.nodes.retrieve import retrieve_node
from app.graph.nodes.grade_documents import grade_documents_node
from app.graph.nodes.web_search import web_search_node
from app.graph.nodes.generate import generate_node
from app.graph.edges import route_after_grading, check_hallucination_and_relevance


def create_crag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

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

    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        check_hallucination_and_relevance,
        {
            "useful": END,
            "not_useful": "web_search",
            "not_grounded": "generate",
        },
    )

    return workflow.compile()
