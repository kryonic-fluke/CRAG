from typing import Any, Dict, List, Optional
from app.graph.state import GraphState
from app.graph.chains.doc_grader import get_doc_grader_chain
from app.schemas.retrieval import SearchResult
from app.schemas.grader import GradeDocuments


def grade_documents_node(
    state: GraphState, grader_chain: Optional[Any] = None
) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])

    chain = grader_chain or get_doc_grader_chain()

    filtered_docs: List[SearchResult] = []
    web_search_needed: bool = False

    for doc in documents:
        grade: GradeDocuments = chain.invoke(
            {"question": question, "document": doc.content}
        )
        if grade.binary_score == "yes":
            filtered_docs.append(doc)

    if not filtered_docs:
        web_search_needed = True

    return {
        "documents": filtered_docs,
        "web_search_needed": web_search_needed,
        "search_query": question if web_search_needed else None,
    }
