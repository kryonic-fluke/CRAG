from typing import Any, Literal, Optional
from app.graph.state import GraphState
from app.graph.chains.hallucination_grader import get_hallucination_grader_chain
from app.graph.chains.answer_grader import get_answer_grader_chain


def route_after_grading(state: GraphState) -> Literal["generate", "web_search"]:
    if state.get("web_search_needed", False):
        return "web_search"
    return "generate"


def check_hallucination_and_relevance(
    state: GraphState,
    hallucination_chain: Optional[Any] = None,
    answer_chain: Optional[Any] = None,
) -> Literal["useful", "not_useful", "not_grounded"]:
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    question = state.get("question", "")
    retry_count = state.get("retry_count", 0)

    context = "\n\n---\n\n".join([doc.content for doc in documents])

    h_chain = hallucination_chain or get_hallucination_grader_chain()
    a_chain = answer_chain or get_answer_grader_chain()

    grade_hallucination = h_chain.invoke(
        {"documents": context, "generation": generation}
    )

    if grade_hallucination.binary_score == "yes":
        grade_answer = a_chain.invoke({"question": question, "generation": generation})
        if grade_answer.binary_score == "yes":
            return "useful"
        return "not_useful"

    if retry_count < 2:
        return "not_grounded"

    return "useful"
