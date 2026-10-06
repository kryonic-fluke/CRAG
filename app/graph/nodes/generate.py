from typing import Any, Dict, Optional
from app.graph.state import GraphState
from app.graph.chains.generation import get_generation_chain


def generate_node(
    state: GraphState, generator_chain: Optional[Any] = None
) -> Dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])

    chain = generator_chain or get_generation_chain()

    context = "\n\n---\n\n".join([doc.content for doc in documents])
    generation = chain.invoke({"context": context, "question": question})

    return {
        "generation": generation,
    }

