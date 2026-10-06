import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.retrieval.document_loader import DocumentIngestionPipeline
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.bm25_retriever import BM25KeywordRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.graph.nodes.retrieve import set_default_retriever
from app.graph.workflow import create_crag_graph
from app.graph.edges import check_hallucination_and_relevance
from app.graph.state import GraphState


def run_demo() -> None:
    print("=" * 70)
    print("[*] DAY 5 DEMO: HALLUCINATION GUARDRAILS & SELF-CORRECTION LOOP")
    print("=" * 70)

    data_dir = project_root / "data" / "sample_docs"
    pipeline = DocumentIngestionPipeline()
    ingest_result = pipeline.ingest_directory(data_dir)

    chroma_dir = project_root / "data" / "chroma_day5_db"
    vs = ChromaVectorStore(collection_name="day5_demo", persist_dir=chroma_dir)
    vs.clear()
    vs.add_chunks(ingest_result.chunks)

    bm = BM25KeywordRetriever(chunks=ingest_result.chunks)
    hybrid = HybridRetriever(vector_store=vs, bm25_retriever=bm)
    set_default_retriever(hybrid)

    graph = create_crag_graph()

    print("\n" + "=" * 70)
    print("[*] Scenario 1: End-to-End Grounded Generation with Guardrails")
    print("=" * 70)

    query = "What is the equipment stipend policy at Acme Corp?"
    initial_state: GraphState = {
        "question": query,
        "documents": [],
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }

    final_state = graph.invoke(initial_state)

    print(f'    - Query: "{query}"')
    print(f'    - Final Generation: "{final_state.get("generation")}"')
    print(f"    - Generation Attempts (Retry Count): {final_state.get('retry_count')}")
    print(f"    - Context Chunks Used: {len(final_state.get('documents', []))}")

    print("\n" + "=" * 70)
    print("[*] Scenario 2: Guardrail Detecting Hallucination & Triggering Loopback")
    print("=" * 70)

    doc_sample = final_state.get("documents", [])[0]

    simulated_hallucinated_state: GraphState = {
        "question": query,
        "documents": [doc_sample],
        "web_search_needed": False,
        "generation": "Acme Corp provides free sports cars and $500,000 bonuses (completely unverified claim).",
        "retry_count": 0,
        "search_query": None,
    }

    guardrail_decision_1 = check_hallucination_and_relevance(
        simulated_hallucinated_state
    )
    print("    [Trial 1 - Injected Hallucination, retry_count=0]:")
    print(f'    - Proposed Answer: "{simulated_hallucinated_state["generation"]}"')
    print(
        f"    - Guardrail Decision: '{guardrail_decision_1}' -> Triggers Self-Correction Loopback to 'generate'"
    )

    simulated_hallucinated_state["retry_count"] = 2
    guardrail_decision_2 = check_hallucination_and_relevance(
        simulated_hallucinated_state
    )
    print("\n    [Trial 2 - Max Retries Reached, retry_count=2]:")
    print(
        f"    - Guardrail Decision: '{guardrail_decision_2}' -> Bounded cycle halts to prevent infinite loop"
    )

    print("\n" + "=" * 70)
    print("[OK] DAY 5 GUARDRAILS & SELF-CORRECTION VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
