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
from app.graph.state import GraphState


def run_demo() -> None:
    print("=" * 70)
    print("[*] DAY 3 DEMO: LANGGRAPH STATE MACHINE & DOCUMENT RELEVANCE GRADER")
    print("=" * 70)

    data_dir = project_root / "data" / "sample_docs"
    pipeline = DocumentIngestionPipeline()
    ingest_result = pipeline.ingest_directory(data_dir)

    chroma_dir = project_root / "data" / "chroma_grader_db"
    vs = ChromaVectorStore(
        collection_name="grader_demo_collection", persist_dir=chroma_dir
    )
    vs.clear()
    vs.add_chunks(ingest_result.chunks)

    bm = BM25KeywordRetriever(chunks=ingest_result.chunks)
    hybrid = HybridRetriever(vector_store=vs, bm25_retriever=bm)
    set_default_retriever(hybrid)

    graph = create_crag_graph()

    test_scenarios = [
        {
            "title": "Scenario A: In-Domain Query (Relevant Docs Exist)",
            "question": "What is Acme Corp's remote work and Slack hours policy?",
            "expected_route": "generate",
        },
        {
            "title": "Scenario B: Completely Out-of-Domain Query (No Relevant Docs)",
            "question": "What is the orbital trajectory and fuel composition of the Saturn V rocket?",
            "expected_route": "web_search",
        },
    ]

    for i, scen in enumerate(test_scenarios, start=1):
        print("\n" + "=" * 70)
        print(f"[*] {scen['title']}")
        print(f'    Query: "{scen["question"]}"')
        print("=" * 70)

        initial_state: GraphState = {
            "question": scen["question"],
            "documents": [],
            "web_search_needed": False,
            "generation": None,
            "retry_count": 0,
            "search_query": None,
        }

        final_state = graph.invoke(initial_state)

        surviving_docs = final_state.get("documents", [])
        web_search = final_state.get("web_search_needed", False)
        generation_output = final_state.get("generation", "")

        print(f"\n    [1] Retrieved & Filtered Docs Count: {len(surviving_docs)}")
        for d in surviving_docs[:2]:
            snippet = d.content.replace("\n", " ")[:80]
            print(
                f'        - Matched [{d.chunk_id}] from {d.metadata.source}: "{snippet}..."'
            )

        print(f"    [2] Web Search Fallback Triggered: {web_search}")
        print(f"    [3] Routing Outcome: {generation_output}")

    print("\n" + "=" * 70)
    print("[OK] DAY 3 LANGGRAPH GRADING & ROUTING DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
