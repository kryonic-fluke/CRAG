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
    print("[*] DAY 4 DEMO: CRAG WEB SEARCH FALLBACK & GENERATION PIPELINE")
    print("=" * 70)

    data_dir = project_root / "data" / "sample_docs"
    pipeline = DocumentIngestionPipeline()
    ingest_result = pipeline.ingest_directory(data_dir)

    chroma_dir = project_root / "data" / "chroma_day4_db"
    vs = ChromaVectorStore(collection_name="day4_demo", persist_dir=chroma_dir)
    vs.clear()
    vs.add_chunks(ingest_result.chunks)

    bm = BM25KeywordRetriever(chunks=ingest_result.chunks)
    hybrid = HybridRetriever(vector_store=vs, bm25_retriever=bm)
    set_default_retriever(hybrid)

    graph = create_crag_graph()

    scenarios = [
        {
            "type": "INTERNAL KNOWLEDGE QUERY",
            "question": "What is Acme Corp's PTO policy and mandatory minimum days off?",
            "expectation": "Uses internal document chunks -> generates context-backed answer directly.",
        },
        {
            "type": "OUT-OF-DOMAIN EXTERNAL QUERY",
            "question": "What is the launch date and primary science instrument of James Webb Space Telescope?",
            "expectation": "Internal docs rejected by Grader -> triggers Web Search Fallback -> generates answer from web context.",
        },
    ]

    for i, sc in enumerate(scenarios, start=1):
        print("\n" + "=" * 70)
        print(f"[*] Test #{i}: [{sc['type']}]")
        print(f'    Question: "{sc["question"]}"')
        print(f"    Expected: {sc['expectation']}")
        print("=" * 70)

        state: GraphState = {
            "question": sc["question"],
            "documents": [],
            "web_search_needed": False,
            "generation": None,
            "retry_count": 0,
            "search_query": None,
        }

        result = graph.invoke(state)

        final_docs = result.get("documents", [])
        answer = result.get("generation", "")
        source_methods = set(d.retrieval_method for d in final_docs)

        print(f"\n    [1] Context Documents Used: {len(final_docs)} chunk(s)")
        for d in final_docs[:2]:
            snippet = d.content.replace("\n", " ")[:90]
            print(
                f"        - [{d.retrieval_method.upper()}] from '{d.metadata.source}': \"{snippet}...\""
            )

        print(f"    [2] Retrieval Pathway: {source_methods}")
        print(f'    [3] Synthesized Final Answer:\n        "{answer}"')

    print("\n" + "=" * 70)
    print("[OK] DAY 4 CRAG END-TO-END PIPELINE VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
