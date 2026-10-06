import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
from app.graph.chains.hallucination_grader import get_hallucination_grader_chain
from app.graph.chains.answer_grader import get_answer_grader_chain

BENCHMARK_DATASET = [
    {
        "id": 1,
        "query": "What is Acme Corp's remote work policy?",
        "category": "In-Domain Policy",
        "expected_fallback": False,
    },
    {
        "id": 2,
        "query": "How much is the equipment stipend for new hires?",
        "category": "In-Domain Policy",
        "expected_fallback": False,
    },
    {
        "id": 3,
        "query": "What is the mandatory minimum PTO days per calendar year?",
        "category": "In-Domain Policy",
        "expected_fallback": False,
    },
    {
        "id": 4,
        "query": "When are production deployments frozen on Fridays?",
        "category": "In-Domain Policy",
        "expected_fallback": False,
    },
    {
        "id": 5,
        "query": "Who must approve a Pull Request before merging?",
        "category": "In-Domain Policy",
        "expected_fallback": False,
    },
    {
        "id": 6,
        "query": "How does Corrective RAG differ from standard RAG?",
        "category": "In-Domain Architecture",
        "expected_fallback": False,
    },
    {
        "id": 7,
        "query": "What two retrieval methods are combined in hybrid retrieval?",
        "category": "In-Domain Architecture",
        "expected_fallback": False,
    },
    {
        "id": 8,
        "query": "What does the Document Relevance Grader node do?",
        "category": "In-Domain Architecture",
        "expected_fallback": False,
    },
    {
        "id": 9,
        "query": "When does the CRAG pipeline trigger a web search fallback?",
        "category": "In-Domain Architecture",
        "expected_fallback": False,
    },
    {
        "id": 10,
        "query": "What role does the hallucination guardrail play in CRAG?",
        "category": "In-Domain Architecture",
        "expected_fallback": False,
    },
    {
        "id": 11,
        "query": "What is the launch date and primary instrument of James Webb Space Telescope?",
        "category": "Out-of-Domain External",
        "expected_fallback": True,
    },
    {
        "id": 12,
        "query": "What was the capital of the Byzantine Empire in the 10th century?",
        "category": "Out-of-Domain External",
        "expected_fallback": True,
    },
    {
        "id": 13,
        "query": "What is the chemical composition of lunar regolith soil?",
        "category": "Out-of-Domain External",
        "expected_fallback": True,
    },
    {
        "id": 14,
        "query": "Who won the FIFA World Cup tournament in 2022?",
        "category": "Out-of-Domain External",
        "expected_fallback": True,
    },
    {
        "id": 15,
        "query": "What is the speed of sound in dry air at room temperature?",
        "category": "Out-of-Domain External",
        "expected_fallback": True,
    },
]


def run_evaluation_benchmark() -> Dict[str, Any]:
    data_dir = project_root / "data" / "sample_docs"
    pipeline = DocumentIngestionPipeline()
    ingest_result = pipeline.ingest_directory(data_dir)

    chroma_dir = project_root / "data" / "chroma_eval_db"
    vs = ChromaVectorStore(collection_name="eval_db", persist_dir=chroma_dir)
    vs.clear()
    vs.add_chunks(ingest_result.chunks)

    bm = BM25KeywordRetriever(chunks=ingest_result.chunks)
    hybrid = HybridRetriever(vector_store=vs, bm25_retriever=bm)
    set_default_retriever(hybrid)

    graph = create_crag_graph()
    hallucination_grader = get_hallucination_grader_chain()
    answer_grader = get_answer_grader_chain()

    results: List[Dict[str, Any]] = []

    for item in BENCHMARK_DATASET:
        t0 = time.perf_counter()
        state: GraphState = {
            "question": item["query"],
            "documents": [],
            "web_search_needed": False,
            "generation": None,
            "retry_count": 0,
            "search_query": None,
        }

        output = graph.invoke(state)
        latency_ms = (time.perf_counter() - t0) * 1000

        docs = output.get("documents", [])
        answer = output.get("generation", "")
        has_web = any(d.retrieval_method == "web" for d in docs)

        routing_correct = has_web == item["expected_fallback"]

        context_str = "\n\n---\n\n".join([d.content for d in docs])
        h_grade = hallucination_grader.invoke(
            {"documents": context_str, "generation": answer}
        )
        is_grounded = h_grade.binary_score == "yes"

        a_grade = answer_grader.invoke(
            {"question": item["query"], "generation": answer}
        )
        is_relevant = a_grade.binary_score == "yes"

        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "category": item["category"],
                "expected_fallback": item["expected_fallback"],
                "actual_fallback": has_web,
                "routing_correct": routing_correct,
                "is_grounded": is_grounded,
                "is_relevant": is_relevant,
                "latency_ms": round(latency_ms, 2),
            }
        )

    total = len(results)
    routing_acc = (sum(1 for r in results if r["routing_correct"]) / total) * 100.0
    faithfulness_rate = (sum(1 for r in results if r["is_grounded"]) / total) * 100.0
    relevance_rate = (sum(1 for r in results if r["is_relevant"]) / total) * 100.0
    avg_latency = sum(r["latency_ms"] for r in results) / total

    summary = {
        "total_queries": total,
        "routing_accuracy_pct": round(routing_acc, 1),
        "faithfulness_score_pct": round(faithfulness_rate, 1),
        "answer_relevance_pct": round(relevance_rate, 1),
        "average_latency_ms": round(avg_latency, 2),
        "results": results,
    }

    return summary


def print_benchmark_report(summary: Dict[str, Any]) -> None:
    print("=" * 80)
    print("🎯 CRAG SYSTEM BENCHMARK EVALUATION SUITE (15 TEST QUERIES)")
    print("=" * 80)

    print(f"\n[SUMMARY METRICS]")
    print(f"  • Total Benchmark Queries:      {summary['total_queries']}")
    print(f"  • Routing Decision Accuracy:    {summary['routing_accuracy_pct']}%")
    print(f"  • Faithfulness Score:           {summary['faithfulness_score_pct']}%")
    print(f"  • Answer Relevance Score:       {summary['answer_relevance_pct']}%")
    print(f"  • Average Latency:              {summary['average_latency_ms']} ms")

    print("\n" + "-" * 80)
    print(
        f"{'#':<3} | {'Category':<22} | {'Routing':<8} | {'Grounded':<8} | {'Relevant':<8} | {'Query'}"
    )
    print("-" * 80)

    for r in summary["results"]:
        route_str = "PASS" if r["routing_correct"] else "FAIL"
        ground_str = "PASS" if r["is_grounded"] else "FAIL"
        rel_str = "PASS" if r["is_relevant"] else "FAIL"
        q_snippet = r["query"][:30] + "..." if len(r["query"]) > 30 else r["query"]
        print(
            f"{r['id']:<3} | {r['category']:<22} | {route_str:<8} | {ground_str:<8} | {rel_str:<8} | {q_snippet}"
        )

    print("=" * 80)


if __name__ == "__main__":
    benchmark_summary = run_evaluation_benchmark()
    print_benchmark_report(benchmark_summary)
