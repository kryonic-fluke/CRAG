"""
Day 2 Retrieval Demo Script
Compares Dense Vector Search vs. Sparse BM25 Keyword Search vs. Hybrid Reciprocal Rank Fusion.
"""

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
from app.schemas.retrieval import HybridRetrieverConfig


def run_demo() -> None:
    print("=" * 70)
    print("[*] DAY 2 DEMO: DENSE vs. BM25 vs. HYBRID RETRIEVAL (RRF)")
    print("=" * 70)

    data_dir = project_root / "data" / "sample_docs"
    print(f"\n[1] Ingesting documents from: {data_dir}...")
    pipeline = DocumentIngestionPipeline()
    ingest_result = pipeline.ingest_directory(data_dir)
    print(
        f"    Loaded {ingest_result.total_documents_loaded} documents -> {ingest_result.total_chunks_created} chunks."
    )

    chroma_dir = project_root / "data" / "chroma_demo_db"
    vector_store = ChromaVectorStore(
        collection_name="demo_collection", persist_dir=chroma_dir
    )
    vector_store.clear()
    vector_store.add_chunks(ingest_result.chunks)

    bm25_retriever = BM25KeywordRetriever(chunks=ingest_result.chunks)

    hybrid_config = HybridRetrieverConfig(
        top_k=2, dense_weight=0.5, bm25_weight=0.5, fusion_strategy="rrf"
    )
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store, bm25_retriever=bm25_retriever, config=hybrid_config
    )

    test_queries = [
        {
            "query": "$1,500 equipment stipend",
            "type": "Exact Keyword / Numerical Match",
            "explanation": "BM25 excels here because '$1,500' and 'stipend' are exact lexical tokens.",
        },
        {
            "query": "preventing employee exhaustion and promoting mental rest",
            "type": "Conceptual / Semantic Match",
            "explanation": "Dense Vector excels here because 'exhaustion/mental rest' maps to 'burnout/PTO' semantically.",
        },
        {
            "query": "How does CRAG handle irrelevant docs using Tavily?",
            "type": "Blended Query (Acronym + Conceptual)",
            "explanation": "Hybrid RRF shines here by combining the exact acronym 'CRAG' with semantic search.",
        },
    ]

    for i, t in enumerate(test_queries, start=1):
        q = t["query"]
        print("\n" + "=" * 70)
        print(f'[*] Query #{i}: "{q}"')
        print(f"    Type: {t['type']}")
        print(f"    Note: {t['explanation']}")
        print("=" * 70)

        dense_hits = vector_store.similarity_search(q, k=2)
        print("\n  [DENSE (Chroma)]")
        for hit in dense_hits:
            preview = hit.content.replace("\n", " ")[:90]
            print(
                f"    Rank #{hit.rank} | Score: {hit.score:.4f} | Source: {hit.metadata.source}"
            )
            print(f'    Snippet: "{preview}..."')

        bm25_hits = bm25_retriever.similarity_search(q, k=2)
        print("\n  [BM25 (Lexical)]")
        for hit in bm25_hits:
            preview = hit.content.replace("\n", " ")[:90]
            print(
                f"    Rank #{hit.rank} | Score: {hit.score:.4f} | Source: {hit.metadata.source}"
            )
            print(f'    Snippet: "{preview}..."')

        hybrid_response = hybrid_retriever.retrieve(q)
        print("\n  [HYBRID (RRF Fused)]")
        for hit in hybrid_response.results:
            preview = hit.content.replace("\n", " ")[:90]
            print(
                f"    Rank #{hit.rank} | Score: {hit.score:.4f} | Method: {hit.retrieval_method}"
            )
            print(f'    Snippet: "{preview}..."')

    print("\n" + "=" * 70)
    print("[OK] DAY 2 HYBRID RETRIEVAL DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
