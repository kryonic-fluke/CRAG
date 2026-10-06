from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.retrieval.document_loader import DocumentIngestionPipeline
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.bm25_retriever import BM25KeywordRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.graph.nodes.retrieve import set_default_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path("data/sample_docs")
    if data_dir.exists():
        pipeline = DocumentIngestionPipeline()
        ingest_res = pipeline.ingest_directory(data_dir)
        vs = ChromaVectorStore()
        if vs.count() == 0 and ingest_res.chunks:
            vs.add_chunks(ingest_res.chunks)
        bm25 = BM25KeywordRetriever(chunks=ingest_res.chunks)
        hybrid = HybridRetriever(vector_store=vs, bm25_retriever=bm25)
        set_default_retriever(hybrid)
    yield


app = FastAPI(
    title="Corrective RAG (CRAG) Service",
    description="Production Agentic Microservice with LangGraph, Hybrid Retrieval, Guardrails, and Fallback Search",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(router)

