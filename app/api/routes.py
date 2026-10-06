import json
import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.schemas.api import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from app.graph.workflow import create_crag_graph
from app.graph.state import GraphState

router = APIRouter()
_graph_instance = None


def get_graph():
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = create_crag_graph()
    return _graph_instance


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    provider = settings.llm_provider.lower()
    model = (
        settings.gemini_model_name
        if provider == "gemini"
        else settings.openai_model_name
    )
    return HealthResponse(
        status="healthy",
        llm_provider=provider,
        model_name=model,
        version="0.1.0",
    )


@router.post("/query", response_model=QueryResponse, tags=["CRAG Pipeline"])
async def query_pipeline(request: QueryRequest) -> QueryResponse:
    graph = get_graph()

    initial_state: GraphState = {
        "question": request.question,
        "documents": [],
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution error: {str(exc)}"
        )

    raw_docs = final_state.get("documents", [])
    sources = [
        SourceDocument(
            chunk_id=d.chunk_id,
            source=d.metadata.source,
            snippet=d.content.replace("\n", " ").strip()[:200],
            retrieval_method=d.retrieval_method,
            score=d.score,
        )
        for d in raw_docs
    ]

    has_web_source = any(d.retrieval_method == "web" for d in raw_docs)

    return QueryResponse(
        question=request.question,
        answer=final_state.get("generation", "No generation produced."),
        sources=sources,
        web_search_triggered=has_web_source,
        retry_count=final_state.get("retry_count", 0),
    )


async def sse_event_generator(question: str) -> AsyncGenerator[str, None]:
    graph = get_graph()

    yield f"data: {json.dumps({'event': 'step', 'message': 'Executing CRAG workflow'})}\n\n"
    await asyncio.sleep(0.01)

    initial_state: GraphState = {
        "question": question,
        "documents": [],
        "web_search_needed": False,
        "generation": None,
        "retry_count": 0,
        "search_query": None,
    }

    final_state = graph.invoke(initial_state)

    answer = final_state.get("generation", "")
    words = answer.split(" ")

    for word in words:
        yield f"data: {json.dumps({'event': 'token', 'token': word + ' '})}\n\n"
        await asyncio.sleep(0.02)

    raw_docs = final_state.get("documents", [])
    has_web = any(d.retrieval_method == "web" for d in raw_docs)

    completion_payload = {
        "event": "done",
        "web_search_triggered": has_web,
        "sources_count": len(raw_docs),
        "retry_count": final_state.get("retry_count", 0),
    }
    yield f"data: {json.dumps(completion_payload)}\n\n"


@router.post("/query/stream", tags=["CRAG Pipeline"])
async def stream_query(request: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        sse_event_generator(request.question),
        media_type="text/event-stream",
    )

