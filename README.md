# Corrective RAG (CRAG) Service

An industry-standard, production-grade Corrective RAG (CRAG) system implemented as an autonomous agentic microservice.

## 🏗️ Architecture Overview

- **Orchestration:** LangGraph (Stateful workflow with conditional routing and self-correction loops)
- **Retrieval:** Hybrid Retrieval (ChromaDB Dense Vectors + BM25 Lexical Keyword Search)
- **API & Serving:** FastAPI (REST + Server-Sent Events token streaming)
- **Validation:** Pydantic (Strict data schemas for state, grader outputs, and API payloads)
- **Fallbacks:** Tavily Web Search API
- **Guardrails:** Hallucination and groundedness evaluation nodes
- **Evaluation:** Quantitative test suite measuring Faithfulness & Relevance

## 🔄 Pipeline Workflow

1. **Hybrid Retrieval:** Dense vector search (ChromaDB) fused with sparse lexical search (BM25).
2. **Relevance Grading:** LLM node grades retrieved documents and filters out noise.
3. **Adaptive Routing:** If documents are insufficient, automatically routes to Web Search (Tavily).
4. **Answer Generation:** Context-augmented prompt synthesizes response.
5. **Guardrail Validation:** Groundedness checker verifies generation against source context before returning.

