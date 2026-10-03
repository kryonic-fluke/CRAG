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

## 📖 Documentation

- [Project Architecture & System Design](ARCHITECTURE_EXPLAINED.md)
- [6-Day Build Roadmap](ROADMAP.md)
- [Engineering Guidelines & Rules](rules.md)
