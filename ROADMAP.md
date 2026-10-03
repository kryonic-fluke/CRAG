# 🗺️ 6-Day CRAG Build Roadmap (`ROADMAP.md`)

> **Architecture:** Corrective RAG (CRAG) with LangGraph, Hybrid Search, Guardrails, Pydantic, and FastAPI.  
> **Goal:** Production-ready AI pipeline with observable decision-making, fallback search, hallucination defense, and measurable evaluation metrics.

---

```
┌────────────────────────────────────────────────────────┐
│                      USER QUERY                        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   HYBRID RETRIEVAL                     │
│               (Dense Vector + Sparse BM25)             │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               DOCUMENT RELEVANCE GRADER                │ ◄─── LangGraph Node (Pydantic Schema)
│            (Evaluates relevance of each chunk)         │
└─────────────┬────────────────────────────┬─────────────┘
              │ [Relevant Docs Found]      │ [No Relevant Docs / Uncertain]
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│    AUGMENT & GENERATE     │  │  WEB SEARCH FALLBACK    │
│  (Context-backed answer)  │  │ (Tavily API / Scraper)  │
└─────────────┬─────────────┘  └───────────┬─────────────┘
              │                            │
              ▼                            │
┌───────────────────────────┐              │
│    HALLUCINATION CHECK    │ ◄────────────┘
│  (Guardrail / Faithfulness│
│   validation via LLM)     │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│      FINAL RESPONSE       │
└───────────────────────────┘
```

---

## 📅 Day-by-Day Implementation Plan

### **Day 1: Project Architecture, Environment & Pydantic Foundations**
- **Learning Spotlight:**
  - *What is Pydantic?* Why production AI uses Pydantic schemas instead of raw Python dictionaries (data validation, automatic serialization, type safety).
  - *Environment configuration* with `pydantic-settings` to manage API keys and configs cleanly.
- **Hands-on Deliverables:**
  - Initialize clean virtual environment (`py -3 -m venv .venv`).
  - Set up directory structure ([`app/`](file:///app/), [`data/`](file:///data/), [`tests/`](file:///tests/)).
  - Define `DocumentChunk` and `IngestConfig` schemas in Pydantic.
  - Implement document loading and chunking pipeline (PDF, Markdown, Text).
- **Verification:** Run ingestion script on sample data; verify chunks have valid schemas and metadata.

---

### **Day 2: Hybrid Retrieval (Dense Vector + BM25 Ensemble)**
- **Learning Spotlight:**
  - *Dense vs. Sparse Retrieval:* Why semantic vector search fails on acronyms, part numbers, or exact names, and why BM25 fixes it.
  - *Reciprocal Rank Fusion (RRF):* Combining score-based dense rankings with frequency-based sparse rankings.
- **Hands-on Deliverables:**
  - Implement ChromaDB vector store for dense embeddings.
  - Implement BM25 retriever for lexical/keyword search.
  - Build `HybridRetriever` unifying both via LangChain `EnsembleRetriever` with weighted rankings.
- **Verification:** Run side-by-side queries showing how BM25 retrieves exact technical keywords where pure vector search fails.

---

### **Day 3: LangGraph State Machine & Document Relevance Grader**
- **Learning Spotlight:**
  - *LangGraph State Machine Anatomy:* Nodes, Edges, State Transitions, and conditional routing.
  - *Structured Outputs via Pydantic:* Instructing LLMs to strictly return `{ "binary_score": "yes" | "no", "reasoning": "..." }`.
- **Hands-on Deliverables:**
  - Define `GraphState` containing: `question`, `documents`, `generation`, `web_search_needed`, `retry_count`.
  - Build **Node 1: Retriever Node** (fetches hybrid documents).
  - Build **Node 2: Document Relevance Grader** (scores each document chunk).
  - Implement Conditional Edge: If any valid documents survive grading $\rightarrow$ proceed; if all documents are irrelevant $\rightarrow$ trigger web search.
- **Verification:** Unit test the grader node with both on-topic and distractor documents to verify correct filtering and routing.

---

### **Day 4: Web Search Fallback & Augment-and-Generate**
- **Learning Spotlight:**
  - *Corrective RAG (CRAG) Philosophy:* What to do when internal knowledge base fails.
  - *Tavily Search API:* Clean web search designed specifically for LLM agents.
- **Hands-on Deliverables:**
  - Build **Node 3: Web Search Node** (invoked conditionally when retrieved docs are irrelevant).
  - Build **Node 4: Generate Node** (context-augmented prompt synthesizing answers from verified documents).
  - Connect full LangGraph loop from User Query $\rightarrow$ Retrieval $\rightarrow$ Grader $\rightarrow$ (Internal Doc OR Web Search) $\rightarrow$ Generation.
- **Verification:** Test query with known internal fact (uses internal doc) vs. unknown external event (triggers Tavily and returns fresh answer).

---

### **Day 5: Hallucination Guardrail & Self-Correction Feedback Loop**
- **Learning Spotlight:**
  - *Guardrails & Faithfulness:* Detecting when the LLM generates unsupported assertions.
  - *Stateful Retry Loops:* How LangGraph allows self-correction with bounded retry limits (preventing infinite loops).
- **Hands-on Deliverables:**
  - Build **Node 5: Hallucination Grader** (validates if generation is grounded in the source context).
  - Build **Node 6: Answer Relevance Grader** (validates if the generation actually addresses the user's initial question).
  - Implement conditional loop:
    - Grounded + Relevant $\rightarrow$ Output response.
    - Hallucinated $\rightarrow$ Re-attempt generation (max 2 retries).
    - Irrelevant $\rightarrow$ Re-query / route to fallback.
- **Verification:** Trigger an adversarial prompt to observe the state machine catch the hallucination and correct itself.

---

### **Day 6: Production FastAPI Service & Evaluation Suite**
- **Learning Spotlight:**
  - *FastAPI Fundamentals:* Async request lifecycle, Pydantic request/response validation, automatic OpenAPI / Swagger documentation (`/docs`).
  - *RAG Evaluation:* How to quantitatively measure RAG quality (Faithfulness, Answer Relevance).
- **Hands-on Deliverables:**
  - Create FastAPI application with endpoints:
    - `POST /api/v1/query`: Standard synchronous query endpoint.
    - `POST /api/v1/query/stream`: Server-Sent Events (SSE) streaming tokens.
    - `GET /api/v1/health`: Healthcheck & model readiness.
  - Build Evaluation Suite ([`eval/eval_suite.py`](file:///eval/eval_suite.py)):
    - 15 benchmark test queries (5 in-domain, 5 out-of-domain, 5 tricky/adversarial).
    - Calculate and print aggregate Faithfulness Score (%) and Answer Relevance Score (%).
- **Verification:** Run FastAPI server with `uvicorn`, query via Swagger UI (`http://127.0.0.1:8000/docs`), and run the eval suite.

---

## 📌 Progress Tracker

- [ ] **Day 1:** Project Setup, Ingestion & Pydantic Basics
- [ ] **Day 2:** Hybrid Retrieval (Dense Vector + BM25)
- [ ] **Day 3:** LangGraph State & Document Grader Node
- [ ] **Day 4:** Web Search Fallback & Generation Node
- [ ] **Day 5:** Hallucination Guardrail & Self-Correction
- [ ] **Day 6:** FastAPI Endpoints & Eval Suite
