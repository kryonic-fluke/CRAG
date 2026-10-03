# 📜 CRAG Project Rules & Engineering Guidelines (`rules.md`)

> **Project:** Production Corrective RAG (CRAG) with LangGraph, Hybrid Retrieval, Guardrails & FastAPI  
> **Target Duration:** 6-Day Build  
> **Target Audience:** Intermediate RAG/LangGraph engineer leveling up to Production-grade AI with Pydantic & FastAPI.

---

## 1. Core Engineering Principles

1. **Explain the "Why" Before the "What" (First-Principles Learning)**
   - Because Pydantic and FastAPI are new, every introduction of schemas, dependency injection, validation, or routers must include a concise, real-world explanation ("Why raw Python dicts fail in production", "Why FastAPI async endpoints matter").
   - Code must be accompanied by intuitive mental models and diagrams, not just code dumps.

2. **No "Magic Dictionaries" — Schema First via Pydantic**
   - No untyped dictionaries passing through nodes.
   - All graph states, LLM structured outputs (graders), and API request/response payloads must be explicitly modeled using Pydantic `BaseModel` or typed LangGraph `TypedDict` with runtime validation.

3. **Incremental Verification (Build, Run, Verify)**
   - Never write more than one logical component without running an execution test.
   - Every node in the LangGraph state machine must be testable in isolation before being linked into the workflow.

4. **Defensive Error Handling & Observability**
   - External APIs (OpenAI/Anthropic/Groq, Tavily, Chroma) must have fallback paths, timeout limits, and clear error logs.
   - LangGraph nodes must always return clean state updates, never unhandled exceptions that crash the pipeline.

5. **Windows-Native Shell & Tooling Compatibility**
   - All CLI instructions must be valid for Windows PowerShell (e.g., using `py -3` launcher, `.\.venv\Scripts\Activate.ps1`, handling backslashes vs forward slashes in file paths).

---

## 2. Code Quality & Architecture Standards

- **Project Structure**:
  ```text
  CRAG/
  ├── app/
  │   ├── api/              # FastAPI endpoints & routes
  │   ├── core/             # Configuration (pydantic-settings), logging
  │   ├── graph/            # LangGraph workflow, state, nodes, edges
  │   │   ├── nodes/        # Individual isolated node functions
  │   │   ├── state.py      # LangGraph GraphState definition
  │   │   ├── workflow.py   # StateGraph compilation & edge routing
  │   │   └── chains/       # Prompt templates & LLM grader chains
  │   ├── retrieval/        # Hybrid retriever (Dense Chroma + Sparse BM25)
  │   └── schemas/          # Pydantic models for API & LLM outputs
  ├── data/                 # Sample documents & knowledge base
  ├── eval/                 # Evaluation suite (Ragas / LLM-as-a-judge)
  ├── tests/                # Unit & integration tests
  ├── .env.example          # Environment variables template
  ├── requirements.txt      # Pinned project dependencies
  ├── rules.md              # Engineering rules (this file)
  └── ROADMAP.md            # The 6-Day step-by-step build plan
  ```

- **Clean Python Standards**:
  - Python 3.12+ compatible.
  - Strict type hints on every function parameter and return type (`def grade_document(state: GraphState) -> Dict[str, Any]:`).
  - Docstrings explaining input state mutations and edge routing conditions.

---

## 3. Communication & Pair-Programming Protocol

- **Code Delivery Format**:
  - Changes will be delivered step-by-step with file links (e.g., [`state.py`](file:///app/graph/state.py)).
  - Important concepts will be highlighted using GitHub-style callouts (`> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`).
- **Interactive Checkpoints**:
  - At the end of each milestone, we verify the output together before advancing to the next day's task.

---

## 4. User Custom Rules (Add Your Rules Below)

> *Feel free to edit, add, or amend any personal preferences, hardware constraints, API providers, or formatting rules below:*

- [ ] *Example: Preferred LLM Provider (e.g., OpenAI gpt-4o-mini, Groq Llama 3.3, Ollama local)*
- [ ] *Example: Preferred Embedding Model (e.g., text-embedding-3-small, HuggingFace BAAI/bge-small-en-v1.5)*
- [ ] *Example: Specific documents or domain for testing (e.g., Financial reports, AI research papers, Internal docs)*
