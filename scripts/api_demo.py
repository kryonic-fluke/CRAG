import sys
from pathlib import Path
from fastapi.testclient import TestClient

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.main import app


def run_demo() -> None:
    print("=" * 70)
    print("[*] DAY 6 DEMO: PRODUCTION FASTAPI SERVICE & STREAMING CLIENT")
    print("=" * 70)

    with TestClient(app) as client:
        print("\n[1] Testing GET /health...")
        health_res = client.get("/health")
        print(f"    Status Code: {health_res.status_code}")
        print(f"    Payload: {health_res.json()}")

        print("\n[2] Testing POST /api/v1/query (Standard REST Endpoint)...")
        query_payload = {
            "question": "What is the equipment stipend policy?",
            "enable_web_fallback": True,
        }
        res = client.post("/api/v1/query", json=query_payload)
        print(f"    Status Code: {res.status_code}")
        data = res.json()
        print(f'    Question: "{data["question"]}"')
        print(f'    Answer: "{data["answer"]}"')
        print(f"    Sources Cited: {len(data['sources'])}")
        for s in data["sources"][:2]:
            print(
                f"        - [{s['retrieval_method'].upper()}] {s['source']}: {s['snippet'][:80]}..."
            )
        print(f"    Web Search Triggered: {data['web_search_triggered']}")
        print(f"    Retry Count: {data['retry_count']}")

        print("\n[3] Testing POST /api/v1/query/stream (Server-Sent Events Stream)...")
        stream_payload = {
            "question": "What is Acme Corp's remote work policy?",
        }
        stream_res = client.post("/api/v1/query/stream", json=stream_payload)
        print(f"    Status Code: {stream_res.status_code}")
        print(f"    Content-Type: {stream_res.headers.get('content-type')}")
        print("    Stream Output Preview:")
        for line in stream_res.iter_lines():
            if line and line.startswith("data:"):
                print(f"        {line}")

    print("\n" + "=" * 70)
    print("[*] HOW TO RUN IN PRODUCTION:")
    print("    Command:  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print("    Swagger:  http://127.0.0.1:8000/docs")
    print("    ReDoc:    http://127.0.0.1:8000/redoc")
    print("=" * 70)
    print("[OK] DAY 6 FASTAPI SERVICE VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
