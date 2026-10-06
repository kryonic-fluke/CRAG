import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path so 'app' is importable when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pydantic import ValidationError

from app.core.config import settings
from app.schemas.document import DocumentChunk, DocumentMetadata, IngestConfig
from app.retrieval.document_loader import DocumentIngestionPipeline


def run_demo() -> None:
    print("=" * 60)
    print("[*] DAY 1 DEMO: DOCUMENT INGESTION & PYDANTIC VALIDATION")
    print("=" * 60)


    # 1. Inspect Settings loaded from pydantic-settings
    print(f"\n[1] Application Settings (via pydantic-settings):")
    print(f"    - LLM Model: {settings.openai_model_name}")
    print(f"    - Default Chunk Size: {settings.chunk_size}")
    print(f"    - Default Overlap: {settings.chunk_overlap}")
    print(f"    - Chroma Directory: {settings.chroma_persist_directory}")

    # 2. Ingest Sample Documents
    data_dir = Path("data/sample_docs")
    print(f"\n[2] Ingesting documents from: {data_dir.resolve()}")
    
    config = IngestConfig(chunk_size=400, chunk_overlap=80)
    pipeline = DocumentIngestionPipeline(config=config)
    result = pipeline.ingest_directory(dir_path=data_dir)

    print(f"\n[3] Ingestion Summary:")
    print(f"    - Total Documents Loaded: {result.total_documents_loaded}")
    print(f"    - Sources: {result.sources}")
    print(f"    - Total Chunks Created: {result.total_chunks_created}")

    # 3. Inspect Individual Validated Chunks
    print(f"\n[4] Sample Validated Chunks (First 3):")
    for i, chunk in enumerate(result.chunks[:3]):
        print(f"\n    --- Chunk #{i+1} ---")
        print(f"    ID:          {chunk.id}")
        print(f"    Source:      {chunk.metadata.source}")
        print(f"    Index:       {chunk.metadata.chunk_index}")
        print(f"    Length:      {chunk.metadata.char_count} chars")
        print(f"    Created At:  {chunk.metadata.created_at.isoformat()}")
        preview = chunk.content.replace('\n', ' ')[:100]
        print(f"    Content:     \"{preview}...\"")

    # 4. Demonstrate Pydantic Validation Protection
    print("\n" + "=" * 60)
    print("[*] DEMONSTRATING PYDANTIC DEFENSIVE VALIDATION")
    print("=" * 60)

    # Demo 4A: What happens if an empty string or whitespace tries to enter?
    print("\n[Test A] Attempting to create DocumentChunk with empty content:")
    try:
        DocumentChunk(
            content="   ",
            metadata=DocumentMetadata(source="test.txt", chunk_index=0, char_count=0)
        )
    except ValidationError as e:
        print("  -> Caught expected ValidationError:")
        for err in e.errors():
            print(f"     Field: {err['loc']} | Error: {err['msg']}")

    # Demo 4B: What happens if overlap >= chunk_size?
    print("\n[Test B] Attempting to configure overlap (500) >= chunk_size (300):")
    try:
        IngestConfig(chunk_size=300, chunk_overlap=500)
    except ValidationError as e:
        print("  -> Caught expected ValidationError:")
        for err in e.errors():
            print(f"     Field: {err['loc']} | Error: {err['msg']}")

    print("\n" + "=" * 60)
    print("[OK] DAY 1 PIPELINE VERIFIED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()

