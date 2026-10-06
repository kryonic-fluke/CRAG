from pathlib import Path
from typing import List, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.schemas.document import DocumentChunk, DocumentMetadata, IngestConfig, IngestResult


class DocumentIngestionPipeline:
    """
    Production-grade document ingestion pipeline.
    
    Responsibilities:
    1. Loads raw documents from disk or string sources.
    2. Splits them intelligently using RecursiveCharacterTextSplitter (respecting paragraphs, newlines, sentences).
    3. Transforms raw LangChain chunks into validated Pydantic `DocumentChunk` models.
    4. Guarantees deterministic chunk IDs and rich metadata tracking.
    """

    def __init__(self, config: Union[IngestConfig, None] = None) -> None:
        if config is None:
            self.config = IngestConfig(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
        else:
            self.config = config

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_file(self, file_path: Union[str, Path]) -> str:
        """Reads a UTF-8 text or markdown file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found at: {file_path}")
        return path.read_text(encoding="utf-8")

    def process_text(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Splits raw text and converts each slice into a validated Pydantic DocumentChunk.
        """
        raw_chunks = self.text_splitter.split_text(text)
        validated_chunks: List[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            metadata = DocumentMetadata(
                source=source,
                chunk_index=idx,
                char_count=len(chunk_text),
                extra={"total_candidate_chunks": str(len(raw_chunks))}
            )

            # Creating the Pydantic model executes runtime validation!
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=metadata
            )
            validated_chunks.append(chunk)

        return validated_chunks

    def ingest_directory(self, dir_path: Union[str, Path], extensions: List[str] = [".txt", ".md"]) -> IngestResult:
        """
        Discovers all matching text documents in a directory and produces an IngestResult.
        """
        directory = Path(dir_path)
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"Provided path is not a valid directory: {dir_path}")

        all_chunks: List[DocumentChunk] = []
        loaded_sources: List[str] = []

        for ext in extensions:
            for file_path in directory.glob(f"*{ext}"):
                text = self.load_file(file_path)
                source_name = file_path.name
                chunks = self.process_text(text=text, source=source_name)
                
                all_chunks.extend(chunks)
                loaded_sources.append(source_name)

        return IngestResult(
            total_documents_loaded=len(loaded_sources),
            total_chunks_created=len(all_chunks),
            sources=loaded_sources,
            chunks=all_chunks
        )

