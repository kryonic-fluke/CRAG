import re
from typing import List, Optional
from rank_bm25 import BM25Okapi

from app.schemas.document import DocumentChunk
from app.schemas.retrieval import SearchResult


class BM25KeywordRetriever:
    """
    BM25 (Best Matching 25) Sparse Lexical Retriever.

    Why this is essential alongside Vector Search:
    - Dense vectors map sentences to semantic "concepts", but often miss exact keyword hits:
      e.g., "$1,500", "PTO", "Node.js", specific error codes, or function names.
    - BM25 scores documents based on exact term frequency (TF) and inverse document frequency (IDF).
    """

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None) -> None:
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

        if chunks:
            self.index_chunks(chunks)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text into lowercase alphanumeric words.
        Handles punctuation gracefully (e.g., '$1,500' -> ['1', '500']).
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Builds the BM25 index over a list of DocumentChunks.
        """
        if not chunks:
            return 0

        self.chunks = list(chunks)
        self.tokenized_corpus = [self.tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        return len(self.chunks)

    def similarity_search(self, query: str, k: int = 4) -> List[SearchResult]:
        """
        Queries the BM25 index and returns top-k results formatted as SearchResults.
        Scores are normalized to [0.0, 1.0] by dividing by the maximum score in the set.
        """
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        raw_scores = self.bm25.get_scores(tokenized_query)
        max_score = float(max(raw_scores)) if len(raw_scores) > 0 else 0.0

        scored_candidates = []
        for chunk, raw_score in zip(self.chunks, raw_scores):
            if raw_score > 0:
                norm_score = (raw_score / max_score) if max_score > 0 else 0.0
                scored_candidates.append((chunk, norm_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        top_candidates = scored_candidates[:k]

        results: List[SearchResult] = []
        for rank_idx, (chunk, score) in enumerate(top_candidates, start=1):
            res = SearchResult(
                chunk_id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=round(score, 4),
                retrieval_method="bm25",
                rank=rank_idx,
            )
            results.append(res)

        return results

    def count(self) -> int:
        return len(self.chunks)
