from typing import Dict, List, Optional
from app.retrieval.bm25_retriever import BM25KeywordRetriever
from app.retrieval.vector_store import ChromaVectorStore
from app.schemas.document import DocumentChunk
from app.schemas.retrieval import HybridRetrieverConfig, RetrievalResponse, SearchResult


class HybridRetriever:
    """
    Production Hybrid Retriever fusing Dense Semantic Vector Search (Chroma)
    and Sparse Lexical Keyword Search (BM25).

    Supports:
    1. Reciprocal Rank Fusion (RRF): Rank-based fusion robust against scale differences.
    2. Weighted Score Fusion: Linear combination of normalized similarity scores.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_retriever: BM25KeywordRetriever,
        config: Optional[HybridRetrieverConfig] = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.config = config or HybridRetrieverConfig()

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Indexes chunks synchronously into both ChromaDB and the BM25 index.
        """
        self.vector_store.add_chunks(chunks)
        self.bm25_retriever.index_chunks(chunks)
        return len(chunks)

    def _reciprocal_rank_fusion(
        self, dense_results: List[SearchResult], bm25_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Computes RRF score for each unique document:
        RRF(d) = sum_{method} [ weight_{method} / (rrf_k + rank_{method}(d)) ]
        """
        k = self.config.rrf_k
        w_dense = self.config.dense_weight
        w_bm25 = self.config.bm25_weight

        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, SearchResult] = {}

        for result in dense_results:
            cid = result.chunk_id
            rank = result.rank or 1
            score_contribution = w_dense * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contribution
            candidate_map[cid] = result

        for result in bm25_results:
            cid = result.chunk_id
            rank = result.rank or 1
            score_contribution = w_bm25 * (1.0 / (k + rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contribution
            if cid not in candidate_map:
                candidate_map[cid] = result

        if not rrf_scores:
            return []

        max_rrf = max(rrf_scores.values())

        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        final_results: List[SearchResult] = []
        for new_rank, cid in enumerate(sorted_ids[: self.config.top_k], start=1):
            original = candidate_map[cid]
            normalized_score = (rrf_scores[cid] / max_rrf) if max_rrf > 0 else 0.0

            hybrid_result = SearchResult(
                chunk_id=original.chunk_id,
                content=original.content,
                metadata=original.metadata,
                score=round(normalized_score, 4),
                retrieval_method="hybrid",
                rank=new_rank,
            )
            final_results.append(hybrid_result)

        return final_results

    def _weighted_score_fusion(
        self, dense_results: List[SearchResult], bm25_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Computes weighted linear score fusion:
        Score(d) = w_dense * DenseScore(d) + w_bm25 * BM25Score(d)
        """
        w_dense = self.config.dense_weight
        w_bm25 = self.config.bm25_weight
        total_w = w_dense + w_bm25

        scores: Dict[str, float] = {}
        candidate_map: Dict[str, SearchResult] = {}

        for res in dense_results:
            cid = res.chunk_id
            scores[cid] = scores.get(cid, 0.0) + (w_dense * res.score)
            candidate_map[cid] = res

        for res in bm25_results:
            cid = res.chunk_id
            scores[cid] = scores.get(cid, 0.0) + (w_bm25 * res.score)
            if cid not in candidate_map:
                candidate_map[cid] = res

        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        final_results: List[SearchResult] = []
        for new_rank, cid in enumerate(sorted_ids[: self.config.top_k], start=1):
            original = candidate_map[cid]
            blended_score = scores[cid] / total_w if total_w > 0 else 0.0

            hybrid_result = SearchResult(
                chunk_id=original.chunk_id,
                content=original.content,
                metadata=original.metadata,
                score=round(min(1.0, blended_score), 4),
                retrieval_method="hybrid",
                rank=new_rank,
            )
            final_results.append(hybrid_result)

        return final_results

    def retrieve(self, query: str) -> RetrievalResponse:
        """
        Executes hybrid retrieval: fetches dense + sparse candidates and fuses them.
        """

        pool_size = max(self.config.top_k * 2, 8)

        dense_results = self.vector_store.similarity_search(query=query, k=pool_size)
        bm25_results = self.bm25_retriever.similarity_search(query=query, k=pool_size)

        if self.config.fusion_strategy == "rrf":
            fused = self._reciprocal_rank_fusion(dense_results, bm25_results)
            strategy_desc = f"Hybrid RRF (k={self.config.rrf_k}, dense_w={self.config.dense_weight}, bm25_w={self.config.bm25_weight})"
        else:
            fused = self._weighted_score_fusion(dense_results, bm25_results)
            strategy_desc = f"Hybrid Weighted Score (dense_w={self.config.dense_weight}, bm25_w={self.config.bm25_weight})"

        return RetrievalResponse(
            query=query,
            total_results=len(fused),
            results=fused,
            strategy_used=strategy_desc,
        )
