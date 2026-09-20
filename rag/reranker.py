"""
Cross-Encoder Reranker
----------------------

Reranks already retrieved chunks using a cross-encoder.

IMPORTANT:
This implementation deliberately preserves ALL original
chunk metadata and text.
"""

from pathlib import Path
from typing import List, Dict

from sentence_transformers import CrossEncoder


# ============================================================
# CONFIG
# ============================================================

RERANKER_MODEL = "BAAI/bge-reranker-base"

DEFAULT_CANDIDATE_K = 10
DEFAULT_FINAL_K = 5


# ============================================================
# RERANKER
# ============================================================

class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        batch_size: int = 16,
    ):

        print(
            f"Loading reranker model: {model_name}"
        )

        self.model_name = model_name

        self.batch_size = batch_size

        self.model = CrossEncoder(
            model_name
        )

        print("Reranker model loaded.")

    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = DEFAULT_FINAL_K,
    ):

        if not candidates:
            return []

        # ----------------------------------------------------
        # Make sure candidates contain text
        # ----------------------------------------------------

        valid_candidates = []

        for candidate in candidates:

            text = candidate.get(
                "text",
                ""
            )

            if not text:
                continue

            valid_candidates.append(candidate)

        if not valid_candidates:
            return []

        # ----------------------------------------------------
        # Prepare pairs
        # ----------------------------------------------------

        pairs = [
            (
                query,
                candidate["text"],
            )
            for candidate in valid_candidates
        ]

        print(
            f"Reranking {len(pairs)} candidates..."
        )

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # ----------------------------------------------------
        # CRITICAL:
        # Keep the complete original candidate.
        # ----------------------------------------------------

        reranked = []

        for candidate, score in zip(
            valid_candidates,
            scores,
        ):

            result = dict(candidate)

            result["reranker_score"] = float(score)

            result["reranked"] = True

            reranked.append(result)

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        reranked.sort(
            key=lambda x: x["reranker_score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # Assign final reranker rank
        # ----------------------------------------------------

        final_results = []

        for rank, result in enumerate(
            reranked[:top_k],
            start=1,
        ):

            result = dict(result)

            result["reranker_rank"] = rank

            final_results.append(result)

        return final_results

    # ========================================================
    # RETRIEVE + RERANK
    # ========================================================

    def retrieve_and_rerank(
        self,
        retriever,
        query: str,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        final_k: int = DEFAULT_FINAL_K,
        domain=None,
        document_type=None,
    ):

        candidates = retriever.retrieve(
            query=query,
            top_k=candidate_k,
            domain=domain,
            document_type=document_type,
        )

        return self.rerank(
            query=query,
            candidates=candidates,
            top_k=final_k,
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    from rag.retriever import HybridRetriever

    retriever = HybridRetriever()

    reranker = CrossEncoderReranker()

    query = "What payment methods are available?"

    results = reranker.retrieve_and_rerank(
        retriever=retriever,
        query=query,
        candidate_k=10,
        final_k=5,
        domain="payments",
    )

    print("\n" + "=" * 80)

    print("RERANKER TEST")

    print("=" * 80)

    for result in results:

        print(
            f"\nRank: "
            f"{result.get('reranker_rank')}"
        )

        print(
            f"Chunk: "
            f"{result.get('chunk_id')}"
        )

        print(
            f"Document: "
            f"{result.get('file_name')}"
        )

        print(
            f"Domain: "
            f"{result.get('domain')}"
        )

        print(
            f"Section: "
            f"{result.get('section')}"
        )

        print(
            f"Score: "
            f"{result.get('reranker_score')}"
        )

        print(
            f"TEXT: "
            f"{result.get('text', '')[:300]}"
        )