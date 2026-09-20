"""
Hybrid Retriever
----------------
Combines:
1. FAISS semantic retrieval
2. BM25 lexical retrieval
3. Reciprocal Rank Fusion (RRF)

Supports metadata filtering by:
- domain
- document_type
"""

from pathlib import Path
import json
import pickle
import re
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "storage" / "chunks" / "chunks.json"
FAISS_INDEX_FILE = PROJECT_ROOT / "storage" / "faiss" / "index.faiss"
BM25_INDEX_FILE = PROJECT_ROOT / "storage" / "bm25" / "index.pkl"


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_TOP_K = 10

RRF_K = 60


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for BM25 and query processing.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str):
    """
    Tokenize normalized text.
    """

    return normalize_text(text).split()


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(
        self,
        chunks_file: Path = CHUNKS_FILE,
        faiss_index_file: Path = FAISS_INDEX_FILE,
        bm25_index_file: Path = BM25_INDEX_FILE,
        embedding_model: str = EMBEDDING_MODEL,
    ):

        print("Initializing HybridRetriever...")

        self.chunks_file = Path(chunks_file)
        self.faiss_index_file = Path(faiss_index_file)
        self.bm25_index_file = Path(bm25_index_file)

        # ----------------------------------------------------
        # Load chunks
        # ----------------------------------------------------

        with open(self.chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if not isinstance(self.chunks, list):
            raise ValueError("chunks.json must contain a list.")

        # ----------------------------------------------------
        # Create chunk lookup
        # ----------------------------------------------------

        self.chunk_map = {
            chunk["chunk_id"]: chunk
            for chunk in self.chunks
            if "chunk_id" in chunk
        }

        # ----------------------------------------------------
        # Load FAISS
        # ----------------------------------------------------

        print(f"Loading embedding model: {embedding_model}")

        self.embedding_model_name = embedding_model

        self.embedding_model = SentenceTransformer(
            embedding_model
        )

        self.faiss_index = faiss.read_index(
            str(self.faiss_index_file)
        )

        # ----------------------------------------------------
        # Load BM25
        # ----------------------------------------------------

        with open(self.bm25_index_file, "rb") as f:
            bm25_data = pickle.load(f)

        # Support both dictionary and direct-object formats
        if isinstance(bm25_data, dict):

            self.bm25 = bm25_data.get("bm25")

            if self.bm25 is None:
                self.bm25 = bm25_data.get("index")

            self.bm25_chunk_ids = bm25_data.get(
                "chunk_ids",
                [chunk["chunk_id"] for chunk in self.chunks]
            )

        else:

            self.bm25 = bm25_data

            self.bm25_chunk_ids = [
                chunk["chunk_id"]
                for chunk in self.chunks
            ]

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if self.faiss_index.ntotal != len(self.chunks):

            raise ValueError(
                f"FAISS index contains {self.faiss_index.ntotal} "
                f"vectors but chunks.json contains "
                f"{len(self.chunks)} chunks."
            )

        if len(self.bm25_chunk_ids) != len(self.chunks):

            raise ValueError(
                "BM25 chunk mapping does not match chunks.json."
            )

        print(
            f"HybridRetriever ready — "
            f"{len(self.chunks)} chunks loaded."
        )

    # ========================================================
    # FILTER CHUNKS
    # ========================================================

    def _allowed_chunk_ids(
        self,
        domain: Optional[str] = None,
        document_type: Optional[str] = None,
    ):

        allowed = []

        for chunk in self.chunks:

            if domain:

                if (
                    chunk.get("domain", "").lower()
                    != domain.lower()
                ):
                    continue

            if document_type:

                if (
                    chunk.get("document_type", "").lower()
                    != document_type.lower()
                ):
                    continue

            allowed.append(chunk["chunk_id"])

        return set(allowed)

    # ========================================================
    # FAISS SEARCH
    # ========================================================

    def _faiss_search(
        self,
        query: str,
        top_k: int,
        allowed_ids=None,
    ):

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # Retrieve a larger candidate pool so metadata filtering
        # doesn't accidentally eliminate everything.
        search_k = min(
            max(top_k * 5, 20),
            len(self.chunks)
        )

        scores, indices = self.faiss_index.search(
            query_embedding,
            search_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0:
                continue

            chunk = self.chunks[index]

            chunk_id = chunk["chunk_id"]

            if allowed_ids is not None:
                if chunk_id not in allowed_ids:
                    continue

            results.append(
                {
                    "chunk_id": chunk_id,
                    "faiss_score": float(score),
                }
            )

            if len(results) >= top_k:
                break

        return results

    # ========================================================
    # BM25 SEARCH
    # ========================================================

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        allowed_ids=None,
    ):

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices:

            chunk_id = self.bm25_chunk_ids[index]

            if allowed_ids is not None:
                if chunk_id not in allowed_ids:
                    continue

            results.append(
                {
                    "chunk_id": chunk_id,
                    "bm25_score": float(scores[index]),
                }
            )

            if len(results) >= top_k:
                break

        return results

    # ========================================================
    # RRF
    # ========================================================

    def _rrf_fusion(
        self,
        faiss_results,
        bm25_results,
    ):

        scores = {}

        faiss_rank = {}

        bm25_rank = {}

        for rank, result in enumerate(
            faiss_results,
            start=1,
        ):

            faiss_rank[result["chunk_id"]] = rank

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):

            bm25_rank[result["chunk_id"]] = rank

        all_ids = set(
            faiss_rank
        ).union(
            bm25_rank
        )

        for chunk_id in all_ids:

            score = 0.0

            if chunk_id in faiss_rank:

                score += 1.0 / (
                    RRF_K + faiss_rank[chunk_id]
                )

            if chunk_id in bm25_rank:

                score += 1.0 / (
                    RRF_K + bm25_rank[chunk_id]
                )

            scores[chunk_id] = score

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked

    # ========================================================
    # PUBLIC RETRIEVE
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        domain: Optional[str] = None,
        document_type: Optional[str] = None,
    ):

        allowed_ids = self._allowed_chunk_ids(
            domain=domain,
            document_type=document_type,
        )

        # ----------------------------------------------------
        # If filter produced no chunks, fail safely.
        # ----------------------------------------------------

        if not allowed_ids:

            return []

        faiss_results = self._faiss_search(
            query=query,
            top_k=top_k,
            allowed_ids=allowed_ids,
        )

        bm25_results = self._bm25_search(
            query=query,
            top_k=top_k,
            allowed_ids=allowed_ids,
        )

        fused = self._rrf_fusion(
            faiss_results,
            bm25_results,
        )

        # ----------------------------------------------------
        # Score maps
        # ----------------------------------------------------

        faiss_score_map = {
            r["chunk_id"]: r["faiss_score"]
            for r in faiss_results
        }

        bm25_score_map = {
            r["chunk_id"]: r["bm25_score"]
            for r in bm25_results
        }

        # ----------------------------------------------------
        # Build complete metadata result
        # ----------------------------------------------------

        results = []

        for rank, (chunk_id, rrf_score) in enumerate(
            fused[:top_k],
            start=1,
        ):

            chunk = self.chunk_map[chunk_id]

            result = dict(chunk)

            result.update(
                {
                    "retrieval_rank": rank,
                    "rrf_score": float(rrf_score),
                    "faiss_score": faiss_score_map.get(
                        chunk_id
                    ),
                    "bm25_score": bm25_score_map.get(
                        chunk_id
                    ),
                    "retrieval_method": "hybrid",
                }
            )

            results.append(result)

        return results

    # ========================================================
    # DEBUG
    # ========================================================

    def debug_retrieve(
        self,
        query: str,
        top_k: int = 5,
        domain: Optional[str] = None,
    ):

        results = self.retrieve(
            query=query,
            top_k=top_k,
            domain=domain,
        )

        print("\n" + "=" * 80)

        print("HYBRID RETRIEVAL")

        print("=" * 80)

        print(f"Query: {query}")

        print(f"Domain filter: {domain}")

        print()

        for result in results:

            print(
                f"{result['retrieval_rank']}. "
                f"{result['chunk_id']}"
            )

            print(
                f"   Domain: "
                f"{result.get('domain')}"
            )

            print(
                f"   Document: "
                f"{result.get('file_name')}"
            )

            print(
                f"   FAISS: "
                f"{result.get('faiss_score')}"
            )

            print(
                f"   BM25: "
                f"{result.get('bm25_score')}"
            )

            print(
                f"   RRF: "
                f"{result.get('rrf_score')}"
            )

            print(
                f"   Text: "
                f"{result.get('text', '')[:200]}"
            )

            print()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    retriever = HybridRetriever()

    tests = [
        (
            "Can a customer return a product?",
            "returns",
        ),
        (
            "How long does a refund take?",
            "refunds",
        ),
        (
            "What payment methods are available?",
            "payments",
        ),
        (
            "My order is delayed, what should I do?",
            "delivery",
        ),
    ]

    for query, domain in tests:

        retriever.debug_retrieve(
            query=query,
            top_k=5,
            domain=domain,
        )