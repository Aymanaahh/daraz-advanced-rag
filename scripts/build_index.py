from pathlib import Path
import json
import hashlib
import pickle
from datetime import datetime, timezone

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "storage" / "chunks" / "chunks.json"
DOCUMENTS_FILE = PROJECT_ROOT / "storage" / "metadata" / "documents.json"

FAISS_DIR = PROJECT_ROOT / "storage" / "faiss"
BM25_DIR = PROJECT_ROOT / "storage" / "bm25"

FAISS_INDEX_FILE = FAISS_DIR / "index.faiss"
FAISS_METADATA_FILE = FAISS_DIR / "index_metadata.json"

BM25_INDEX_FILE = BM25_DIR / "index.pkl"
BM25_METADATA_FILE = BM25_DIR / "index_metadata.json"


# English corpus → suitable lightweight retrieval model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Batch size keeps memory usage reasonable
EMBEDDING_BATCH_SIZE = 32


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_now():
    """Return current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def ensure_directories():
    """Create required storage directories."""
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    BM25_DIR.mkdir(parents=True, exist_ok=True)


def load_json(file_path: Path):
    """Load JSON file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}\n\n"
            "Run scripts/extract_and_chunk.py first."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_text(text: str):
    """
    Simple BM25 tokenizer.

    Lowercase + whitespace tokenization.
    Keeps the implementation lightweight and deterministic.
    """
    return text.lower().split()


# ============================================================
# LOAD DATA
# ============================================================

def load_chunks():
    """Load chunk records from chunks.json."""

    data = load_json(CHUNKS_FILE)

    if isinstance(data, dict):
        # Supports either:
        # {"chunks": [...]}
        # or another dictionary-based structure.
        chunks = data.get("chunks", [])

        if not chunks:
            raise ValueError(
                "chunks.json was found, but no chunks were detected."
            )
    elif isinstance(data, list):
        chunks = data
    else:
        raise ValueError("Unsupported chunks.json format.")

    return chunks


def load_documents():
    """Load document metadata."""

    if not DOCUMENTS_FILE.exists():
        print(
            "WARNING: documents.json was not found.\n"
            "Continuing with chunks.json only."
        )
        return []

    data = load_json(DOCUMENTS_FILE)

    if isinstance(data, dict):
        documents = data.get("documents", [])

        if not documents:
            # Some formats may store documents directly as values.
            documents = list(data.values())

    elif isinstance(data, list):
        documents = data

    else:
        documents = []

    return documents


# ============================================================
# VALIDATION
# ============================================================

def validate_chunks(chunks):
    """Validate chunk records before indexing."""

    required_fields = [
        "chunk_id",
        "text",
    ]

    valid_chunks = []

    for i, chunk in enumerate(chunks):

        missing = [
            field
            for field in required_fields
            if field not in chunk
        ]

        if missing:
            print(
                f"WARNING: Chunk #{i + 1} missing fields: {missing}"
            )
            continue

        text = str(chunk["text"]).strip()

        if not text:
            print(
                f"WARNING: Empty text for chunk "
                f"{chunk.get('chunk_id', i)}"
            )
            continue

        valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError("No valid chunks available for indexing.")

    return valid_chunks


# ============================================================
# EMBEDDINGS + FAISS
# ============================================================

def build_faiss_index(chunks, source_hash):
    """
    Generate embeddings and build a normalized
    FAISS inner-product index.

    Because vectors are normalized, inner product
    behaves like cosine similarity.
    """

    print("\n" + "=" * 70)
    print("BUILDING EMBEDDINGS + FAISS")
    print("=" * 70)

    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Chunks to embed: {len(chunks)}")

    texts = [
        str(chunk["text"]).strip()
        for chunk in chunks
    ]

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding model loaded.")

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    embedding_dimension = embeddings.shape[1]

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    print("\nCreating FAISS index...")

    index = faiss.IndexFlatIP(embedding_dimension)

    index.add(embeddings)

    print(f"FAISS vectors stored: {index.ntotal}")

    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        str(FAISS_INDEX_FILE),
    )

    print(
        f"FAISS index saved:\n{FAISS_INDEX_FILE}"
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    metadata = {
        "index_type": "IndexFlatIP",
        "similarity": "cosine",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": int(embedding_dimension),
        "total_vectors": int(index.ntotal),
        "chunk_count": len(chunks),
        "chunk_ids": chunk_ids,
        "source_chunks_hash": source_hash,
        "created_at": utc_now(),
    }

    with open(
        FAISS_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"FAISS metadata saved:\n{FAISS_METADATA_FILE}"
    )

    return {
        "embedding_dimension": embedding_dimension,
        "vector_count": index.ntotal,
    }


# ============================================================
# BM25
# ============================================================

def build_bm25_index(chunks, source_hash):
    """
    Build a persistent BM25 lexical search index.
    """

    print("\n" + "=" * 70)
    print("BUILDING BM25 INDEX")
    print("=" * 70)

    texts = [
        str(chunk["text"]).strip()
        for chunk in chunks
    ]

    print(f"Documents for BM25: {len(texts)}")

    print("\nTokenizing documents...")

    tokenized_corpus = [
        tokenize_text(text)
        for text in texts
    ]

    print("Building BM25 index...")

    bm25 = BM25Okapi(tokenized_corpus)

    # --------------------------------------------------------
    # Store BM25 object + associated chunk IDs
    # --------------------------------------------------------

    bm25_payload = {
        "bm25": bm25,
        "chunk_ids": [
            chunk["chunk_id"]
            for chunk in chunks
        ],
        "texts": texts,
    }

    with open(
        BM25_INDEX_FILE,
        "wb",
    ) as f:
        pickle.dump(
            bm25_payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"BM25 index saved:\n{BM25_INDEX_FILE}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "index_type": "BM25Okapi",
        "chunk_count": len(chunks),
        "chunk_ids": [
            chunk["chunk_id"]
            for chunk in chunks
        ],
        "source_chunks_hash": source_hash,
        "created_at": utc_now(),
    }

    with open(
        BM25_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"BM25 metadata saved:\n{BM25_METADATA_FILE}"
    )

    return {
        "document_count": len(texts),
    }


# ============================================================
# INDEX STATUS
# ============================================================

def check_existing_indexes(source_hash):
    """
    Check whether existing indexes were generated
    from the same chunks.json.

    This prevents unnecessary rebuilding.
    """

    if not FAISS_INDEX_FILE.exists():
        return False

    if not BM25_INDEX_FILE.exists():
        return False

    if not FAISS_METADATA_FILE.exists():
        return False

    if not BM25_METADATA_FILE.exists():
        return False

    try:
        with open(
            FAISS_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            faiss_meta = json.load(f)

        with open(
            BM25_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            bm25_meta = json.load(f)

        faiss_hash = faiss_meta.get(
            "source_chunks_hash"
        )

        bm25_hash = bm25_meta.get(
            "source_chunks_hash"
        )

        if (
            faiss_hash == source_hash
            and bm25_hash == source_hash
        ):
            return True

    except Exception:
        return False

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("DARAZ ADVANCED RAG — INDEX BUILDER")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nChunks file:")
    print(CHUNKS_FILE)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"\nchunks.json not found:\n{CHUNKS_FILE}\n\n"
            "Run this first:\n"
            "python scripts/extract_and_chunk.py"
        )

    ensure_directories()

    # --------------------------------------------------------
    # Source hash
    # --------------------------------------------------------

    print("\nCalculating source hash...")

    source_hash = sha256_file(CHUNKS_FILE)

    print(f"Source hash: {source_hash}")

    # --------------------------------------------------------
    # Check whether rebuilding is necessary
    # --------------------------------------------------------

    if check_existing_indexes(source_hash):

        print("\n" + "=" * 70)
        print("INDEXES ARE ALREADY UP TO DATE")
        print("=" * 70)

        print(
            "\nFAISS and BM25 indexes were built "
            "from the current chunks.json."
        )

        print("\nNo rebuilding required.")

        return

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    print("\nLoading chunks...")

    chunks = load_chunks()

    chunks = validate_chunks(chunks)

    print(
        f"Valid chunks loaded: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    documents = load_documents()

    print(
        f"Document metadata records: {len(documents)}"
    )

    # --------------------------------------------------------
    # Build FAISS
    # --------------------------------------------------------

    faiss_stats = build_faiss_index(
        chunks,
        source_hash,
    )

    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    bm25_stats = build_bm25_index(
        chunks,
        source_hash,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INDEX BUILD COMPLETE")
    print("=" * 70)

    print(
        f"\nDocuments: {len(documents)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL_NAME}"
    )

    print(
        f"Embedding dimension: "
        f"{faiss_stats['embedding_dimension']}"
    )

    print(
        f"FAISS vectors: "
        f"{faiss_stats['vector_count']}"
    )

    print(
        f"BM25 documents: "
        f"{bm25_stats['document_count']}"
    )

    print("\nGenerated files:")

    print(
        f"  ✓ {FAISS_INDEX_FILE}"
    )

    print(
        f"  ✓ {FAISS_METADATA_FILE}"
    )

    print(
        f"  ✓ {BM25_INDEX_FILE}"
    )

    print(
        f"  ✓ {BM25_METADATA_FILE}"
    )

    print("\nNext stage:")

    print(
        "  Hybrid FAISS + BM25 retrieval"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()