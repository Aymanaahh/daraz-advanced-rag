# 🛒 Daraz Operations AI

An enterprise-style Retrieval-Augmented Generation (RAG) application for answering Daraz operational and policy questions using a grounded internal knowledge base.

The application combines:

- PDF document processing
- Section-aware chunking
- Metadata enrichment
- BGE semantic embeddings
- FAISS vector search
- BM25 lexical search
- Hybrid retrieval
- Reciprocal Rank Fusion
- BGE cross-encoder reranking
- Query analysis and decomposition
- Groq LLM generation
- Citation validation
- Retrieval logging
- Streamlit interface

---

# 1. Project Overview

Daraz operations involve many policy-driven questions related to:

- Sellers
- Seller onboarding
- Refunds
- Refund timelines
- Delivery
- Delivery timelines
- Payments
- Returns
- Return eligibility
- Customer support

Traditional keyword search can retrieve documents but does not provide a grounded natural-language response.

This application uses RAG to retrieve relevant evidence first and then generate an answer using only the retrieved knowledge.

The application also attaches source identifiers to generated claims and validates those citations against the retrieved evidence.

---

# 2. Architecture

## Offline / Index Building

```text
PDF Documents
      │
      ▼
Text Extraction
      │
      ▼
Cleaning
      │
      ▼
Section Detection
      │
      ▼
Chunking
      │
      ▼
Metadata
      │
      ├───────────────┐
      ▼               ▼
 Embeddings         BM25
      │               │
      ▼               ▼
    FAISS          BM25 Index
      │               │
      └───────┬───────┘
              ▼
      Persistent Storage