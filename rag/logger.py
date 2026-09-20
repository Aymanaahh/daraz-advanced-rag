"""
STEP 9 — RAG RETRIEVAL / ANSWER LOGGER

Purpose:
    Persist complete RAG execution records in JSONL format.

Output:
    storage/logs/retrieval_log.jsonl

Each line represents one complete RAG request.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class RAGLogger:
    """
    Enterprise-style JSONL logger for RAG requests.

    One JSON object is written per line so logs can be:
    - appended efficiently
    - inspected manually
    - processed later with Python/Pandas
    - imported into monitoring systems
    """

    def __init__(
        self,
        log_path: str = "storage/logs/retrieval_log.jsonl",
    ):
        self.log_path = Path(log_path)

        # Create parent directory automatically.
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_timestamp() -> str:
        """Return an ISO-8601 UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_value(value: Any) -> Any:
        """
        Convert objects into JSON-safe values.
        """
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, (list, tuple)):
            return [
                RAGLogger._safe_value(item)
                for item in value
            ]

        if isinstance(value, dict):
            return {
                str(key): RAGLogger._safe_value(val)
                for key, val in value.items()
            }

        if hasattr(value, "__dict__"):
            return RAGLogger._safe_value(vars(value))

        return str(value)

    # ------------------------------------------------------------------
    # Retrieval logging
    # ------------------------------------------------------------------

    def log_retrieval(
        self,
        query: str,
        query_analysis: Optional[Dict[str, Any]] = None,
        retrieval_queries: Optional[List[Dict[str, Any]]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        retrieval_statistics: Optional[Dict[str, Any]] = None,
        execution_time_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Log retrieval-stage information.

        Returns the complete log record.
        """

        evidence = evidence or []
        retrieval_queries = retrieval_queries or []
        query_analysis = query_analysis or {}
        retrieval_statistics = retrieval_statistics or {}

        record = {
            "event_type": "retrieval",
            "timestamp": self._utc_timestamp(),

            "query": query,

            "query_analysis": self._safe_value(
                query_analysis
            ),

            "retrieval_queries": self._safe_value(
                retrieval_queries
            ),

            "retrieval_statistics": self._safe_value(
                retrieval_statistics
            ),

            "evidence": self._safe_value(
                evidence
            ),

            "evidence_chunk_ids": [
                item.get("chunk_id")
                for item in evidence
                if isinstance(item, dict)
                and item.get("chunk_id")
            ],

            "execution_time_ms": execution_time_ms,
        }

        self._write(record)

        return record

    # ------------------------------------------------------------------
    # Complete RAG request logging
    # ------------------------------------------------------------------

    def log_rag_request(
        self,
        query: str,
        query_analysis: Optional[Dict[str, Any]] = None,
        retrieval_queries: Optional[List[Dict[str, Any]]] = None,
        evidence: Optional[List[Dict[str, Any]]] = None,
        retrieval_statistics: Optional[Dict[str, Any]] = None,
        answer: Optional[str] = None,
        citations: Optional[List[str]] = None,
        citation_validation: Optional[Dict[str, Any]] = None,
        total_execution_time_ms: Optional[float] = None,
        retrieval_time_ms: Optional[float] = None,
        generation_time_ms: Optional[float] = None,
        validation_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a complete RAG request.

        This is the main method that the final Streamlit application
        will use.
        """

        evidence = evidence or []
        citations = citations or []
        query_analysis = query_analysis or {}
        retrieval_queries = retrieval_queries or []
        retrieval_statistics = retrieval_statistics or {}
        metadata = metadata or {}

        record = {
            "event_type": "rag_request",
            "timestamp": self._utc_timestamp(),

            "query": query,

            "query_analysis": self._safe_value(
                query_analysis
            ),

            "retrieval_queries": self._safe_value(
                retrieval_queries
            ),

            "retrieval_statistics": self._safe_value(
                retrieval_statistics
            ),

            "evidence": self._safe_value(
                evidence
            ),

            "evidence_chunk_ids": [
                item.get("chunk_id")
                for item in evidence
                if isinstance(item, dict)
                and item.get("chunk_id")
            ],

            "answer": answer,

            "citations": self._safe_value(
                citations
            ),

            "citation_validation": self._safe_value(
                citation_validation
            ),

            "timing": {
                "total_execution_time_ms": total_execution_time_ms,
                "retrieval_time_ms": retrieval_time_ms,
                "generation_time_ms": generation_time_ms,
                "validation_time_ms": validation_time_ms,
            },

            "metadata": self._safe_value(
                metadata
            ),
        }

        self._write(record)

        return record

    # ------------------------------------------------------------------
    # Error logging
    # ------------------------------------------------------------------

    def log_error(
        self,
        query: str,
        error: Exception,
        stage: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log an error without crashing the logging process.
        """

        record = {
            "event_type": "error",
            "timestamp": self._utc_timestamp(),

            "query": query,

            "stage": stage,

            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },

            "metadata": self._safe_value(
                metadata or {}
            ),
        }

        self._write(record)

        return record

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _write(self, record: Dict[str, Any]) -> None:
        """
        Append one JSON object to the JSONL file.
        """

        try:
            with self.log_path.open(
                "a",
                encoding="utf-8",
            ) as file:

                json.dump(
                    self._safe_value(record),
                    file,
                    ensure_ascii=False,
                )

                file.write("\n")

        except Exception as exc:
            # Logging should never break the main RAG application.
            print(
                f"WARNING: Failed to write RAG log: {exc}"
            )

    def read_logs(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read existing JSONL records.

        Parameters
        ----------
        limit:
            Optional number of most recent records to return.
        """

        if not self.log_path.exists():
            return []

        records = []

        with self.log_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(
                        json.loads(line)
                    )
                except json.JSONDecodeError:
                    continue

        if limit is not None:
            return records[-limit:]

        return records

    def count_logs(self) -> int:
        """Return number of valid log records."""

        return len(self.read_logs())

    def clear_logs(self) -> None:
        """Delete the current log file."""

        if self.log_path.exists():
            self.log_path.unlink()


# ======================================================================
# STEP 9 TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("STEP 9 — RAG LOGGER TEST")
    print("=" * 80)

    logger = RAGLogger()

    print("\nLogger initialized.")
    print(f"Log file: {logger.log_path}")

    # --------------------------------------------------------------
    # Test 1 — Retrieval logging
    # --------------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 1 — RETRIEVAL LOG")
    print("-" * 80)

    retrieval_record = logger.log_retrieval(
        query="How long does a refund take?",

        query_analysis={
            "domains": ["refunds"],
            "primary_domain": "refunds",
            "intents": ["timeline"],
            "primary_intent": "timeline",
            "complex_query": False,
        },

        retrieval_queries=[
            {
                "query": "How long does a refund take?",
                "domain": "refunds",
                "result_count": 4,
            }
        ],

        evidence=[
            {
                "chunk_id": "REF-002-P01-C001",
                "document_id": "REF-002",
                "file_name": "refund_timelines.pdf",
                "domain": "refunds",
                "page": 1,
                "section": "Processing Time by Method",
                "reranker_score": 0.9074,
            },
            {
                "chunk_id": "REF-002-P01-C002",
                "document_id": "REF-002",
                "file_name": "refund_timelines.pdf",
                "domain": "refunds",
                "page": 1,
                "section": "What Affects Refund Speed",
                "reranker_score": 0.0653,
            },
        ],

        retrieval_statistics={
            "raw_results": 5,
            "unique_results": 5,
            "final_results": 4,
        },

        execution_time_ms=152.4,
    )

    print("Retrieval log written.")
    print(
        "Evidence:",
        retrieval_record["evidence_chunk_ids"]
    )

    # --------------------------------------------------------------
    # Test 2 — Complete RAG request
    # --------------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 2 — COMPLETE RAG REQUEST")
    print("-" * 80)

    answer = (
        "Daraz Wallet refunds usually arrive within 24-48 hours "
        "after approval. Credit/debit card refunds typically take "
        "5-10 business days. [SOURCE: REF-002-P01-C001]"
    )

    complete_record = logger.log_rag_request(

        query="How long does a refund take?",

        query_analysis={
            "domains": ["refunds"],
            "primary_domain": "refunds",
            "intents": ["timeline"],
            "primary_intent": "timeline",
            "complex_query": False,
        },

        retrieval_queries=[
            {
                "query": "How long does a refund take?",
                "domain": "refunds",
                "result_count": 4,
            }
        ],

        evidence=[
            {
                "chunk_id": "REF-002-P01-C001",
                "document_id": "REF-002",
                "file_name": "refund_timelines.pdf",
                "domain": "refunds",
                "page": 1,
                "section": "Processing Time by Method",
                "reranker_score": 0.9074,
            }
        ],

        retrieval_statistics={
            "raw_results": 5,
            "unique_results": 5,
            "final_results": 4,
        },

        answer=answer,

        citations=[
            "REF-002-P01-C001"
        ],

        citation_validation={
            "valid": True,
            "citation_count": 1,
            "valid_citation_count": 1,
            "coverage": 100.0,
            "invalid_citations": [],
            "issues": [],
        },

        total_execution_time_ms=2431.8,
        retrieval_time_ms=1250.4,
        generation_time_ms=1100.7,
        validation_time_ms=80.7,

        metadata={
            "application": "daraz-advanced-rag",
            "environment": "development",
        },
    )

    print("Complete RAG request logged.")

    # --------------------------------------------------------------
    # Test 3 — Read logs
    # --------------------------------------------------------------

    print("\n" + "-" * 80)
    print("TEST 3 — READ LOGS")
    print("-" * 80)

    records = logger.read_logs()

    print(
        f"Total log records: {len(records)}"
    )

    if records:
        latest = records[-1]

        print(
            "Latest event type:",
            latest.get("event_type")
        )

        print(
            "Latest query:",
            latest.get("query")
        )

        print(
            "Latest citations:",
            latest.get("citations")
        )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 9 TEST COMPLETE")
    print("=" * 80)

    print(
        f"\nLog file created at:\n{logger.log_path}"
    )