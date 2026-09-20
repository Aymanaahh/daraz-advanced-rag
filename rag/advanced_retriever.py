"""
Advanced Retriever
------------------

Production-oriented retrieval pipeline:

1. Query processing
2. Query decomposition
3. Domain-aware retrieval
4. FAISS + BM25 hybrid retrieval
5. Cross-encoder reranking
6. Deduplication
7. Evidence selection
8. Metadata preservation

Compatible with the existing QueryProcessor
which returns a ProcessedQuery object.
"""

from rag.query_processor import QueryProcessor
from rag.retriever import HybridRetriever
from rag.reranker import CrossEncoderReranker


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_K = 10
RERANKED_K = 5
FINAL_EVIDENCE_K = 5
MAX_CHUNKS_PER_DOCUMENT = 2


# ============================================================
# ADVANCED RETRIEVER
# ============================================================

class AdvancedRetriever:

    def __init__(self):

        print("\nInitializing AdvancedRetriever...")

        # ----------------------------------------------------
        # 1. QUERY PROCESSOR
        # ----------------------------------------------------

        print("\n[1/3] Loading Query Processor...")

        self.query_processor = QueryProcessor()

        print("Query Processor ready.")

        # ----------------------------------------------------
        # 2. HYBRID RETRIEVER
        # ----------------------------------------------------

        print("\n[2/3] Loading Hybrid Retriever...")

        self.hybrid_retriever = HybridRetriever()

        print("Hybrid Retriever ready.")

        # ----------------------------------------------------
        # 3. RERANKER
        # ----------------------------------------------------

        print("\n[3/3] Loading Cross-Encoder Reranker...")

        self.reranker = CrossEncoderReranker()

        print("Cross-Encoder Reranker ready.")

        print("\nAdvancedRetriever initialized successfully.")

    # ========================================================
    # CONVERT ProcessedQuery -> DICTIONARY
    # ========================================================

    @staticmethod
    def processed_query_to_dict(processed):

        """
        Convert the custom ProcessedQuery object returned by
        QueryProcessor.process() into a normal dictionary.

        This avoids coupling the rest of the retrieval
        pipeline to the QueryProcessor class implementation.
        """

        # ----------------------------------------------------
        # Already a dictionary
        # ----------------------------------------------------

        if isinstance(processed, dict):

            return processed

        # ----------------------------------------------------
        # Dataclass
        # ----------------------------------------------------

        try:

            from dataclasses import asdict, is_dataclass

            if is_dataclass(processed):

                return asdict(processed)

        except Exception:
            pass

        # ----------------------------------------------------
        # Object with __dict__
        # ----------------------------------------------------

        if hasattr(processed, "__dict__"):

            return dict(
                vars(processed)
            )

        # ----------------------------------------------------
        # Object with common attributes
        # ----------------------------------------------------

        possible_fields = [
            "original_query",
            "query",
            "normalized_query",
            "domains",
            "primary_domain",
            "intents",
            "primary_intent",
            "subqueries",
            "is_complex",
            "complex_query",
            "confidence",
            "rewritten_query",
        ]

        result = {}

        for field in possible_fields:

            if hasattr(processed, field):

                result[field] = getattr(
                    processed,
                    field
                )

        return result

    # ========================================================
    # PROCESS QUERY
    # ========================================================

    def process_query(self, query):

        processed = self.query_processor.process(
            query
        )

        return self.processed_query_to_dict(
            processed
        )

    # ========================================================
    # REPAIR QUERY
    # ========================================================

    @staticmethod
    def repair_query(query):

        if not query:

            return ""

        query = str(query).strip()

        # Previous Step 5 decomposition could produce:
        #
        # "long will my refund take?"
        #
        # Repair it to:
        #
        # "How long will my refund take?"

        if query.lower().startswith("long "):

            query = "How " + query

        return query

    # ========================================================
    # NORMALIZE DOMAINS
    # ========================================================

    @staticmethod
    def normalize_domains(
        domains
    ):

        if domains is None:

            return []

        if isinstance(
            domains,
            str
        ):

            return [domains]

        return list(domains)

    # ========================================================
    # GET SUBQUERIES
    # ========================================================

    def get_subqueries(
        self,
        analysis,
        original_query
    ):

        subqueries = analysis.get(
            "subqueries",
            []
        )

        # ----------------------------------------------------
        # No decomposition
        # ----------------------------------------------------

        if not subqueries:

            return [
                self.repair_query(
                    original_query
                )
            ]

        # ----------------------------------------------------
        # Repair each subquery
        # ----------------------------------------------------

        repaired = []

        for subquery in subqueries:

            repaired_query = self.repair_query(
                subquery
            )

            if repaired_query:

                repaired.append(
                    repaired_query
                )

        if not repaired:

            return [
                self.repair_query(
                    original_query
                )
            ]

        return repaired

    # ========================================================
    # DETERMINE DOMAIN FOR SUBQUERY
    # ========================================================

    def determine_subquery_domain(
        self,
        subquery,
        parent_analysis,
        index
    ):

        # ----------------------------------------------------
        # Try processing the subquery itself.
        # ----------------------------------------------------

        sub_analysis = self.process_query(
            subquery
        )

        sub_domains = self.normalize_domains(
            sub_analysis.get(
                "domains",
                []
            )
        )

        if sub_domains:

            return sub_domains[0]

        # ----------------------------------------------------
        # Parent domains
        # ----------------------------------------------------

        parent_domains = self.normalize_domains(
            parent_analysis.get(
                "domains",
                []
            )
        )

        # ----------------------------------------------------
        # Single-domain query
        # ----------------------------------------------------

        if len(parent_domains) == 1:

            return parent_domains[0]

        # ----------------------------------------------------
        # Multi-domain query
        # ----------------------------------------------------

        if index < len(parent_domains):

            return parent_domains[index]

        # ----------------------------------------------------
        # Primary domain fallback
        # ----------------------------------------------------

        return parent_analysis.get(
            "primary_domain"
        )

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    @staticmethod
    def deduplicate(
        results
    ):

        seen = set()

        unique = []

        for result in results:

            chunk_id = result.get(
                "chunk_id"
            )

            if not chunk_id:

                continue

            if chunk_id in seen:

                continue

            seen.add(
                chunk_id
            )

            unique.append(
                result
            )

        return unique

    # ========================================================
    # EVIDENCE SELECTION
    # ========================================================

    @staticmethod
    def select_evidence(
        results,
        final_k=FINAL_EVIDENCE_K,
        max_chunks_per_document=
            MAX_CHUNKS_PER_DOCUMENT
    ):

        selected = []

        document_counts = {}

        for result in results:

            document_id = result.get(
                "document_id"
            )

            # ------------------------------------------------
            # Fallback identifiers
            # ------------------------------------------------

            if not document_id:

                document_id = result.get(
                    "file_name"
                )

            if not document_id:

                document_id = result.get(
                    "chunk_id"
                )

            current_count = (
                document_counts.get(
                    document_id,
                    0
                )
            )

            if (
                current_count
                >=
                max_chunks_per_document
            ):

                continue

            selected_result = dict(
                result
            )

            selected_result[
                "selected_for_context"
            ] = True

            selected.append(
                selected_result
            )

            document_counts[
                document_id
            ] = current_count + 1

            if len(selected) >= final_k:

                break

        return selected

    # ========================================================
    # RETRIEVE ONE SUBQUERY
    # ========================================================

    def retrieve_subquery(
        self,
        subquery,
        domain=None
    ):

        subquery = self.repair_query(
            subquery
        )

        print("\nRetrieving subquery:")

        print(
            f"  Query: {subquery}"
        )

        print(
            f"  Domain filter: {domain}"
        )

        # ----------------------------------------------------
        # HYBRID RETRIEVAL
        # ----------------------------------------------------

        candidates = (
            self.hybrid_retriever.retrieve(
                query=subquery,
                top_k=CANDIDATE_K,
                domain=domain
            )
        )

        if not candidates:

            print(
                "  No candidates found."
            )

            return []

        print(
            f"  Hybrid candidates: "
            f"{len(candidates)}"
        )

        # ----------------------------------------------------
        # RERANK
        # ----------------------------------------------------

        reranked = self.reranker.rerank(
            query=subquery,
            candidates=candidates,
            top_k=RERANKED_K
        )

        # ----------------------------------------------------
        # METADATA VALIDATION
        # ----------------------------------------------------

        candidate_map = {

            item["chunk_id"]: item

            for item in candidates

        }

        validated = []

        for reranked_item in reranked:

            chunk_id = reranked_item.get(
                "chunk_id"
            )

            original = candidate_map.get(
                chunk_id
            )

            if original is None:

                continue

            # ------------------------------------------------
            # Start with the COMPLETE original chunk.
            # ------------------------------------------------

            merged = dict(
                original
            )

            # ------------------------------------------------
            # Add reranking information.
            # ------------------------------------------------

            merged[
                "reranker_score"
            ] = reranked_item.get(
                "reranker_score"
            )

            merged[
                "reranked"
            ] = True

            merged[
                "reranker_rank"
            ] = reranked_item.get(
                "reranker_rank"
            )

            merged[
                "source_query"
            ] = subquery

            validated.append(
                merged
            )

        return validated

    # ========================================================
    # MAIN RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query
    ):

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        analysis = self.process_query(
            query
        )

        # ----------------------------------------------------
        # STEP 2
        # ----------------------------------------------------

        subqueries = self.get_subqueries(
            analysis,
            query
        )

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        all_results = []

        query_details = []

        for index, subquery in enumerate(
            subqueries
        ):

            domain = (
                self.determine_subquery_domain(
                    subquery=subquery,
                    parent_analysis=analysis,
                    index=index
                )
            )

            results = (
                self.retrieve_subquery(
                    subquery=subquery,
                    domain=domain
                )
            )

            all_results.extend(
                results
            )

            query_details.append(
                {
                    "query": subquery,
                    "domain": domain,
                    "result_count": len(
                        results
                    )
                }
            )

        # ----------------------------------------------------
        # STEP 4 — DEDUPLICATE
        # ----------------------------------------------------

        unique_results = (
            self.deduplicate(
                all_results
            )
        )

        # ----------------------------------------------------
        # STEP 5 — SORT
        # ----------------------------------------------------

        unique_results.sort(
            key=lambda item:
                item.get(
                    "reranker_score",
                    float("-inf")
                ),
            reverse=True
        )

        # ----------------------------------------------------
        # STEP 6 — FINAL EVIDENCE
        # ----------------------------------------------------

        final_results = (
            self.select_evidence(
                results=unique_results,
                final_k=FINAL_EVIDENCE_K,
                max_chunks_per_document=
                    MAX_CHUNKS_PER_DOCUMENT
            )
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        statistics = {

            "raw_results":
                len(all_results),

            "unique_results":
                len(unique_results),

            "final_results":
                len(final_results),

            "subqueries":
                len(subqueries),

            "domains":
                self.normalize_domains(
                    analysis.get(
                        "domains",
                        []
                    )
                )
        }

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return {

            "original_query":
                query,

            "query_analysis":
                analysis,

            "retrieval_queries":
                subqueries,

            "query_details":
                query_details,

            "all_results":
                all_results,

            "unique_results":
                unique_results,

            "final_results":
                final_results,

            "statistics":
                statistics
        }

    # ========================================================
    # DISPLAY
    # ========================================================

    def display_result(
        self,
        result
    ):

        print("\n" + "=" * 80)

        print(
            "ADVANCED RETRIEVAL RESULT"
        )

        print("=" * 80)

        print(
            "\nORIGINAL QUERY"
        )

        print(
            result[
                "original_query"
            ]
        )

        # ----------------------------------------------------
        # QUERY ANALYSIS
        # ----------------------------------------------------

        analysis = result[
            "query_analysis"
        ]

        domains = self.normalize_domains(
            analysis.get(
                "domains",
                []
            )
        )

        print(
            "\nQUERY ANALYSIS"
        )

        print(
            "Domains:",
            ", ".join(domains)
        )

        print(
            "Primary domain:",
            analysis.get(
                "primary_domain"
            )
        )

        intents = analysis.get(
            "intents",
            []
        )

        if isinstance(
            intents,
            str
        ):

            intents = [intents]

        print(
            "Intents:",
            ", ".join(intents)
        )

        print(
            "Primary intent:",
            analysis.get(
                "primary_intent"
            )
        )

        print(
            "Complex query:",
            analysis.get(
                "complex_query",
                analysis.get(
                    "is_complex",
                    False
                )
            )
        )

        # ----------------------------------------------------
        # RETRIEVAL QUERIES
        # ----------------------------------------------------

        print(
            "\nRETRIEVAL QUERIES"
        )

        for index, detail in enumerate(
            result[
                "query_details"
            ],
            start=1
        ):

            print(
                f"{index}. "
                f"{detail['query']}"
            )

            print(
                f"   Domain filter: "
                f"{detail['domain']}"
            )

            print(
                f"   Results: "
                f"{detail['result_count']}"
            )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        statistics = result[
            "statistics"
        ]

        print(
            "\nRETRIEVAL STATISTICS"
        )

        print(
            f"Raw results: "
            f"{statistics['raw_results']}"
        )

        print(
            f"Unique results: "
            f"{statistics['unique_results']}"
        )

        print(
            f"Final results: "
            f"{statistics['final_results']}"
        )

        # ----------------------------------------------------
        # FINAL EVIDENCE
        # ----------------------------------------------------

        print(
            "\n" + "-" * 80
        )

        print(
            "FINAL EVIDENCE"
        )

        print(
            "-" * 80
        )

        for rank, item in enumerate(
            result[
                "final_results"
            ],
            start=1
        ):

            print(
                "\n" + "-" * 80
            )

            print(
                f"Final Rank: {rank}"
            )

            print(
                f"Chunk: "
                f"{item.get('chunk_id')}"
            )

            print(
                f"Reranker Score: "
                f"{item.get('reranker_score')}"
            )

            print(
                f"Document: "
                f"{item.get('file_name')}"
            )

            print(
                f"Document ID: "
                f"{item.get('document_id')}"
            )

            print(
                f"Domain: "
                f"{item.get('domain')}"
            )

            print(
                f"Document Type: "
                f"{item.get('document_type')}"
            )

            print(
                f"Page: "
                f"{item.get('page')}"
            )

            print(
                f"Section: "
                f"{item.get('section')}"
            )

            print(
                f"Source Query: "
                f"{item.get('source_query')}"
            )

            print(
                "\nTEXT:"
            )

            print(
                item.get(
                    "text",
                    ""
                )
            )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)

    print(
        "STEP 6 — FINAL COMPATIBILITY TEST"
    )

    print("=" * 80)

    retriever = AdvancedRetriever()

    test_queries = [

        "Can a customer return a product?",

        "How long does a refund take?",

        "What payment methods are available?",

        "Can I return a laptop after 10 days and how long will my refund take?",

        "My order is delayed, what should I do and how long can delivery take?",

        "My payment failed and the amount was deducted, what should I do?"

    ]

    for query in test_queries:

        result = retriever.retrieve(
            query
        )

        retriever.display_result(
            result
        )

        print(
            "\n" + "=" * 80
        )

    print(
        "\nSTEP 6 FINAL COMPATIBILITY TEST COMPLETE"
    )