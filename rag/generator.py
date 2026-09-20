"""
Step 8A — Advanced Multi-Domain Retriever

Responsibilities:
- Process the user's query
- Detect domains and intents
- Handle complex queries
- Retrieve independently for each relevant domain/subquery
- Combine results
- Deduplicate chunks
- Rerank evidence
- Preserve complete metadata
- Prevent one domain from hiding another domain

Pipeline:

User Query
    ↓
Query Processor
    ↓
Subqueries + Domains
    ↓
FAISS + BM25 Hybrid Retrieval
    ↓
Merge Results
    ↓
Deduplicate
    ↓
Cross-Encoder Reranking
    ↓
Final Evidence
"""

from typing import Any, Dict, List, Optional
from dataclasses import asdict, is_dataclass

from .query_processor import QueryProcessor
from .retriever import HybridRetriever
from .reranker import CrossEncoderReranker


class AdvancedRetriever:

    # Number of candidates retrieved from each domain/subquery
    CANDIDATE_K = 5

    # Maximum final evidence chunks
    FINAL_K = 6

    # Maximum chunks from one document
    MAX_PER_DOCUMENT = 3

    def __init__(self):

        print("Initializing AdvancedRetriever...")

        # --------------------------------------------------------------
        # Query Processor
        # --------------------------------------------------------------

        print("\n[1/3] Loading Query Processor...")

        self.query_processor = QueryProcessor()

        print("Query Processor ready.")

        # --------------------------------------------------------------
        # Hybrid Retriever
        # --------------------------------------------------------------

        print("\n[2/3] Loading Hybrid Retriever...")

        self.hybrid_retriever = HybridRetriever()

        print("Hybrid Retriever ready.")

        # --------------------------------------------------------------
        # Reranker
        # --------------------------------------------------------------

        print("\n[3/3] Loading Cross-Encoder Reranker...")

        self.reranker = CrossEncoderReranker()

        print("Cross-Encoder Reranker ready.")

        print("\nAdvancedRetriever initialized successfully.")

    # ==================================================================
    # ProcessedQuery compatibility
    # ==================================================================

    @staticmethod
    def processed_query_to_dict(processed: Any) -> Dict[str, Any]:
        """
        Convert QueryProcessor's ProcessedQuery object into a dictionary.

        Supports:
        - dict
        - dataclass
        - normal Python object with __dict__
        """

        if isinstance(processed, dict):
            return processed

        if is_dataclass(processed):
            return asdict(processed)

        if hasattr(processed, "__dict__"):
            return dict(vars(processed))

        possible_fields = [
            "original_query",
            "query",
            "normalized_query",
            "rewritten_query",
            "domains",
            "primary_domain",
            "intents",
            "primary_intent",
            "subqueries",
            "is_complex",
            "complex_query",
            "confidence",
        ]

        result = {}

        for field in possible_fields:

            if hasattr(processed, field):
                result[field] = getattr(processed, field)

        return result

    # ==================================================================
    # Query processing
    # ==================================================================

    def process_query(self, query: str) -> Dict[str, Any]:

        processed = self.query_processor.process(query)

        return self.processed_query_to_dict(processed)

    # ==================================================================
    # Normalize domains
    # ==================================================================

    @staticmethod
    def normalize_domains(
        domains: Any,
        primary_domain: Optional[str] = None,
    ) -> List[str]:
        """
        Normalize domains into a clean list.
        """

        if domains is None:

            result = []

        elif isinstance(domains, str):

            result = [
                x.strip()
                for x in domains.split(",")
                if x.strip()
            ]

        elif isinstance(domains, (list, tuple, set)):

            result = [
                str(x).strip()
                for x in domains
                if str(x).strip()
            ]

        else:

            result = []

        if primary_domain:

            primary_domain = str(primary_domain).strip()

            if primary_domain and primary_domain not in result:
                result.insert(0, primary_domain)

        # Remove duplicates while preserving order
        unique = []

        for domain in result:

            if domain not in unique:
                unique.append(domain)

        return unique

    # ==================================================================
    # Build retrieval plan
    # ==================================================================

    def build_retrieval_plan(
        self,
        query: str,
        processed: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Build independent retrieval tasks.

        Important behavior:

        Simple query:
            one query + one domain

        Complex multi-domain query:
            each subquery is paired with its appropriate domain.

        If the QueryProcessor gives subqueries without explicit
        domain assignments, domains are distributed intelligently.
        """

        domains = self.normalize_domains(
            processed.get("domains"),
            processed.get("primary_domain"),
        )

        subqueries = processed.get("subqueries")

        if isinstance(subqueries, str):
            subqueries = [subqueries]

        if not subqueries:
            subqueries = [query]

        subqueries = [
            str(q).strip()
            for q in subqueries
            if str(q).strip()
        ]

        if not subqueries:
            subqueries = [query]

        plan = []

        # --------------------------------------------------------------
        # Simple query
        # --------------------------------------------------------------

        if len(subqueries) == 1:

            subquery = subqueries[0]

            domain = domains[0] if domains else None

            plan.append(
                {
                    "query": subquery,
                    "domain": domain,
                }
            )

            return plan

        # --------------------------------------------------------------
        # Complex query
        # --------------------------------------------------------------

        # If there are equal numbers of subqueries and domains,
        # map them directly.
        if len(subqueries) == len(domains):

            for subquery, domain in zip(subqueries, domains):

                plan.append(
                    {
                        "query": subquery,
                        "domain": domain,
                    }
                )

            return plan

        # --------------------------------------------------------------
        # More subqueries than domains
        # --------------------------------------------------------------

        if domains:

            for index, subquery in enumerate(subqueries):

                if index < len(domains):

                    domain = domains[index]

                else:

                    # For extra subqueries, use the primary domain
                    domain = domains[0]

                plan.append(
                    {
                        "query": subquery,
                        "domain": domain,
                    }
                )

            return plan

        # --------------------------------------------------------------
        # No domains detected
        # --------------------------------------------------------------

        for subquery in subqueries:

            plan.append(
                {
                    "query": subquery,
                    "domain": None,
                }
            )

        return plan

    # ==================================================================
    # Domain-aware fallback
    # ==================================================================

    @staticmethod
    def infer_domain_for_subquery(
        subquery: str,
        domains: List[str],
        index: int,
    ) -> Optional[str]:
        """
        Lightweight fallback for assigning a domain.

        This is only used when QueryProcessor does not provide enough
        information to map subqueries to domains.
        """

        if not domains:
            return None

        text = subquery.lower()

        domain_keywords = {
            "returns": [
                "return",
                "returned",
                "eligible",
                "eligibility",
                "laptop",
                "product",
            ],
            "refunds": [
                "refund",
                "refunded",
                "money back",
                "reimbursement",
            ],
            "delivery": [
                "delivery",
                "deliver",
                "delayed",
                "delay",
                "courier",
                "shipment",
                "shipping",
            ],
            "payments": [
                "payment",
                "paid",
                "card",
                "wallet",
                "cod",
                "deducted",
            ],
            "sellers": [
                "seller",
                "selling",
                "seller account",
                "onboarding",
            ],
            "customer_support": [
                "support",
                "contact",
                "complaint",
                "help",
            ],
        }

        scores = {}

        for domain in domains:

            keywords = domain_keywords.get(domain, [])

            score = 0

            for keyword in keywords:

                if keyword in text:
                    score += 1

            scores[domain] = score

        if scores:

            best_domain = max(
                scores,
                key=scores.get,
            )

            if scores[best_domain] > 0:
                return best_domain

        # Last fallback
        return domains[index % len(domains)]

    # ==================================================================
    # Retrieve candidates
    # ==================================================================

    def retrieve_candidates(
        self,
        retrieval_plan: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidates independently for every retrieval task.
        """

        all_candidates = []

        for task_index, task in enumerate(retrieval_plan):

            subquery = task["query"]
            domain = task.get("domain")

            print("\n" + "=" * 80)

            print("RETRIEVING SUBQUERY")
            print(f"  Query: {subquery}")
            print(f"  Domain filter: {domain}")

            candidates = self.hybrid_retriever.retrieve(
                query=subquery,
                top_k=self.CANDIDATE_K,
                domain=domain,
            )

            print(
                f"  Hybrid candidates: {len(candidates)}"
            )

            for candidate in candidates:

                item = dict(candidate)

                item["source_query"] = subquery
                item["source_domain"] = domain
                item["retrieval_task"] = task_index + 1

                all_candidates.append(item)

        return all_candidates

    # ==================================================================
    # Deduplicate
    # ==================================================================

    @staticmethod
    def deduplicate_candidates(
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate using chunk_id.

        If the same chunk is retrieved by multiple subqueries,
        preserve the first occurrence and record all source queries.
        """

        unique = {}
        source_queries = {}

        for item in candidates:

            chunk_id = item.get("chunk_id")

            if not chunk_id:
                continue

            if chunk_id not in unique:

                unique[chunk_id] = dict(item)

                source_queries[chunk_id] = []

            source_query = item.get("source_query")

            if source_query and source_query not in source_queries[chunk_id]:

                source_queries[chunk_id].append(source_query)

        results = []

        for chunk_id, item in unique.items():

            item["source_queries"] = source_queries[chunk_id]

            results.append(item)

        return results

    # ==================================================================
    # Reranking
    # ==================================================================

    def rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using the cross encoder.

        Metadata from the original candidate is restored after
        reranking.
        """

        if not candidates:
            return []

        print(
            f"\nReranking {len(candidates)} candidates..."
        )

        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=len(candidates),
        )

        candidate_map = {
            item["chunk_id"]: item
            for item in candidates
            if item.get("chunk_id")
        }

        final = []

        for rank, reranked_item in enumerate(
            reranked,
            start=1,
        ):

            chunk_id = reranked_item.get("chunk_id")

            original = candidate_map.get(chunk_id)

            if original is None:
                continue

            merged = dict(original)

            merged["reranker_score"] = (
                reranked_item.get("reranker_score")
            )

            merged["reranked"] = True
            merged["reranker_rank"] = rank

            final.append(merged)

        return final

    # ==================================================================
    # Final evidence selection
    # ==================================================================

    def select_final_evidence(
        self,
        reranked: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Select final evidence while maintaining document diversity.

        Maximum:
        - FINAL_K total chunks
        - MAX_PER_DOCUMENT chunks from one document
        """

        selected = []

        document_counts = {}

        for item in reranked:

            document_id = item.get(
                "document_id",
                item.get("file_name", "unknown"),
            )

            current_count = document_counts.get(
                document_id,
                0,
            )

            if current_count >= self.MAX_PER_DOCUMENT:
                continue

            selected.append(item)

            document_counts[document_id] = (
                current_count + 1
            )

            if len(selected) >= self.FINAL_K:
                break

        return selected

    # ==================================================================
    # Main retrieval function
    # ==================================================================

    def retrieve(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Complete advanced retrieval pipeline.
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        query = query.strip()

        # --------------------------------------------------------------
        # Query analysis
        # --------------------------------------------------------------

        processed = self.process_query(query)

        domains = self.normalize_domains(
            processed.get("domains"),
            processed.get("primary_domain"),
        )

        # --------------------------------------------------------------
        # Build retrieval plan
        # --------------------------------------------------------------

        retrieval_plan = self.build_retrieval_plan(
            query=query,
            processed=processed,
        )

        # --------------------------------------------------------------
        # Improve domain assignment
        # --------------------------------------------------------------

        for index, task in enumerate(retrieval_plan):

            if task.get("domain") is None:

                task["domain"] = self.infer_domain_for_subquery(
                    subquery=task["query"],
                    domains=domains,
                    index=index,
                )

        # --------------------------------------------------------------
        # Retrieve
        # --------------------------------------------------------------

        raw_candidates = self.retrieve_candidates(
            retrieval_plan
        )

        # --------------------------------------------------------------
        # Deduplicate
        # --------------------------------------------------------------

        unique_candidates = self.deduplicate_candidates(
            raw_candidates
        )

        # --------------------------------------------------------------
        # Reranking
        # --------------------------------------------------------------

        # Use the original query for global reranking.
        reranked = self.rerank_candidates(
            query=query,
            candidates=unique_candidates,
        )

        # --------------------------------------------------------------
        # Final evidence
        # --------------------------------------------------------------

        final_evidence = self.select_final_evidence(
            reranked
        )

        # --------------------------------------------------------------
        # Return structured result
        # --------------------------------------------------------------

        return {
            "original_query": query,
            "query_analysis": processed,
            "domains": domains,
            "retrieval_plan": retrieval_plan,
            "raw_results": raw_candidates,
            "unique_results": unique_candidates,
            "reranked_results": reranked,
            "final_evidence": final_evidence,
            "statistics": {
                "raw_results": len(raw_candidates),
                "unique_results": len(unique_candidates),
                "reranked_results": len(reranked),
                "final_results": len(final_evidence),
            },
        }

    # ==================================================================
    # Pretty debug output
    # ==================================================================

    def debug_retrieve(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """
        Run retrieval and print a detailed diagnostic report.
        """

        result = self.retrieve(query)

        print("\n")
        print("=" * 80)
        print("ADVANCED RETRIEVAL RESULT")
        print("=" * 80)

        print("\nORIGINAL QUERY")
        print(result["original_query"])

        analysis = result["query_analysis"]

        print("\nQUERY ANALYSIS")

        print(
            "Domains:",
            ", ".join(result["domains"])
            if result["domains"]
            else "None",
        )

        print(
            "Primary domain:",
            analysis.get(
                "primary_domain",
                "None",
            ),
        )

        intents = analysis.get("intents", [])

        if isinstance(intents, list):
            intents_text = ", ".join(
                str(x) for x in intents
            )
        else:
            intents_text = str(intents)

        print(
            "Intents:",
            intents_text or "None",
        )

        print(
            "Primary intent:",
            analysis.get(
                "primary_intent",
                "None",
            ),
        )

        print(
            "Complex query:",
            analysis.get(
                "is_complex",
                analysis.get(
                    "complex_query",
                    False,
                ),
            ),
        )

        print("\nRETRIEVAL PLAN")

        for index, task in enumerate(
            result["retrieval_plan"],
            start=1,
        ):

            print(
                f"{index}. {task['query']}"
            )

            print(
                f"   Domain filter: {task.get('domain')}"
            )

        stats = result["statistics"]

        print("\nRETRIEVAL STATISTICS")

        print(
            "Raw results:",
            stats["raw_results"],
        )

        print(
            "Unique results:",
            stats["unique_results"],
        )

        print(
            "Reranked results:",
            stats["reranked_results"],
        )

        print(
            "Final results:",
            stats["final_results"],
        )

        print("\n")
        print("-" * 80)
        print("FINAL EVIDENCE")
        print("-" * 80)

        for index, item in enumerate(
            result["final_evidence"],
            start=1,
        ):

            print("\n")
            print("-" * 80)

            print(
                f"Final Rank: {index}"
            )

            print(
                f"Chunk: {item.get('chunk_id')}"
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
                "Source Queries:",
                ", ".join(
                    item.get(
                        "source_queries",
                        [],
                    )
                ),
            )

            print("\nTEXT:")

            print(
                item.get(
                    "text",
                    "",
                )
            )

        return result


# ======================================================================
# Standalone test
# ======================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("STEP 8A — MULTI-DOMAIN ADVANCED RETRIEVAL TEST")
    print("=" * 80)

    retriever = AdvancedRetriever()

    test_queries = [

        "Can a customer return a product?",

        "How long does a refund take?",

        "What payment methods are available?",

        "Can I return a laptop after 10 days and how long will my refund take?",

        "My order is delayed, what should I do and how long can delivery take?",

        "My payment failed and the amount was deducted, what should I do?",
    ]

    for query in test_queries:

        retriever.debug_retrieve(query)

    print("\n")
    print("=" * 80)
    print("STEP 8A TEST COMPLETE")
    print("=" * 80)