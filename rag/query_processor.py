"""
Step 5 — Query Processor / Query Decomposition

Responsibilities:
1. Normalize the user's query
2. Detect likely domain(s)
3. Detect query intent
4. Detect whether the query is complex
5. Decompose complex queries into focused sub-queries
6. Generate retrieval queries
7. Preserve the original user query

This module does NOT call the LLM.
The LLM will be introduced in the Generator stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


# ============================================================
# DOMAIN CONFIGURATION
# ============================================================

DOMAIN_KEYWORDS = {
    "returns": [
        "return",
        "returns",
        "returning",
        "eligible",
        "eligibility",
        "return window",
        "returned",
        "send back",
        "wrong item",
        "damaged item",
        "defective item",
    ],

    "refunds": [
        "refund",
        "refunds",
        "refunded",
        "money back",
        "refund time",
        "refund status",
        "refund amount",
        "refund method",
        "reimbursement",
    ],

    "delivery": [
        "delivery",
        "delivered",
        "delivery time",
        "shipping",
        "shipment",
        "courier",
        "delivery delay",
        "failed delivery",
        "delivery location",
    ],

    "payments": [
        "payment",
        "payments",
        "pay",
        "paid",
        "payment method",
        "card",
        "credit card",
        "debit card",
        "wallet",
        "cash on delivery",
        "cod",
        "bank transfer",
        "installment",
        "voucher",
        "promo code",
    ],

    "sellers": [
        "seller",
        "sellers",
        "selling",
        "seller account",
        "seller onboarding",
        "seller policy",
        "product listing",
        "seller registration",
    ],

    "customer_support": [
        "support",
        "customer support",
        "complaint",
        "complaints",
        "contact support",
        "live chat",
        "phone support",
        "email support",
        "escalation",
    ],
}


# ============================================================
# INTENT KEYWORDS
# ============================================================

INTENT_PATTERNS = {
    "eligibility": [
        "can i",
        "can a",
        "am i eligible",
        "is it eligible",
        "eligible",
        "allowed",
        "qualify",
        "qualification",
        "can this",
        "can the customer",
    ],

    "procedure": [
        "how do i",
        "how can i",
        "how to",
        "what is the process",
        "process",
        "steps",
        "procedure",
        "what should i do",
    ],

    "timeline": [
        "how long",
        "when",
        "how soon",
        "days",
        "hours",
        "deadline",
        "time",
        "timeline",
        "within",
    ],

    "availability": [
        "available",
        "what options",
        "which options",
        "what methods",
        "which methods",
        "payment methods",
        "support channels",
    ],

    "troubleshooting": [
        "failed",
        "failure",
        "problem",
        "issue",
        "error",
        "not working",
        "didn't work",
        "cannot",
        "can't",
        "unable",
    ],

    "status": [
        "status",
        "where is",
        "what is happening",
        "track",
        "tracking",
        "check",
        "current status",
    ],

    "policy": [
        "policy",
        "rule",
        "rules",
        "guideline",
        "guidelines",
        "conditions",
        "requirement",
        "requirements",
    ],

    "general_information": [
        "what is",
        "what are",
        "tell me",
        "explain",
        "information",
        "details",
    ],
}


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class ProcessedQuery:
    original_query: str
    normalized_query: str

    domains: List[str]
    primary_domain: str

    intents: List[str]
    primary_intent: str

    is_complex: bool
    sub_queries: List[str]

    retrieval_queries: List[str]

    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# QUERY PROCESSOR
# ============================================================

class QueryProcessor:

    def __init__(self):
        self.domain_keywords = DOMAIN_KEYWORDS
        self.intent_patterns = INTENT_PATTERNS

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def normalize(query: str) -> str:
        """
        Normalize whitespace and punctuation while preserving
        the semantic content of the user's query.
        """

        if not query:
            return ""

        query = query.strip()

        # Normalize whitespace
        query = re.sub(r"\s+", " ", query)

        return query

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        """
        Detect a keyword or phrase safely.
        """

        return phrase.lower() in text.lower()

    # --------------------------------------------------------
    # DOMAIN DETECTION
    # --------------------------------------------------------

    def detect_domains(self, query: str) -> List[str]:
        """
        Detect one or more likely domains.

        A query can belong to multiple domains.
        Example:

        "Can I return this item and how long will the refund take?"

        -> returns
        -> refunds
        """

        text = query.lower()

        scores = {}

        for domain, keywords in self.domain_keywords.items():

            score = 0

            for keyword in keywords:

                if self._contains_phrase(text, keyword):
                    score += 1

            if score > 0:
                scores[domain] = score

        if not scores:
            return ["general"]

        # Sort by number of keyword matches
        ranked_domains = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Keep domains with meaningful matches
        domains = [domain for domain, score in ranked_domains if score > 0]

        return domains

    # --------------------------------------------------------
    # INTENT DETECTION
    # --------------------------------------------------------

    def detect_intents(self, query: str) -> List[str]:

        text = query.lower()

        scores = {}

        for intent, patterns in self.intent_patterns.items():

            score = 0

            for pattern in patterns:

                if self._contains_phrase(text, pattern):
                    score += 1

            if score > 0:
                scores[intent] = score

        if not scores:
            return ["general_information"]

        ranked_intents = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [intent for intent, score in ranked_intents]

    # --------------------------------------------------------
    # COMPLEXITY DETECTION
    # --------------------------------------------------------

    @staticmethod
    def is_complex_query(query: str) -> bool:

        text = query.lower()

        # Multiple questions
        question_marks = text.count("?")

        if question_marks > 1:
            return True

        # Multiple conjunctions usually indicate multiple requests
        conjunctions = [
            " and ",
            " also ",
            " as well as ",
            " plus ",
            " then ",
            " while ",
        ]

        conjunction_count = sum(
            text.count(conjunction)
            for conjunction in conjunctions
        )

        if conjunction_count >= 1:
            return True

        # Explicit multi-part wording
        multi_part_patterns = [
            "first",
            "second",
            "both",
            "multiple",
            "two questions",
            "two things",
            "separately",
        ]

        if any(pattern in text for pattern in multi_part_patterns):
            return True

        return False

    # --------------------------------------------------------
    # QUERY SPLITTING
    # --------------------------------------------------------

    @staticmethod
    def _clean_subquery(query: str) -> str:

        query = query.strip()

        query = re.sub(
            r"^(and|also|then|plus|first|second)[,\s]+",
            "",
            query,
            flags=re.IGNORECASE,
        )

        query = query.strip(" ,.;:")

        if query and not query.endswith("?"):
            query += "?"

        return query

    def split_into_subqueries(self, query: str) -> List[str]:
        """
        Lightweight rule-based decomposition.

        This is intentionally deterministic.
        """

        normalized = self.normalize(query)

        # ----------------------------------------------------
        # Split on question marks
        # ----------------------------------------------------

        if normalized.count("?") > 1:

            parts = normalized.split("?")

            subqueries = []

            for part in parts:

                part = self._clean_subquery(part)

                if len(part) > 8:
                    subqueries.append(part)

            if len(subqueries) > 1:
                return subqueries

        # ----------------------------------------------------
        # Handle common conjunction patterns
        # ----------------------------------------------------

        patterns = [
            r"\s+and\s+how\s+",
            r"\s+and\s+what\s+",
            r"\s+and\s+can\s+",
            r"\s+and\s+is\s+",
            r"\s+and\s+when\s+",
            r"\s+also\s+",
            r"\s+as well as\s+",
        ]

        for pattern in patterns:

            parts = re.split(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            )

            if len(parts) > 1:

                subqueries = []

                for index, part in enumerate(parts):

                    part = self._clean_subquery(part)

                    if len(part) > 8:
                        subqueries.append(part)

                if len(subqueries) > 1:
                    return subqueries

        # ----------------------------------------------------
        # No decomposition required
        # ----------------------------------------------------

        return [normalized]

    # --------------------------------------------------------
    # QUERY REWRITING
    # --------------------------------------------------------

    def rewrite_query(
        self,
        query: str,
        domain: str,
        intent: str,
    ) -> str:
        """
        Produce a retrieval-friendly version of a query.

        The rewrite does not replace the user's original query.
        It is only used to improve retrieval.
        """

        query = query.strip()

        if not query:
            return query

        # Domain-specific retrieval terms
        domain_terms = {
            "returns": "return eligibility return window return conditions",
            "refunds": "refund process refund timeline refund method",
            "delivery": "delivery process delivery timeline delivery conditions",
            "payments": "payment methods payment options payment process",
            "sellers": "seller policy seller onboarding seller requirements",
            "customer_support": "customer support complaint escalation support channels",
        }

        intent_terms = {
            "eligibility": "eligibility requirements conditions",
            "procedure": "process procedure steps",
            "timeline": "timeline processing time duration",
            "availability": "available options methods",
            "troubleshooting": "problem troubleshooting failed payment issue",
            "status": "status tracking current status",
            "policy": "policy rules guidelines requirements",
            "general_information": "",
        }

        additions = []

        if domain in domain_terms:
            additions.append(domain_terms[domain])

        if intent in intent_terms and intent_terms[intent]:
            additions.append(intent_terms[intent])

        if additions:
            return query + " " + " ".join(additions)

        return query

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    @staticmethod
    def calculate_confidence(
        domains: List[str],
        intents: List[str],
        query: str,
    ) -> float:
        """
        Simple deterministic confidence estimate.

        This is NOT an ML probability.
        It is an internal heuristic.
        """

        score = 0.50

        if domains and domains[0] != "general":
            score += 0.20

        if intents:
            score += 0.15

        if len(domains) > 1:
            score += 0.05

        if len(query.split()) >= 5:
            score += 0.05

        return min(score, 0.95)

    # --------------------------------------------------------
    # MAIN PROCESSOR
    # --------------------------------------------------------

    def process(self, query: str) -> ProcessedQuery:

        original_query = query

        normalized_query = self.normalize(query)

        if not normalized_query:
            return ProcessedQuery(
                original_query=original_query,
                normalized_query="",
                domains=["general"],
                primary_domain="general",
                intents=["general_information"],
                primary_intent="general_information",
                is_complex=False,
                sub_queries=[],
                retrieval_queries=[],
                confidence=0.0,
            )

        # Domain detection
        domains = self.detect_domains(normalized_query)

        primary_domain = domains[0]

        # Intent detection
        intents = self.detect_intents(normalized_query)

        primary_intent = intents[0]

        # Complexity
        complex_query = self.is_complex_query(normalized_query)

        # Decomposition
        if complex_query:

            sub_queries = self.split_into_subqueries(
                normalized_query
            )

        else:

            sub_queries = [normalized_query]

        # ----------------------------------------------------
        # Generate retrieval queries
        # ----------------------------------------------------

        retrieval_queries = []

        for subquery in sub_queries:

            sub_domains = self.detect_domains(subquery)

            sub_intents = self.detect_intents(subquery)

            sub_domain = (
                sub_domains[0]
                if sub_domains
                else primary_domain
            )

            sub_intent = (
                sub_intents[0]
                if sub_intents
                else primary_intent
            )

            rewritten = self.rewrite_query(
                subquery,
                sub_domain,
                sub_intent,
            )

            retrieval_queries.append(rewritten)

        # Remove duplicates while preserving order
        retrieval_queries = list(
            dict.fromkeys(retrieval_queries)
        )

        confidence = self.calculate_confidence(
            domains,
            intents,
            normalized_query,
        )

        return ProcessedQuery(
            original_query=original_query,
            normalized_query=normalized_query,
            domains=domains,
            primary_domain=primary_domain,
            intents=intents,
            primary_intent=primary_intent,
            is_complex=complex_query,
            sub_queries=sub_queries,
            retrieval_queries=retrieval_queries,
            confidence=confidence,
        )


# ============================================================
# TEST / DEMO
# ============================================================

def print_result(result: ProcessedQuery):

    print("\n" + "=" * 70)

    print("ORIGINAL QUERY")
    print(result.original_query)

    print("\nNORMALIZED QUERY")
    print(result.normalized_query)

    print("\nDOMAINS")
    print(", ".join(result.domains))

    print("\nPRIMARY DOMAIN")
    print(result.primary_domain)

    print("\nINTENTS")
    print(", ".join(result.intents))

    print("\nPRIMARY INTENT")
    print(result.primary_intent)

    print("\nCOMPLEX QUERY")
    print(result.is_complex)

    print("\nSUB-QUERIES")

    for index, query in enumerate(
        result.sub_queries,
        start=1,
    ):
        print(f"{index}. {query}")

    print("\nRETRIEVAL QUERIES")

    for index, query in enumerate(
        result.retrieval_queries,
        start=1,
    ):
        print(f"{index}. {query}")

    print("\nCONFIDENCE")
    print(f"{result.confidence:.2f}")

    print("=" * 70)


def main():

    print("\n")
    print("=" * 70)
    print("STEP 5 — QUERY PROCESSOR / QUERY DECOMPOSITION TEST")
    print("=" * 70)

    processor = QueryProcessor()

    test_queries = [

        # Simple query
        "Can a customer return a product?",

        # Timeline
        "How long does a refund take?",

        # Payment
        "What payment methods are available?",

        # Complex query
        "Can I return a laptop after 10 days and how long will my refund take?",

        # Delivery
        "My order is delayed, what should I do and how long can delivery take?",

        # Seller
        "What are the requirements for seller onboarding?",

        # Troubleshooting
        "My payment failed and the amount was deducted, what should I do?",
    ]

    for query in test_queries:

        result = processor.process(query)

        print_result(result)

    print("\n")
    print("=" * 70)
    print("STEP 5 TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()