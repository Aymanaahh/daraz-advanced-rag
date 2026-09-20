from __future__ import annotations

import time
from typing import Any, Dict, List

import streamlit as st

from rag.advanced_retriever import AdvancedRetriever
from rag.generator import GroqGenerator
from rag.citation_validator import CitationValidator
from rag.logger import RAGLogger


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Daraz Advanced RAG Assistant",
    page_icon="🛒",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

@st.cache_resource
def load_components():

    retriever = AdvancedRetriever()
    generator = GroqGenerator()
    validator = CitationValidator()
    logger = RAGLogger()

    return retriever, generator, validator, logger


retriever, generator, validator, logger = load_components()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_get(obj: Any, key: str, default=None):

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def normalize_evidence(items: Any) -> List[Dict[str, Any]]:
    """
    Normalize retrieved evidence into a predictable list of dictionaries.
    """

    if items is None:
        return []

    if isinstance(items, dict):
        items = [items]

    if not isinstance(items, list):
        return []

    normalized = []

    for item in items:

        if isinstance(item, dict):

            normalized.append(item)

        else:

            try:
                normalized.append(vars(item))
            except Exception:
                continue

    return normalized


def get_evidence(result: Any) -> List[Dict[str, Any]]:
    """
    AdvancedRetriever currently returns evidence under final_results.
    """

    if not isinstance(result, dict):
        return []

    # Current architecture
    if "final_results" in result:

        evidence = normalize_evidence(
            result.get("final_results")
        )

        if evidence:
            return evidence

    # Compatibility fallbacks
    for key in [
        "evidence",
        "results",
        "documents",
        "chunks",
    ]:

        if key in result:

            evidence = normalize_evidence(
                result.get(key)
            )

            if evidence:
                return evidence

    return []


def get_query_analysis(result: Dict[str, Any]) -> Dict[str, Any]:

    analysis = result.get(
        "query_analysis",
        {}
    )

    if isinstance(analysis, dict):
        return analysis

    try:
        return vars(analysis)
    except Exception:
        return {}


def get_retrieval_queries(result: Dict[str, Any]) -> List[Dict[str, Any]]:

    queries = result.get(
        "retrieval_queries",
        []
    )

    if not isinstance(queries, list):
        return []

    normalized = []

    for item in queries:

        if isinstance(item, dict):
            normalized.append(item)

        else:

            try:
                normalized.append(vars(item))
            except Exception:
                continue

    return normalized


def get_retrieval_statistics(result: Dict[str, Any]) -> Dict[str, Any]:

    statistics = result.get(
        "statistics",
        {}
    )

    if isinstance(statistics, dict):
        return statistics

    try:
        return vars(statistics)
    except Exception:
        return {}


# ============================================================
# QUERY ANALYSIS DISPLAY
# ============================================================

def display_query_analysis(
    analysis: Dict[str, Any]
):

    if not analysis:
        return

    st.subheader("Query Analysis")

    domains = analysis.get(
        "domains",
        []
    )

    intents = analysis.get(
        "intents",
        []
    )

    primary_domain = analysis.get(
        "primary_domain"
    )

    complex_query = analysis.get(
        "complex_query",
        analysis.get(
            "is_complex",
            analysis.get(
                "complex",
                False
            )
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if domains:
            st.metric(
                "Detected Domains",
                len(domains)
            )
        else:
            st.metric(
                "Detected Domains",
                0
            )

    with col2:

        st.metric(
            "Complex Query",
            "Yes" if complex_query else "No"
        )

    with col3:

        st.write("**Primary Domain**")

        if primary_domain:
            st.write(primary_domain)
        else:
            st.write("—")

    with col4:

        st.write("**Primary Intent**")

        if intents:
            st.write(intents[0])
        else:
            st.write("—")

    if domains:

        st.write(
            "**Domains:** "
            + ", ".join(
                str(domain)
                for domain in domains
            )
        )

    if intents:

        st.write(
            "**Intents:** "
            + ", ".join(
                str(intent)
                for intent in intents
            )
        )


# ============================================================
# EVIDENCE DISPLAY
# ============================================================

def display_sources(
    evidence: List[Dict[str, Any]]
):

    st.subheader(
        f"Retrieved Evidence ({len(evidence)})"
    )

    if not evidence:

        st.info(
            "No evidence chunks were retrieved."
        )

        return

    st.caption(
        f"{len(evidence)} evidence chunks passed to the generation layer."
    )

    for index, item in enumerate(
        evidence,
        start=1
    ):

        chunk_id = item.get(
            "chunk_id",
            "Unknown"
        )

        file_name = item.get(
            "file_name",
            item.get(
                "source",
                "Unknown source"
            )
        )

        domain = item.get(
            "domain",
            "Unknown"
        )

        page = item.get(
            "page",
            "—"
        )

        section = item.get(
            "section",
            "—"
        )

        score = item.get(
            "reranker_score",
            item.get(
                "score"
            )
        )

        text = item.get(
            "text",
            item.get(
                "content",
                ""
            )
        )

        # Native Streamlit container.
        # This avoids HTML escaping/rendering problems on Streamlit Cloud.
        with st.container(border=True):

            top_col, score_col = st.columns(
                [4, 1]
            )

            with top_col:

                st.markdown(
                    f"**{index}. {chunk_id}**"
                )

            with score_col:

                if score is not None:

                    try:

                        st.metric(
                            "Reranker",
                            f"{float(score):.4f}"
                        )

                    except Exception:

                        st.write(
                            f"Score: {score}"
                        )

            st.caption(
                f"{file_name}  •  "
                f"Domain: {domain}  •  "
                f"Page: {page}  •  "
                f"Section: {section}"
            )

            if text:

                st.write(text)


# ============================================================
# CITATION VALIDATION DISPLAY
# ============================================================

def display_validation(
    validation: Dict[str, Any]
):

    st.subheader(
        "Citation Validation"
    )

    if not validation:

        st.warning(
            "Citation validation information is unavailable."
        )

        return

    valid = validation.get(
        "valid",
        False
    )

    citation_count = validation.get(
        "citation_count",
        0
    )

    valid_count = validation.get(
        "valid_citation_count",
        0
    )

    # Validator stores coverage as a ratio:
    # 0.5 means 50%.
    coverage_ratio = validation.get(
        "coverage",
        0.0
    )

    try:

        coverage_percentage = (
            float(coverage_ratio) * 100
        )

    except Exception:

        coverage_percentage = 0.0

    if valid:

        st.success(
            "✅ Citation validation passed."
        )

    else:

        st.warning(
            "⚠️ Citation validation found issues."
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Valid Citations",
            f"{valid_count}/{citation_count}"
        )

    with col2:

        st.metric(
            "Coverage",
            f"{coverage_percentage:.1f}%"
        )

    with col3:

        invalid_count = len(
            validation.get(
                "invalid_citations",
                []
            )
        )

        st.metric(
            "Invalid Citations",
            invalid_count
        )

    invalid = validation.get(
        "invalid_citations",
        []
    )

    if invalid:

        st.error(
            "Invalid citations: "
            + ", ".join(
                str(item)
                for item in invalid
            )
        )

    issues = validation.get(
        "issues",
        []
    )

    if issues:

        with st.expander(
            "Validation Issues"
        ):

            for issue in issues:

                st.write(
                    f"- {issue}"
                )


# ============================================================
# PIPELINE DETAILS
# ============================================================

def display_pipeline_details(
    result: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    citations: List[str],
    elapsed: float,
):

    st.subheader(
        "Pipeline Details"
    )

    statistics = get_retrieval_statistics(
        result
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        retrieved_count = statistics.get(
            "final_results",
            len(evidence)
        )

        st.metric(
            "Retrieved Chunks",
            retrieved_count
        )

    with col2:

        st.metric(
            "Citations",
            len(citations)
        )

    with col3:

        st.metric(
            "Response Time",
            f"{elapsed:.2f}s"
        )

    with col4:

        st.metric(
            "Evidence Passed",
            len(evidence)
        )


# ============================================================
# LOGGING
# ============================================================

def write_log(
    query: str,
    result: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    answer: str,
    citations: List[str],
    validation: Dict[str, Any],
    elapsed: float,
):

    """
    Write the complete RAG request using the actual
    RAGLogger.log_rag_request() signature.
    """

    try:

        analysis = get_query_analysis(
            result
        )

        retrieval_queries = get_retrieval_queries(
            result
        )

        retrieval_statistics = get_retrieval_statistics(
            result
        )

        logger.log_rag_request(

            query=query,

            query_analysis=analysis,

            retrieval_queries=retrieval_queries,

            evidence=evidence,

            retrieval_statistics=retrieval_statistics,

            answer=answer,

            citations=citations,

            citation_validation=validation,

            total_execution_time_ms=(
                elapsed * 1000
            ),

            metadata={
                "application": "daraz-advanced-rag",
                "environment": "streamlit-cloud",
            },
        )

        return True, None

    except Exception as exc:

        return False, exc


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def process_question(
    query: str
):

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    result = retriever.retrieve(
        query
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence = get_evidence(
        result
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    generation_start = time.perf_counter()

    answer = generator.generate(
        query=query,
        evidence=evidence,
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    # --------------------------------------------------------
    # Citation extraction
    # --------------------------------------------------------

    citations = validator.extract_citations(
        answer
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_start = time.perf_counter()

    validation = validator.validate(
        answer=answer,
        evidence=evidence,
    )

    validation_time = (
        time.perf_counter()
        - validation_start
    )

    # --------------------------------------------------------
    # Total time
    # --------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log_success, log_error = write_log(
        query=query,
        result=result,
        evidence=evidence,
        answer=answer,
        citations=citations,
        validation=validation,
        elapsed=elapsed,
    )

    return {
        "result": result,
        "evidence": evidence,
        "answer": answer,
        "citations": citations,
        "validation": validation,
        "elapsed": elapsed,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "validation_time": validation_time,
        "log_success": log_success,
        "log_error": log_error,
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🛒 Daraz RAG"
    )

    st.markdown(
        """
### Advanced Operations Assistant

Ask questions about:

- Sellers
- Refunds
- Delivery
- Payments
- Returns
- Customer Support
"""
    )

    st.divider()

    st.markdown(
        "**Example questions**"
    )

    examples = [
        "How long does a refund take?",
        "What payment methods are available?",
        "Can a customer return a product?",
        "What are the delivery timelines?",
        "What should a seller do during onboarding?",
    ]

    for example in examples:

        if st.button(
            example,
            use_container_width=True,
        ):

            st.session_state.selected_example = example


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛒 Daraz Operations & Policy Assistant"
)

st.write(
    "Enterprise-style hybrid RAG with semantic retrieval, "
    "BM25 search, reranking, grounded generation, "
    "citation validation, and request logging."
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# INPUT
# ============================================================

selected_example = st.session_state.get(
    "selected_example"
)

query = st.chat_input(
    "Ask a Daraz operations or policy question..."
)

if selected_example and not query:

    query = selected_example

    # Prevent repeated automatic execution
    st.session_state.selected_example = None


# ============================================================
# EXECUTE
# ============================================================

if query:

    query = query.strip()

    if not query:

        st.warning(
            "Please enter a question."
        )

    else:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(query)

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Running advanced RAG pipeline..."
            ):

                try:

                    output = process_question(
                        query
                    )

                    result = output["result"]

                    evidence = output["evidence"]

                    answer = output["answer"]

                    citations = output["citations"]

                    validation = output["validation"]

                    elapsed = output["elapsed"]

                    # ------------------------------------------------
                    # Pipeline status
                    # ------------------------------------------------

                    st.success(
                        "RAG pipeline completed"
                    )

                    # ------------------------------------------------
                    # Answer
                    # ------------------------------------------------

                    st.subheader(
                        "Answer"
                    )

                    st.markdown(
                        answer
                    )

                    # ------------------------------------------------
                    # Validation
                    # ------------------------------------------------

                    display_validation(
                        validation
                    )

                    # ------------------------------------------------
                    # Query analysis
                    # ------------------------------------------------

                    display_query_analysis(
                        get_query_analysis(
                            result
                        )
                    )

                    # ------------------------------------------------
                    # Evidence
                    # ------------------------------------------------

                    display_sources(
                        evidence
                    )

                    # ------------------------------------------------
                    # Pipeline details
                    # ------------------------------------------------

                    display_pipeline_details(
                        result=result,
                        evidence=evidence,
                        citations=citations,
                        elapsed=elapsed,
                    )

                    # ------------------------------------------------
                    # Logging status
                    # ------------------------------------------------

                    if output["log_success"]:

                        st.caption(
                            "✅ RAG request logged successfully."
                        )

                    else:

                        st.warning(
                            "⚠️ Response generated, but request logging failed."
                        )

                        if output["log_error"]:

                            st.caption(
                                f"Logging error: {output['log_error']}"
                            )

                    # ------------------------------------------------
                    # Save assistant response
                    # ------------------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                    st.session_state.last_result = output

                except Exception as exc:

                    st.error(
                        "The RAG pipeline encountered an error."
                    )

                    st.exception(
                        exc
                    )