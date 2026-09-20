import os
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from rag.advanced_retriever import AdvancedRetriever
from rag.generator import GroqGenerator
from rag.citation_validator import CitationValidator
from rag.logger import RAGLogger


# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

st.set_page_config(
    page_title="Daraz Operations AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .app-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #6b7280;
        }

        .source-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            background: #ffffff;
        }

        .source-id {
            font-weight: 700;
            font-size: 0.95rem;
        }

        .source-meta {
            color: #6b7280;
            font-size: 0.82rem;
            margin-top: 0.25rem;
            margin-bottom: 0.6rem;
        }

        .source-text {
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .status-success {
            padding: 0.7rem;
            border-radius: 8px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
        }

        .status-warning {
            padding: 0.7rem;
            border-radius: 8px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }

        .citation-badge {
            display: inline-block;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 6px;
            padding: 2px 7px;
            margin: 2px;
            font-size: 0.78rem;
        }

        .history-item {
            padding: 0.5rem 0;
            border-bottom: 1px solid #e5e7eb;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "system_initialized" not in st.session_state:
    st.session_state.system_initialized = False


# =============================================================================
# LOAD RAG COMPONENTS
# =============================================================================

@st.cache_resource
def load_rag_system():
    """
    Load the complete RAG pipeline once and cache it.

    This prevents the embedding model, reranker and Groq-related
    components from being recreated on every Streamlit interaction.
    """

    retriever = AdvancedRetriever()
    generator = GroqGenerator()
    validator = CitationValidator()
    logger = RAGLogger()

    return retriever, generator, validator, logger


try:
    retriever, generator, validator, logger = load_rag_system()
    st.session_state.system_initialized = True
except Exception as exc:
    st.error("The RAG system could not be initialized.")
    st.exception(exc)

    st.stop()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_get(obj, key, default=None):
    """
    Safely retrieve a value from either a dictionary or an object.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def normalize_evidence(evidence):
    """
    Convert different possible retriever output structures into
    a simple list of evidence dictionaries.
    """

    if evidence is None:
        return []

    if isinstance(evidence, dict):
        for key in ["evidence", "results", "documents", "chunks"]:
            if key in evidence and isinstance(evidence[key], list):
                evidence = evidence[key]
                break

    if not isinstance(evidence, list):
        try:
            evidence = list(evidence)
        except Exception:
            return []

    normalized = []

    for item in evidence:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            normalized.append(
                {
                    "chunk_id": safe_get(item, "chunk_id"),
                    "document_id": safe_get(item, "document_id"),
                    "file_name": safe_get(item, "file_name"),
                    "domain": safe_get(item, "domain"),
                    "document_type": safe_get(item, "document_type"),
                    "page": safe_get(item, "page"),
                    "section": safe_get(item, "section"),
                    "text": safe_get(item, "text", ""),
                    "score": safe_get(item, "score"),
                }
            )

    return normalized


def get_evidence(result):
    """
    Extract evidence from AdvancedRetriever output.
    """

    if isinstance(result, dict):
        for key in ["evidence", "results", "documents", "chunks"]:
            if key in result:
                return normalize_evidence(result[key])

    for key in ["evidence", "results", "documents", "chunks"]:
        value = safe_get(result, key)
        if value is not None:
            return normalize_evidence(value)

    if isinstance(result, list):
        return normalize_evidence(result)

    return []


def get_processed_query(result):
    """
    Extract query analysis information from retriever output.
    """

    if isinstance(result, dict):
        for key in ["query_analysis", "analysis", "processed_query"]:
            if key in result:
                return result[key]

    for key in ["query_analysis", "analysis", "processed_query"]:
        value = safe_get(result, key)
        if value is not None:
            return value

    return None


def call_retriever(query):
    """
    Call the existing AdvancedRetriever.

    The current project uses:
        retriever.retrieve(query)
    """

    if hasattr(retriever, "retrieve"):
        return retriever.retrieve(query)

    raise AttributeError(
        "AdvancedRetriever does not expose a retrieve(query) method."
    )


def call_generator(query, evidence):
    """
    Call the existing Groq generator.

    The generator is expected to accept the original question and
    retrieved evidence.
    """

    if hasattr(generator, "generate"):
        return generator.generate(query, evidence)

    raise AttributeError(
        "GroqGenerator does not expose a generate(query, evidence) method."
    )


def call_validator(answer, evidence):
    """
    Validate generated citations against retrieved evidence.
    """

    if hasattr(validator, "validate"):
        return validator.validate(answer, evidence)

    raise AttributeError(
        "CitationValidator does not expose a validate(answer, evidence) method."
    )


def extract_answer(generated):
    """
    Normalize different possible generator response formats.
    """

    if generated is None:
        return ""

    if isinstance(generated, str):
        return generated

    if isinstance(generated, dict):
        for key in ["answer", "response", "text", "content"]:
            if key in generated:
                return str(generated[key])

    for key in ["answer", "response", "text", "content"]:
        value = safe_get(generated, key)
        if value is not None:
            return str(value)

    return str(generated)


def extract_citations(answer):
    """
    Extract source IDs from the generated answer.
    """

    import re

    if not answer:
        return []

    pattern = r"\[SOURCE:\s*([A-Za-z0-9_-]+)\]"
    citations = re.findall(pattern, answer)

    return list(dict.fromkeys(citations))


def format_score(score):
    if score is None:
        return "N/A"

    try:
        return f"{float(score):.4f}"
    except Exception:
        return str(score)


def display_query_analysis(analysis):
    """
    Display query decomposition and intent information.
    """

    if analysis is None:
        return

    domains = safe_get(analysis, "domains", [])
    intents = safe_get(analysis, "intents", [])
    primary_domain = safe_get(analysis, "primary_domain")
    complex_query = safe_get(analysis, "complex", False)
    subqueries = safe_get(analysis, "subqueries", [])

    if isinstance(domains, str):
        domains = [domains]

    if isinstance(intents, str):
        intents = [intents]

    st.markdown("### 🔎 Query Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {primary_domain or "General"}
                </div>
                <div class="metric-label">Primary Domain</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {len(domains)}
                </div>
                <div class="metric-label">Detected Domains</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">
                    {"Yes" if complex_query else "No"}
                </div>
                <div class="metric-label">Complex Query</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if domains:
        st.write("**Domains:**", ", ".join(str(x) for x in domains))

    if intents:
        st.write("**Intents:**", ", ".join(str(x) for x in intents))

    if subqueries:
        with st.expander("View decomposed subqueries"):
            for index, subquery in enumerate(subqueries, start=1):
                st.write(f"**{index}.** {subquery}")


def display_sources(evidence):
    """
    Render retrieved evidence/source cards.
    """

    st.markdown("### 📚 Retrieved Evidence")

    if not evidence:
        st.warning("No evidence was retrieved for this question.")
        return

    st.caption(
        f"{len(evidence)} evidence chunk(s) passed to the generation layer."
    )

    for index, item in enumerate(evidence, start=1):

        chunk_id = item.get("chunk_id", "Unknown")
        document_id = item.get("document_id", "Unknown")
        file_name = item.get("file_name", "Unknown")
        domain = item.get("domain", "Unknown")
        document_type = item.get("document_type", "Unknown")
        page = item.get("page", "N/A")
        section = item.get("section", "N/A")
        score = item.get("score")
        text = item.get("text", "")

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-id">
                    {index}. {chunk_id}
                </div>

                <div class="source-meta">
                    Document: {file_name}
                    &nbsp; | &nbsp;
                    Domain: {domain}
                    &nbsp; | &nbsp;
                    Page: {page}
                    &nbsp; | &nbsp;
                    Score: {format_score(score)}
                </div>

                <div class="source-meta">
                    Section: {section}
                    &nbsp; | &nbsp;
                    Type: {document_type}
                    &nbsp; | &nbsp;
                    Document ID: {document_id}
                </div>

                <div class="source-text">
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_validation(validation):
    """
    Render citation validation result.
    """

    if validation is None:
        return

    valid = safe_get(validation, "valid", False)
    coverage = safe_get(validation, "coverage", 0)
    citation_count = safe_get(validation, "citation_count", 0)
    valid_count = safe_get(validation, "valid_citation_count", 0)

    if valid:
        st.markdown(
            f"""
            <div class="status-success">
                ✅ <strong>Citation validation passed.</strong><br>
                {valid_count}/{citation_count} citations are valid.
                Coverage: {float(coverage):.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="status-warning">
                ⚠️ <strong>Citation validation requires attention.</strong><br>
                Valid citations: {valid_count}/{citation_count}.
                Coverage: {float(coverage):.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

    invalid = safe_get(validation, "invalid_citations", [])

    if invalid:
        st.write("**Invalid citations:**")
        for citation in invalid:
            st.code(str(citation))

    not_cited = safe_get(validation, "retrieved_not_cited", [])

    if not_cited:
        st.write("**Retrieved evidence not cited:**")
        for citation in not_cited:
            st.code(str(citation))

    issues = safe_get(validation, "issues", [])

    if issues:
        st.write("**Validation issues:**")
        for issue in issues:
            st.warning(str(issue))


def write_log(query, result, answer, evidence, citations, validation, elapsed):
    """
    Write the complete request to the existing RAG logger.
    """

    analysis = get_processed_query(result)

    try:
        if hasattr(logger, "log_rag_request"):

            logger.log_rag_request(
                query=query,
                analysis=analysis,
                retrieval_queries=[],
                evidence=evidence,
                answer=answer,
                citations=citations,
                validation=validation,
                timing={
                    "total_seconds": elapsed
                },
                metadata={
                    "application": "Daraz Operations AI",
                    "timestamp": datetime.now().isoformat(),
                },
            )

            return True

    except TypeError:
        # Compatibility fallback for slightly different logger signatures.
        try:
            logger.log_rag_request(
                query=query,
                answer=answer,
                citations=citations,
                evidence=evidence,
                validation=validation,
            )

            return True

        except Exception:
            pass

    except Exception:
        pass

    return False


def process_question(query):
    """
    Execute the complete runtime RAG pipeline.
    """

    start_time = time.perf_counter()

    # -------------------------------------------------------------------------
    # 1. RETRIEVAL
    # -------------------------------------------------------------------------

    retrieval_result = call_retriever(query)

    evidence = get_evidence(retrieval_result)
    analysis = get_processed_query(retrieval_result)

    # -------------------------------------------------------------------------
    # 2. GENERATION
    # -------------------------------------------------------------------------

    generated = call_generator(query, evidence)
    answer = extract_answer(generated)

    # -------------------------------------------------------------------------
    # 3. CITATION EXTRACTION
    # -------------------------------------------------------------------------

    citations = extract_citations(answer)

    # -------------------------------------------------------------------------
    # 4. CITATION VALIDATION
    # -------------------------------------------------------------------------

    validation = call_validator(answer, evidence)

    # -------------------------------------------------------------------------
    # 5. LOGGING
    # -------------------------------------------------------------------------

    elapsed = time.perf_counter() - start_time

    log_written = write_log(
        query=query,
        result=retrieval_result,
        answer=answer,
        evidence=evidence,
        citations=citations,
        validation=validation,
        elapsed=elapsed,
    )

    return {
        "query": query,
        "result": retrieval_result,
        "analysis": analysis,
        "evidence": evidence,
        "answer": answer,
        "citations": citations,
        "validation": validation,
        "elapsed": elapsed,
        "log_written": log_written,
    }


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.markdown("## 🛒 Daraz Operations AI")

    st.caption(
        "Enterprise-style RAG decision-support assistant "
        "for operational and policy questions."
    )

    st.divider()

    st.markdown("### ⚙️ System Status")

    st.success("RAG system initialized")

    st.write("**Vector Store:** FAISS")
    st.write("**Lexical Search:** BM25")
    st.write("**Reranker:** BGE Cross Encoder")
    st.write("**LLM:** Groq")
    st.write("**Citations:** Enabled")
    st.write("**Logging:** JSONL")

    st.divider()

    st.markdown("### 📂 Knowledge Domains")

    domains = [
        "🛍️ Sellers",
        "💰 Refunds",
        "🚚 Delivery",
        "💳 Payments",
        "↩️ Returns",
        "🎧 Customer Support",
    ]

    for domain in domains:
        st.write(domain)

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()

    st.markdown("### 💡 Example Questions")

    example_questions = [
        "How long does a refund take?",
        "What payment methods are available?",
        "Can a customer return a product?",
        "My order is delayed. What should I do?",
        "Can I return a laptop after 10 days and how long will my refund take?",
        "What should a new seller do to get started?",
    ]

    for example in example_questions:
        if st.button(
            example,
            key=f"example_{example}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example


# =============================================================================
# MAIN HEADER
# =============================================================================

st.markdown(
    '<div class="app-title">🛒 Daraz Operations AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
        Grounded RAG assistant for Daraz operational policies,
        procedures, customer support, sellers, payments, refunds,
        returns, and delivery.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# TOP METRICS
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Knowledge Chunks", "21")

with col2:
    st.metric("Knowledge Domains", "6")

with col3:
    st.metric("Retrieval", "Hybrid")

with col4:
    st.metric("Grounding", "Citations")


st.divider()


# =============================================================================
# CHAT HISTORY
# =============================================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            result = message.get("result")

            if result:

                citations = result.get("citations", [])

                if citations:
                    st.markdown("**Sources:**")

                    for citation in citations:
                        st.markdown(
                            f'<span class="citation-badge">{citation}</span>',
                            unsafe_allow_html=True,
                        )


# =============================================================================
# INPUT
# =============================================================================

pending_question = st.session_state.pop("pending_question", None)

query = st.chat_input(
    "Ask a question about Daraz operations, policies, refunds, returns, delivery, payments, or sellers..."
)

if pending_question and not query:
    query = pending_question


# =============================================================================
# PROCESS QUESTION
# =============================================================================

if query:

    query = query.strip()

    if not query:
        st.warning("Please enter a question.")
        st.stop()

    # -------------------------------------------------------------------------
    # USER MESSAGE
    # -------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # -------------------------------------------------------------------------
    # ASSISTANT PROCESSING
    # -------------------------------------------------------------------------

    with st.chat_message("assistant"):

        progress = st.status(
            "Running RAG pipeline...",
            expanded=True,
        )

        try:

            progress.write("🔎 Analyzing query...")
            progress.write("🔍 Searching FAISS + BM25...")
            progress.write("🎯 Reranking retrieved evidence...")
            progress.write("🤖 Generating grounded answer...")
            progress.write("🔗 Validating citations...")
            progress.write("📝 Writing retrieval log...")

            result = process_question(query)

            progress.update(
                label="RAG pipeline completed",
                state="complete",
                expanded=False,
            )

        except Exception as exc:

            progress.update(
                label="RAG pipeline failed",
                state="error",
                expanded=True,
            )

            st.error("An error occurred while processing the question.")
            st.exception(exc)

            st.stop()

        # ---------------------------------------------------------------------
        # ANSWER
        # ---------------------------------------------------------------------

        st.markdown("## Answer")

        st.markdown(result["answer"])

        # ---------------------------------------------------------------------
        # CITATIONS
        # ---------------------------------------------------------------------

        if result["citations"]:

            st.markdown("### 🔗 Citations")

            for citation in result["citations"]:
                st.markdown(
                    f'<span class="citation-badge">{citation}</span>',
                    unsafe_allow_html=True,
                )

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        with st.expander("🛡️ Citation Validation", expanded=False):

            display_validation(result["validation"])

        # ---------------------------------------------------------------------
        # QUERY ANALYSIS
        # ---------------------------------------------------------------------

        with st.expander("🧠 Query Analysis", expanded=False):

            display_query_analysis(result["analysis"])

        # ---------------------------------------------------------------------
        # EVIDENCE
        # ---------------------------------------------------------------------

        with st.expander(
            f"📚 Retrieved Evidence ({len(result['evidence'])})",
            expanded=False,
        ):

            display_sources(result["evidence"])

        # ---------------------------------------------------------------------
        # PIPELINE METADATA
        # ---------------------------------------------------------------------

        with st.expander("⚙️ Pipeline Details", expanded=False):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Retrieved Chunks",
                    len(result["evidence"]),
                )

            with col2:
                st.metric(
                    "Citations",
                    len(result["citations"]),
                )

            with col3:
                st.metric(
                    "Response Time",
                    f"{result['elapsed']:.2f}s",
                )

            if result["log_written"]:
                st.success("Request logged successfully.")
            else:
                st.warning("Response generated, but request logging failed.")

    # -------------------------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # -------------------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "result": result,
        }
    )

    st.session_state.last_result = result


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Daraz Operations AI • Hybrid RAG • FAISS + BM25 • "
    "BGE Reranking • Groq LLM • Citation Validation • Retrieval Logging"
)