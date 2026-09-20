from pathlib import Path
import pymupdf
import json
import hashlib
import re
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/documents")

STORAGE_DIR = Path("storage")
CHUNKS_DIR = STORAGE_DIR / "chunks"
METADATA_DIR = STORAGE_DIR / "metadata"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# Maximum target size for a chunk
# Measured in words.
MAX_CHUNK_WORDS = 180

# Minimum size before we consider merging a small section
MIN_CHUNK_WORDS = 35


# ============================================================
# DOMAIN CONFIGURATION
# ============================================================

DOMAIN_CODES = {
    "sellers": "SEL",
    "refunds": "REF",
    "delivery": "DEL",
    "payments": "PAY",
    "returns": "RET",
    "customer_support": "SUP",
}


# ============================================================
# DOCUMENT TYPE DETECTION
# ============================================================

def get_document_type(file_name):
    """
    Determine the document type from the filename.
    """

    name = file_name.lower()

    if "policy" in name:
        return "policy"

    if "procedure" in name:
        return "procedure"

    if "guideline" in name:
        return "guideline"

    if "faq" in name:
        return "faq"

    if "timeline" in name:
        return "timeline"

    if "onboarding" in name:
        return "guide"

    if "methods" in name:
        return "reference"

    return "document"


# ============================================================
# FILE HASH
# ============================================================

def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash of the source PDF.

    This allows us to detect whether a document
    has changed before rebuilding the index later.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(8192)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean extracted PDF text.

    Removes:
    - standalone G artifacts
    - excessive whitespace
    - unnecessary line breaks

    Preserves:
    - meaningful text
    - sentence structure
    """

    if not text:
        return ""

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):

        # Remove whitespace around line
        line = line.strip()

        # Remove standalone PDF artifact
        if line.upper() == "G":
            continue

        # Remove repeated spaces/tabs
        line = re.sub(r"[ \t]+", " ", line)

        if line:
            lines.append(line)

    # Join lines while preserving paragraph/section boundaries
    cleaned = "\n".join(lines)

    # Remove multiple blank lines
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)

    return cleaned.strip()


# ============================================================
# EXTRACT PDF PAGES
# ============================================================

def extract_pdf_pages(pdf_path):
    """
    Extract text page-by-page using PyMuPDF.
    """

    pages = []

    document = pymupdf.open(pdf_path)

    try:

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            text = clean_text(text)

            if text:

                pages.append({
                    "page": page_number,
                    "text": text
                })

    finally:

        document.close()

    return pages


# ============================================================
# HEADING DETECTION
# ============================================================

def looks_like_heading(line):
    """
    Heuristic heading detection.

    This is intentionally conservative because
    we don't want ordinary sentences to become headings.
    """

    line = line.strip()

    if not line:
        return False

    # Very long lines are unlikely to be headings
    if len(line) > 100:
        return False

    # Questions are generally content, not headings
    if line.endswith("?"):
        return False

    # Numbered steps are content
    if re.match(r"^(Step\s+\d+|\d+[\.\)])", line, re.IGNORECASE):
        return False

    # Common heading patterns
    heading_keywords = [
        "Support Channels",
        "Complaint Handling Process",
        "Escalation Guidelines",
        "Service Standards",
        "Delivery Process",
        "Delivery Locations",
        "Cash on Delivery",
        "Failed Deliveries",
        "Standard Timelines",
        "Express Delivery",
        "Factors That May Delay Delivery",
        "Tracking a Delivery",
        "Common Questions",
        "Failed Payment Troubleshooting",
        "Promo Codes & Vouchers",
        "Available Payment Options",
        "Installment Plans",
        "Payment Security",
        "When a Refund Is Issued",
        "Refund Methods",
        "Refund Process Steps",
        "Processing Time by Method",
        "What Affects Refund Speed",
        "Checking Refund Status",
        "Eligibility Checklist",
        "Category-Specific Rules",
        "When a Return Is Rejected",
        "General Return Window",
        "What Can Be Returned",
        "What Cannot Be Returned",
        "How to Start a Return",
        "Registration Requirements",
        "Onboarding Steps",
        "Training & Support",
        "Listing Requirements",
        "Order Fulfillment Obligations",
        "Returns & Refund Obligations for Sellers",
        "Performance Standards",
    ]

    if line in heading_keywords:
        return True

    # Main document title:
    # usually short and title-like
    words = line.split()

    if len(words) <= 7:

        # Avoid treating obvious content sentences as headings
        content_starters = (
            "A ",
            "An ",
            "The ",
            "If ",
            "Customers ",
            "Sellers ",
            "Items ",
            "Most ",
            "Once ",
            "For ",
            "All ",
            "Only ",
            "When ",
        )

        if not line.startswith(content_starters):
            return True

    return False


# ============================================================
# SECTION PARSING
# ============================================================

def parse_sections(page_text):
    """
    Split a page into logical sections based on headings.

    Returns:

    [
        {
            "heading": "...",
            "text": "..."
        }
    ]
    """

    lines = page_text.split("\n")

    sections = []

    current_heading = None
    current_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if looks_like_heading(line):

            # Save previous section
            if current_heading is not None and current_lines:

                sections.append({
                    "heading": current_heading,
                    "text": " ".join(current_lines).strip()
                })

            # Start new section
            current_heading = line
            current_lines = []

        else:

            current_lines.append(line)

    # Save final section
    if current_heading is not None and current_lines:

        sections.append({
            "heading": current_heading,
            "text": " ".join(current_lines).strip()
        })

    # If no headings were detected
    if not sections:

        return [
            {
                "heading": "General",
                "text": page_text.replace("\n", " ").strip()
            }
        ]

    return sections


# ============================================================
# WORD-BASED SPLITTING
# ============================================================

def split_large_text(text, max_words=MAX_CHUNK_WORDS):
    """
    Split a large section into smaller word-based chunks.
    """

    words = text.split()

    if len(words) <= max_words:
        return [text.strip()]

    chunks = []

    start = 0

    while start < len(words):

        end = start + max_words

        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        start = end

    return chunks


# ============================================================
# CHUNK CREATION
# ============================================================

def create_section_chunks(section, page_number):
    """
    Convert a logical section into one or more chunks.
    """

    heading = section["heading"]
    text = section["text"]

    if not text:
        return []

    text_chunks = split_large_text(text)

    chunks = []

    for index, chunk_text in enumerate(text_chunks, start=1):

        # Add heading to every chunk.
        # This improves semantic retrieval.
        final_text = f"{heading}: {chunk_text}"

        chunks.append({
            "heading": heading,
            "text": final_text,
            "page": page_number,
            "section_chunk_index": index
        })

    return chunks


# ============================================================
# MERGE VERY SMALL CHUNKS
# ============================================================

def merge_small_chunks(chunks):
    """
    Merge very small adjacent sections.

    This prevents tiny chunks such as a single sentence
    from becoming independent vector entries.
    """

    if not chunks:
        return []

    merged = []

    for chunk in chunks:

        word_count = len(chunk["text"].split())

        if (
            word_count < MIN_CHUNK_WORDS
            and merged
            and merged[-1]["page"] == chunk["page"]
        ):

            merged[-1]["text"] += " " + chunk["text"]

        else:

            merged.append(chunk.copy())

    return merged


# ============================================================
# MAIN INGESTION
# ============================================================

def main():

    print("=" * 70)
    print("DARAZ KNOWLEDGE BASE INGESTION")
    print("=" * 70)

    pdf_files = sorted(DATA_DIR.rglob("*.pdf"))

    print(f"\nPDF files found: {len(pdf_files)}")

    if not pdf_files:

        print("\nERROR: No PDF files found.")

        print(
            "\nExpected folder structure:"
            "\n  data/documents/<domain>/*.pdf"
        )

        return

    all_chunks = []
    documents = []

    # Keep numbering stable within each domain
    domain_counters = {}

    # Global chunk counter
    total_chunk_counter = 0

    # ========================================================
    # PROCESS EACH PDF
    # ========================================================

    for pdf_path in pdf_files:

        print("\n" + "-" * 70)

        print(f"Processing: {pdf_path.name}")

        # --------------------------------------------
        # DOMAIN
        # --------------------------------------------

        domain = pdf_path.parent.name.lower()

        domain_code = DOMAIN_CODES.get(domain, "DOC")

        domain_counters[domain] = (
            domain_counters.get(domain, 0) + 1
        )

        document_number = domain_counters[domain]

        document_id = (
            f"{domain_code}-{document_number:03d}"
        )

        # --------------------------------------------
        # DOCUMENT TYPE
        # --------------------------------------------

        document_type = get_document_type(
            pdf_path.name
        )

        # --------------------------------------------
        # FILE HASH
        # --------------------------------------------

        file_hash = calculate_file_hash(
            pdf_path
        )

        # --------------------------------------------
        # EXTRACT PAGES
        # --------------------------------------------

        pages = extract_pdf_pages(pdf_path)

        print(f"Pages extracted: {len(pages)}")

        document_chunks = []

        # --------------------------------------------
        # PROCESS EACH PAGE
        # --------------------------------------------

        for page_data in pages:

            page_number = page_data["page"]

            page_text = page_data["text"]

            # ----------------------------------------
            # Parse logical sections
            # ----------------------------------------

            sections = parse_sections(page_text)

            # ----------------------------------------
            # Create section chunks
            # ----------------------------------------

            page_chunks = []

            for section in sections:

                section_chunks = create_section_chunks(
                    section,
                    page_number
                )

                page_chunks.extend(
                    section_chunks
                )

            # ----------------------------------------
            # Merge tiny chunks
            # ----------------------------------------

            page_chunks = merge_small_chunks(
                page_chunks
            )

            # ----------------------------------------
            # Convert to final chunk records
            # ----------------------------------------

            for page_chunk in page_chunks:

                total_chunk_counter += 1

                chunk_index = (
                    len(document_chunks) + 1
                )

                chunk_id = (
                    f"{document_id}"
                    f"-P{page_number:02d}"
                    f"-C{chunk_index:03d}"
                )

                chunk_text = page_chunk["text"]

                chunk_record = {

                    # -----------------------------
                    # IDENTIFIERS
                    # -----------------------------

                    "chunk_id": chunk_id,

                    "document_id": document_id,

                    # -----------------------------
                    # SOURCE
                    # -----------------------------

                    "file_name": pdf_path.name,

                    "file_path": str(pdf_path),

                    "domain": domain,

                    "document_type": document_type,

                    # -----------------------------
                    # LOCATION
                    # -----------------------------

                    "page": page_number,

                    "section": page_chunk["heading"],

                    "chunk_index": chunk_index,

                    # -----------------------------
                    # CONTENT
                    # -----------------------------

                    "text": chunk_text,

                    "char_count": len(chunk_text),

                    "word_count": len(
                        chunk_text.split()
                    ),

                    # -----------------------------
                    # VERSIONING
                    # -----------------------------

                    "source_hash": file_hash,

                    "created_at": datetime.now().isoformat()
                }

                document_chunks.append(
                    chunk_record
                )

                all_chunks.append(
                    chunk_record
                )

        # ====================================================
        # DOCUMENT METADATA
        # ====================================================

        document_record = {

            "document_id": document_id,

            "file_name": pdf_path.name,

            "file_path": str(pdf_path),

            "domain": domain,

            "document_type": document_type,

            "page_count": len(pages),

            "chunk_count": len(
                document_chunks
            ),

            "file_hash": file_hash,

            "indexed_at": datetime.now().isoformat()
        }

        documents.append(
            document_record
        )

        print(
            f"Chunks created: "
            f"{len(document_chunks)}"
        )

    # ========================================================
    # SAVE CHUNKS
    # ========================================================

    chunks_file = (
        CHUNKS_DIR / "chunks.json"
    )

    with open(
        chunks_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SAVE DOCUMENT METADATA
    # ========================================================

    metadata_file = (
        METADATA_DIR / "documents.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)

    print("INGESTION COMPLETE")

    print("=" * 70)

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Total chunks: {len(all_chunks)}"
    )

    print(
        f"Average chunks/document: "
        f"{len(all_chunks) / len(documents):.2f}"
    )

    print("\nChunks saved to:")

    print(chunks_file)

    print("\nMetadata saved to:")

    print(metadata_file)

    # ========================================================
    # CHUNK SUMMARY
    # ========================================================

    print("\n" + "-" * 70)

    print("CHUNK SUMMARY")

    print("-" * 70)

    for document in documents:

        print(
            f"{document['document_id']} | "
            f"{document['file_name']} | "
            f"{document['chunk_count']} chunks"
        )

    # ========================================================
    # SAMPLE CHUNKS
    # ========================================================

    print("\n" + "-" * 70)

    print("SAMPLE CHUNKS")

    print("-" * 70)

    for chunk in all_chunks[:5]:

        print(
            f"\n[{chunk['chunk_id']}]"
        )

        print(
            f"Domain: {chunk['domain']}"
        )

        print(
            f"Section: {chunk['section']}"
        )

        print(
            f"Page: {chunk['page']}"
        )

        print(
            f"Words: {chunk['word_count']}"
        )

        print(
            f"Text: {chunk['text'][:400]}"
        )

        print("-" * 50)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()