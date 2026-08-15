"""
ingest.py
---------
Dynamic ingestion pipeline for CRAG-Ops.

Design goals (per project spec):
  1. Zero local vector storage -- everything lands in Pinecone.
  2. Idempotent -- running this script (or re-uploading the same PDF via
     the Streamlit UI) N times only ever embeds the document ONCE. The
     check is performed against Pinecone itself (not a local cache file),
     so it survives app restarts and redeploys.
  3. Reusable -- the same `ingest_file()` function backs both the CLI
     bulk-loader (`python ingest.py`) and the Streamlit sidebar uploader.

Deduplication strategy
-----------------------
We hash the raw bytes of each PDF (SHA-256) to get a stable `file_hash`.
Every chunk of that document is upserted with a DETERMINISTIC vector id:

    f"{file_hash}-{chunk_index}"

Before embedding anything, we check whether the vector with id
`f"{file_hash}-0"` (the document's first chunk) already exists in
Pinecone via a cheap `index.fetch()` call. If it does, we know this exact
file has already been fully processed and we skip it entirely -- no
embedding API calls, no upsert. This is far cheaper and more precise than
re-querying by similarity, and it means two differently-named files with
identical content are correctly recognized as duplicates, while a file
that changes content changes hash and is (re-)ingested as "new".
"""

import glob
import hashlib
import io
import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    _PYMUPDF_AVAILABLE = True
except ImportError:
    _PYMUPDF_AVAILABLE = False

try:
    import pypdfium2 as _pdfium
    _PDFIUM_AVAILABLE = True
except ImportError:
    _PDFIUM_AVAILABLE = False

try:
    import fitz as _fitz_for_ocr  # PyMuPDF, reused here to rasterize pages for OCR
    import pytesseract
    from PIL import Image
    _OCR_LIBS_AVAILABLE = True
except ImportError:
    _OCR_LIBS_AVAILABLE = False

import pypdf as _pypdf_raw  # used directly for the encryption diagnostic, not just via the loader

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    ENABLE_OCR_FALLBACK,
    EMBEDDING_DIMENSION,
    MIN_EXTRACTABLE_CHARS,
    OCR_DPI,
    get_or_create_index,
    get_vectorstore,
)


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------
def compute_file_hash(file_path: str) -> str:
    """Return a short, stable SHA-256 hash of a file's raw bytes.

    Hashing content (not filename) means re-uploading a byte-identical PDF
    under a new filename is still correctly recognized as a duplicate.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    # 16 hex chars (64 bits) is plenty of collision resistance for this use
    # case and keeps vector IDs short.
    return sha256.hexdigest()[:16]


# --------------------------------------------------------------------------
# Cloud-side dedup check
# --------------------------------------------------------------------------
def document_already_ingested(index, file_hash: str) -> bool:
    """Check Pinecone (not local disk/memory) for whether this document's
    first chunk already exists. This is the source of truth that survives
    process/container restarts.
    """
    try:
        result = index.fetch(ids=[f"{file_hash}-0"])
        # pinecone-python-client returns a FetchResponse with a `.vectors`
        # dict keyed by id; an empty dict means no match was found.
        vectors = getattr(result, "vectors", None) or {}
        return len(vectors) > 0
    except Exception:
        # If the fetch itself fails (e.g. transient network issue), fail
        # open toward re-ingestion rather than silently losing a document.
        return False


# --------------------------------------------------------------------------
# Load + chunk
# --------------------------------------------------------------------------
def _total_chars(pages) -> int:
    return sum(len(p.page_content.strip()) for p in pages)


def _is_encrypted(file_path: str) -> bool:
    """Diagnostic only -- used to make failure messages more actionable,
    not to change extraction behavior (pypdf auto-decrypts empty-password
    PDFs transparently in modern versions)."""
    try:
        return _pypdf_raw.PdfReader(file_path).is_encrypted
    except Exception:
        return False


def _extract_via_pypdf(file_path: str) -> list:
    return PyPDFLoader(file_path).load()


def _extract_via_pymupdf(file_path: str) -> list:
    return PyMuPDFLoader(file_path).load()


def _extract_via_pdfium(file_path: str) -> list:
    """
    Extract via pypdfium2 -- Google's PDFium engine, the SAME engine Chrome
    and Edge use for their built-in PDF viewer (including the "select and
    copy text" feature). This tier specifically catches the common case
    where a PDF's font has a broken/non-standard Unicode mapping table:
    pypdf and PyMuPDF's extractors rely on that table and come back empty,
    while PDFium reconstructs text from glyph shapes and succeeds -- which
    is exactly what you're seeing when you can select/copy text in your
    browser from a PDF that pypdf reports as "textless".
    """
    pdf = _pdfium.PdfDocument(file_path)
    pages = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            textpage.close()
            page.close()
            pages.append(Document(page_content=text, metadata={"page": i}))
    finally:
        pdf.close()
    return pages


def _extract_via_ocr(file_path: str, dpi: int = OCR_DPI, progress_cb=None) -> list:
    """
    Genuine last resort: rasterize each page to an image and run Tesseract
    OCR on it. Only reached if pypdf, PyMuPDF, AND pypdfium2 all fail to
    find real embedded text -- i.e. the PDF is very likely a true scanned
    image (or uses vector-drawn "Type 3" glyphs with no encoded text at
    all). This is slow (roughly 0.5-2s per page) but correct.

    Requires the `tesseract-ocr` system binary to be installed separately
    from the Python packages (see packages.txt for Streamlit Cloud, or
    your OS package manager for local dev). If it's missing, this raises
    and the caller treats it the same as "OCR unavailable".
    """
    doc = _fitz_for_ocr.open(file_path)
    pages = []
    try:
        n = len(doc)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            pages.append(Document(page_content=text, metadata={"page": i}))
            if progress_cb:
                progress_cb(i + 1, n)
    finally:
        doc.close()
    return pages


def _load_pages_with_fallback(file_path: str, progress_cb=None) -> tuple:
    """
    Try progressively more powerful (and more expensive) text-extraction
    strategies until one produces real text:

        1. pypdf       -- fast, handles the vast majority of PDFs
        2. PyMuPDF     -- different parser, recovers some pypdf misses
        3. pypdfium2   -- Chrome/Edge's own engine; recovers PDFs with
                          broken font Unicode tables that are nonetheless
                          visibly selectable in a browser
        4. OCR         -- genuine last resort for true image-only scans
                          or vector-only "Type 3" glyph fonts with zero
                          encoded text

    Returns (pages, extractor_name, attempts) where `attempts` is a list of
    (tier_name, result) pairs for diagnostics, so a total failure can be
    explained precisely rather than just reported as "0 characters".
    """
    tiers = [("pypdf", _extract_via_pypdf, True)]
    if _PYMUPDF_AVAILABLE:
        tiers.append(("pymupdf", _extract_via_pymupdf, True))
    if _PDFIUM_AVAILABLE:
        tiers.append(("pypdfium2", _extract_via_pdfium, True))

    attempts = []
    for name, fn, _ in tiers:
        try:
            pages = fn(file_path)
            chars = _total_chars(pages)
            attempts.append((name, f"{chars} chars"))
            if chars >= MIN_EXTRACTABLE_CHARS:
                return pages, name, attempts
        except Exception as exc:
            attempts.append((name, f"error: {exc}"))

    if ENABLE_OCR_FALLBACK and _OCR_LIBS_AVAILABLE:
        try:
            pages = _extract_via_ocr(file_path, progress_cb=progress_cb)
            chars = _total_chars(pages)
            attempts.append(("ocr", f"{chars} chars"))
            if chars >= MIN_EXTRACTABLE_CHARS:
                return pages, "ocr (last-resort fallback)", attempts
        except Exception as exc:
            attempts.append(("ocr", f"error: {exc} (is tesseract-ocr installed on this machine?)"))
    elif ENABLE_OCR_FALLBACK and not _OCR_LIBS_AVAILABLE:
        attempts.append(("ocr", "skipped -- pytesseract/Pillow not installed"))

    # Every tier failed. Return the last attempted pages (likely empty) so
    # the caller can report a precise, honest failure.
    return [], "none", attempts


def load_and_chunk_pdf(file_path: str, file_hash: str, progress_cb=None) -> tuple:
    """Load a PDF and split it into overlapping chunks, tagging every
    chunk with metadata needed for dedup checks, source attribution in
    the final answer, and Pinecone filtering.

    Returns (chunks, extraction_warning). `extraction_warning` is an empty
    string on success, or a detailed, actionable explanation (including
    which extraction tiers were tried and why each failed) if no usable
    text could be found by any tier.
    """
    filename = os.path.basename(file_path)
    pages, extractor_used, attempts = _load_pages_with_fallback(file_path, progress_cb=progress_cb)

    if _total_chars(pages) < MIN_EXTRACTABLE_CHARS:
        attempts_summary = "; ".join(f"{name}: {result}" for name, result in attempts)
        encrypted_note = (
            " This file IS flagged as encrypted, which can cause extraction "
            "tools to fail even when a viewer opens it with no password prompt."
            if _is_encrypted(file_path)
            else ""
        )
        warning = (
            f"No usable text could be extracted from '{filename}' after trying "
            f"every available extraction method ({attempts_summary}).{encrypted_note} "
            "If you can select/copy text from this PDF in a normal viewer but "
            "every tier above still failed, the file most likely uses vector-drawn "
            "'Type 3' glyph fonts with no real encoded text -- OCR is the only way "
            "to read those, and it was tried above if enabled. If OCR also failed, "
            "confirm the `tesseract-ocr` system binary is installed (not just the "
            "Python `pytesseract` package)."
        )
        return [], warning

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "source": filename,
                "file_hash": file_hash,
                "chunk_index": i,
                "page": chunk.metadata.get("page", -1),
                "extractor": extractor_used,
            }
        )
    return chunks, ""


# --------------------------------------------------------------------------
# Embed + upsert
# --------------------------------------------------------------------------
def embed_and_upsert(index, chunks, file_hash: str) -> int:
    """Embed all chunks and upsert them to Pinecone with deterministic IDs."""
    if not chunks:
        return 0

    vectorstore = get_vectorstore(index)
    ids = [f"{file_hash}-{i}" for i in range(len(chunks))]

    # LangChain's PineconeVectorStore batches embedding + upsert calls
    # internally, which keeps this robust for large 10-K filings that can
    # split into hundreds of chunks.
    vectorstore.add_documents(documents=chunks, ids=ids)
    return len(chunks)


# --------------------------------------------------------------------------
# Public entry point -- used by both the CLI and the Streamlit uploader
# --------------------------------------------------------------------------
def ingest_file(file_path: str, index=None, progress_cb=None) -> dict:
    """
    Ingest a single PDF, skipping it entirely if it's already in Pinecone.

    `progress_cb(current_page, total_pages)`, if given, is called during
    the OCR fallback tier only (the other tiers are fast enough not to
    need progress reporting). Safe to leave as None.

    Returns a small status dict so callers (CLI or Streamlit) can report
    back to the user without needing to know internal implementation
    details.
    """
    if index is None:
        index = get_or_create_index()

    filename = os.path.basename(file_path)
    file_hash = compute_file_hash(file_path)

    if document_already_ingested(index, file_hash):
        return {
            "status": "skipped",
            "filename": filename,
            "file_hash": file_hash,
            "chunks": 0,
            "reason": "Document already exists in Pinecone -- not re-embedded.",
        }

    chunks, extraction_warning = load_and_chunk_pdf(file_path, file_hash, progress_cb=progress_cb)

    if extraction_warning:
        # Explicitly NOT "ingested" -- there is nothing in Pinecone for this
        # file, and re-running ingest.py will correctly retry it (rather
        # than the old behavior of reporting false success).
        return {
            "status": "failed",
            "filename": filename,
            "file_hash": file_hash,
            "chunks": 0,
            "reason": extraction_warning,
        }

    n_chunks = embed_and_upsert(index, chunks, file_hash)

    return {
        "status": "ingested",
        "filename": filename,
        "file_hash": file_hash,
        "chunks": n_chunks,
        "reason": f"Embedded {n_chunks} new chunks.",
    }


def ingest_directory(data_dir: str = DATA_DIR) -> list:
    """Bulk-ingest every PDF found in `data_dir`. Safe to run repeatedly --
    already-ingested files are detected and skipped (see module docstring).
    """
    index = get_or_create_index()
    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))

    if not pdf_paths:
        print(f"No PDF files found in '{data_dir}/'. Add some 10-K filings and re-run.")
        return []

    results = []
    for path in pdf_paths:
        result = ingest_file(path, index=index)
        results.append(result)
        icon = {
            "skipped": "⏭️  SKIPPED ",
            "ingested": "✅ INGESTED",
            "failed": "❌ FAILED  ",
        }[result["status"]]
        print(f"{icon} | {result['filename']:<40} | {result['reason']}")

    return results


def list_ingested_files(index=None, limit: int = 500) -> list:
    """
    Best-effort listing of distinct source filenames currently stored in
    Pinecone.

    Pinecone has no native "list distinct metadata values" operation, so
    we exploit our own ID scheme: every document's first chunk always has
    `chunk_index == 0`. Filtering on that metadata field and querying with
    a neutral (all-zero) vector returns exactly one hit per ingested
    document, which is exactly the list we want.
    """
    if index is None:
        index = get_or_create_index()

    try:
        zero_vector = [0.0] * EMBEDDING_DIMENSION
        response = index.query(
            vector=zero_vector,
            top_k=limit,
            include_metadata=True,
            filter={"chunk_index": {"$eq": 0}},
        )
        matches = getattr(response, "matches", []) or []
        return sorted(
            {m.metadata.get("source", "unknown") for m in matches if m.metadata}
        )
    except Exception:
        return []


def reset_knowledge_base(index=None) -> dict:
    """
    Wipe every vector from the Pinecone index -- use this when you want to
    completely change the knowledge base (remove all previously ingested
    PDFs before loading a new set).

    This clears the DEFAULT namespace of the existing index; it does not
    delete the index itself, so no dimension/region reconfiguration is
    needed afterward -- just re-run ingestion with your new PDFs.

    This is destructive and cannot be undone. Callers (CLI, Streamlit UI)
    are responsible for confirming with the user before calling this.
    """
    if index is None:
        index = get_or_create_index()

    try:
        index.delete(delete_all=True)
        return {"status": "reset", "reason": "All vectors deleted from the index."}
    except Exception as exc:
        # Pinecone serverless raises if you call delete_all on an index
        # that's already empty -- treat that as a harmless no-op rather
        # than a real failure.
        if "Namespace not found" in str(exc) or "404" in str(exc):
            return {"status": "reset", "reason": "Index was already empty."}
        return {"status": "error", "reason": f"Reset failed: {exc}"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CRAG-Ops ingestion pipeline")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe ALL vectors from the Pinecone index before ingesting (use this "
        "to completely swap out the knowledge base for a new set of PDFs).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt for --reset (for scripting/CI).",
    )
    args = parser.parse_args()

    if args.reset:
        if not args.yes:
            confirm = input(
                "⚠️  This will permanently delete ALL vectors in your Pinecone index. "
                "Type 'RESET' to confirm: "
            )
            if confirm.strip() != "RESET":
                print("Aborted -- no changes made.")
                raise SystemExit(0)
        print("Resetting knowledge base...")
        result = reset_knowledge_base()
        print(f"{'✅' if result['status'] == 'reset' else '❌'} {result['reason']}")
        if result["status"] != "reset":
            raise SystemExit(1)

    print(f"Starting bulk ingestion from '{DATA_DIR}/' ...")
    ingest_directory()
    print("Done.")