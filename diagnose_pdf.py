"""
diagnose_pdf.py
---------------
Standalone diagnostic: check whether a PDF has extractable text BEFORE
running it through the full ingestion pipeline. Mirrors the exact same
four-tier extraction cascade `ingest.py` uses, so a "PASS" here is a
reliable predictor that `ingest.py` will succeed on the same file.

Usage:
    python diagnose_pdf.py data/Amex.pdf

This does not touch Pinecone or call any paid API -- it's pure local text
extraction (+ optional local OCR), safe to run as many times as you like
while debugging a problem file.
"""

import io
import sys

from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False

try:
    import fitz
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

import pypdf

MIN_CHARS = 50


def _chars(pages) -> int:
    return sum(len(p.page_content.strip()) for p in pages)


def _preview(pages) -> str:
    if not pages:
        return ""
    return pages[0].page_content.strip()[:150].replace("\n", " ")


def diagnose(file_path: str) -> None:
    print(f"Diagnosing: {file_path}\n")

    try:
        encrypted = pypdf.PdfReader(file_path).is_encrypted
        print(f"Encrypted (per pypdf): {encrypted}")
    except Exception as exc:
        print(f"Could not check encryption status: {exc}")
    print()

    winner = None

    # --- Tier 1: pypdf ---
    pypdf_pages = PyPDFLoader(file_path).load()
    pypdf_chars = _chars(pypdf_pages)
    print(f"[1. pypdf]     {len(pypdf_pages)} page(s), {pypdf_chars} chars")
    if pypdf_chars > 0:
        print(f"               preview: {_preview(pypdf_pages)!r}")
    if pypdf_chars >= MIN_CHARS and winner is None:
        winner = ("pypdf", pypdf_pages)

    # --- Tier 2: PyMuPDF ---
    if PYMUPDF_AVAILABLE:
        mupdf_pages = PyMuPDFLoader(file_path).load()
        mupdf_chars = _chars(mupdf_pages)
        print(f"[2. pymupdf]   {len(mupdf_pages)} page(s), {mupdf_chars} chars")
        if mupdf_chars > 0:
            print(f"               preview: {_preview(mupdf_pages)!r}")
        if mupdf_chars >= MIN_CHARS and winner is None:
            winner = ("pymupdf", mupdf_pages)
    else:
        print("[2. pymupdf]   not installed -- skipped")

    # --- Tier 3: pypdfium2 (Chrome/Edge's own engine) ---
    if PDFIUM_AVAILABLE:
        pdf = pdfium.PdfDocument(file_path)
        fium_pages_text = []
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            fium_pages_text.append(tp.get_text_range())
            tp.close()
            page.close()
        pdf.close()
        fium_chars = sum(len(t.strip()) for t in fium_pages_text)
        print(f"[3. pypdfium2] {len(fium_pages_text)} page(s), {fium_chars} chars  "
              f"(this is the SAME engine Chrome/Edge use for in-browser text selection)")
        if fium_chars > 0:
            print(f"               preview: {fium_pages_text[0].strip()[:150]!r}")
        if fium_chars >= MIN_CHARS and winner is None:
            winner = ("pypdfium2", None)
    else:
        print("[3. pypdfium2] not installed -- skipped")

    # --- Tier 4: OCR (only run if nothing above worked, since it's slow) ---
    if winner is None:
        if OCR_AVAILABLE:
            print("[4. OCR]       nothing else worked -- running Tesseract OCR on page 1 "
                  "as a quick check (full ingestion would OCR every page)...")
            try:
                doc = fitz.open(file_path)
                pix = doc[0].get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img)
                doc.close()
                print(f"               page 1 OCR chars: {len(ocr_text.strip())}")
                if ocr_text.strip():
                    print(f"               preview: {ocr_text.strip()[:150]!r}")
                    winner = ("ocr", None)
            except Exception as exc:
                print(f"               OCR failed: {exc}")
                print("               (is the `tesseract-ocr` system binary installed? "
                      "`pip install pytesseract` alone is not enough.)")
        else:
            print("[4. OCR]       not installed -- skipped (pip install pytesseract pillow, "
                  "plus the tesseract-ocr system binary)")

    print()
    if winner:
        print(f"✅ RESULT: Extractable via '{winner[0]}'. `ingest.py` should embed this file correctly.")
    else:
        print(
            "❌ RESULT: No tier could extract usable text -- including OCR, if it ran.\n"
            "   This file most likely has no readable content on the page(s) checked,\n"
            "   or OCR isn't available in this environment. If you can visually see text\n"
            "   on the page, try increasing OCR_DPI in config.py, or confirm the\n"
            "   tesseract-ocr binary is actually installed and on PATH."
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_pdf.py <path-to-pdf>")
        sys.exit(1)
    diagnose(sys.argv[1])
