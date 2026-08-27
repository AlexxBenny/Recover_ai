"""
PDF text extraction using PyMuPDF.

Extracts text from each page of a PDF, cleans it, and returns
structured documents with source metadata for the vector store.

Handles error cases:
- File not found / not a PDF
- Corrupted or unreadable PDFs
- Password-protected PDFs
- Empty PDFs (no extractable text)
- Oversized files
"""

import logging
import re
from pathlib import Path
from typing import Union

import pymupdf

from . import config

logger = logging.getLogger(__name__)


class PDFLoadError(Exception):
    """Raised when a PDF cannot be loaded or processed."""

    pass


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> list[dict]:
    """
    Extract text from a PDF file, page by page.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts, each with:
            - "text": cleaned text content of the page
            - "metadata": {"source": filename, "page_number": int, "total_pages": int}

    Raises:
        FileNotFoundError: If the PDF does not exist
        ValueError: If the file is not a PDF or exceeds size limits
        PDFLoadError: If the PDF is corrupted, password-protected, or unreadable
    """
    pdf_path = Path(pdf_path)

    # --- Validation ---
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Not a PDF file (got '{pdf_path.suffix}'): {pdf_path.name}. "
            "Only .pdf files are supported."
        )

    # Check file size
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > config.MAX_PDF_SIZE_MB:
        raise ValueError(
            f"PDF too large: {file_size_mb:.1f} MB "
            f"(limit: {config.MAX_PDF_SIZE_MB} MB). "
            f"File: {pdf_path.name}"
        )

    if pdf_path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty (0 bytes): {pdf_path.name}")

    # --- Open and extract ---
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        raise PDFLoadError(
            f"Failed to open PDF '{pdf_path.name}': {e}. "
            "The file may be corrupted or in an unsupported format."
        ) from e

    try:
        # Check for password protection
        if doc.is_encrypted:
            doc.close()
            raise PDFLoadError(
                f"PDF is password-protected: {pdf_path.name}. "
                "Please upload an unprotected version."
            )

        if len(doc) == 0:
            doc.close()
            raise PDFLoadError(
                f"PDF has no pages: {pdf_path.name}."
            )

        documents = []
        empty_pages = 0

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text("text")
            except Exception as e:
                logger.warning(
                    f"Failed to extract text from page {page_num + 1} "
                    f"of '{pdf_path.name}': {e}. Skipping page."
                )
                continue

            # Clean the extracted text
            text = _clean_text(text)

            if text.strip():
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": pdf_path.name,
                            "page_number": page_num + 1,
                            "total_pages": len(doc),
                        },
                    }
                )
            else:
                empty_pages += 1

        if empty_pages > 0:
            logger.info(
                f"'{pdf_path.name}': Skipped {empty_pages} empty page(s) "
                f"out of {len(doc)} total."
            )

        if not documents:
            raise PDFLoadError(
                f"No extractable text found in '{pdf_path.name}'. "
                "The PDF may contain only images/scans. "
                "OCR-based PDFs are not currently supported."
            )

        logger.info(
            f"Extracted text from '{pdf_path.name}': "
            f"{len(documents)} page(s) with content out of {len(doc)} total."
        )

        return documents

    finally:
        doc.close()


def _clean_text(text: str) -> str:
    """
    Clean text extracted from PDF.

    Handles common PDF extraction artifacts like excessive whitespace,
    page numbers, and header/footer remnants.
    """
    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize excessive spaces
    text = re.sub(r" {2,}", " ", text)
    # Remove common page number patterns
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    # Strip leading/trailing whitespace
    return text.strip()
