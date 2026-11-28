import re
from pathlib import Path
from typing import Any

import pymupdf

from app.core.config import settings


def _split_long_text(text: str, max_size: int) -> list[str]:
    """Split oversized sentences at word boundaries."""

    words = text.split()
    segments: list[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()

        if len(candidate) > max_size and current:
            segments.append(current)
            current = word
        else:
            current = candidate

    if current:
        segments.append(current)

    return segments


def _extract_paragraphs(page: pymupdf.Page) -> list[str]:
    """
    Extract ordered text blocks from a PDF page.

    Empty blocks are ignored and paragraph boundaries are preserved.
    """

    blocks = page.get_text("blocks", sort=True)

    paragraphs: list[str] = []

    for block in blocks:
        # PyMuPDF block tuple:
        # x0, y0, x1, y1, text, block_no, block_type
        if len(block) < 7 or block[6] != 0:
            continue

        block_text = block[4].strip()

        if not block_text:
            continue

        # Separate paragraphs within the same text block.
        parts = re.split(r"\n\s*\n", block_text)

        paragraphs.extend(part.strip() for part in parts if part.strip())

    return paragraphs


def _split_paragraph(paragraph: str, max_size: int) -> list[str]:
    """Split a paragraph while preserving sentence boundaries."""

    sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())

    segments: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > max_size:
            segments.extend(_split_long_text(sentence, max_size))
        else:
            segments.append(sentence)

    return segments


def _build_chunks(
    paragraphs: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Group paragraphs and sentences into overlapping chunks.

    Overlap is preserved using complete sentence segments whenever possible.
    """

    units: list[str] = []

    for paragraph in paragraphs:
        units.extend(_split_paragraph(paragraph, chunk_size))

    chunks: list[str] = []
    current_units: list[str] = []
    current_length = 0

    for unit in units:
        separator_length = 2 if current_units else 0

        if current_units and current_length + separator_length + len(unit) > chunk_size:
            chunks.append("\n\n".join(current_units))

            # Keep the last complete sentences as overlap.
            overlap_units: list[str] = []
            overlap_length = 0

            for previous_unit in reversed(current_units):
                additional_length = len(previous_unit) + (2 if overlap_units else 0)

                if overlap_length + additional_length > chunk_overlap:
                    break

                overlap_units.insert(0, previous_unit)
                overlap_length += additional_length

            current_units = overlap_units
            current_length = overlap_length

            # Ensure the next segment always fits.
            while current_units and current_length + 2 + len(unit) > chunk_size:
                removed = current_units.pop(0)
                current_length -= len(removed)

                if current_units:
                    current_length -= 2

        current_units.append(unit)
        current_length += len(unit)

        if len(current_units) > 1:
            current_length += 2

    if current_units:
        final_chunk = "\n\n".join(current_units)

        # Preserve a short final chunk rather than discarding its content.
        if (
            len(final_chunk) < settings.min_chunk_chars
            and chunks
            and len(chunks[-1]) + 2 + len(final_chunk) <= chunk_size
        ):
            chunks[-1] += "\n\n" + final_chunk
        else:
            chunks.append(final_chunk)

    return chunks


def extract_chunks(pdf_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract structured text chunks from a PDF document.

    Chunk boundaries prioritize paragraphs and sentences.
    Page numbers and source filenames are preserved.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported file format: {pdf_path}")

    if settings.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    document_chunks: list[dict[str, Any]] = []

    with pymupdf.open(pdf_path) as document:
        if document.needs_pass:
            raise ValueError(f"Password-protected PDF: {pdf_path}")

        for page_number, page in enumerate(document, start=1):
            paragraphs = _extract_paragraphs(page)

            if not paragraphs:
                continue

            chunks = _build_chunks(
                paragraphs=paragraphs,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            for chunk_text in chunks:
                if not chunk_text.strip():
                    continue

                document_chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number,
                        "source": pdf_path.name,
                    }
                )

    return document_chunks
