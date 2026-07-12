"""Text chunking for RAG ingestion.
"""

import re


def chunk_text(text: str, *, max_chars: int = 800, overlap_chars: int = 100) -> list[str]:
    """Split text into overlapping chunks, breaking on paragraph/sentence
    boundaries where possible rather than mid-sentence.

    Overlap preserves context across chunk boundaries so a fact split
    across two paragraphs isn't silently lost from retrieval.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
            continue

        if current:
            chunks.append(current)
            # carry the tail of the previous chunk forward for context overlap
            current = current[-overlap_chars:] + "\n\n" + para
        else:
            current = para

        # A single paragraph longer than max_chars: hard-split it.
        while len(current) > max_chars:
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap_chars :]

    if current:
        chunks.append(current)

    return chunks
