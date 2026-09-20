from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        
        # Split on sentence boundaries while keeping punctuation attached
        raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences:
            return []

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_str = " ".join(group).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        sep = remaining_separators[0]
        next_seps = remaining_separators[1:]

        if sep == "":
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        if sep not in current_text:
            return self._split(current_text, next_seps)

        raw_pieces = current_text.split(sep)
        sub_chunks: list[str] = []
        for piece in raw_pieces:
            if not piece and sep in ("\n\n", "\n"):
                continue
            if len(piece) <= self.chunk_size:
                sub_chunks.append(piece)
            else:
                sub_chunks.extend(self._split(piece, next_seps))

        merged: list[str] = []
        current_chunk = ""
        for piece in sub_chunks:
            if not current_chunk:
                current_chunk = piece
            else:
                candidate = current_chunk + sep + piece
                if len(candidate) <= self.chunk_size:
                    current_chunk = candidate
                else:
                    merged.append(current_chunk)
                    current_chunk = piece
        if current_chunk:
            merged.append(current_chunk)

        return merged


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(y * y for y in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        overlap = min(50, chunk_size // 4)
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        strategies = {
            "fixed_size": fixed_chunker.chunk(text),
            "by_sentences": sentence_chunker.chunk(text),
            "recursive": recursive_chunker.chunk(text),
        }

        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count > 0 else 0.0
            result[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks,
            }
        return result


class HeadingChunker:
    """
    Split markdown text into sections based on headings (#, ##, ###).
    If a section exceeds max_chunk_size, recursively split it and prepend
    the section heading to each sub-chunk so context is not lost.
    """

    def __init__(self, max_chunk_size: int = 500) -> None:
        self.max_chunk_size = max_chunk_size
        self.recursive_fallback = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sections = re.split(r"(?m)^(#{1,3}\s+.*)$", text)
        chunks: list[str] = []
        current_header = ""

        for part in sections:
            part_str = part.strip()
            if not part_str:
                continue

            if re.match(r"^#{1,3}\s+", part_str):
                current_header = part_str
            else:
                full_section = f"{current_header}\n\n{part_str}".strip() if current_header else part_str
                if len(full_section) <= self.max_chunk_size:
                    chunks.append(full_section)
                else:
                    sub_pieces = self.recursive_fallback.chunk(part_str)
                    for sub in sub_pieces:
                        sub_with_header = f"{current_header}\n\n{sub}".strip() if current_header else sub
                        chunks.append(sub_with_header)

        return chunks


