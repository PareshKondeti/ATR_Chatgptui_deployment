from typing import List, Dict
import re


def split_into_sentences(text: str) -> List[str]:
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def semantic_chunks(text: str, max_tokens: int = 120) -> List[str]:
    # Approximate tokens by words; pack sentences up to budget
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for s in sentences:
        words = s.split()
        wl = len(words)
        if current_len + wl > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(s)
        current_len += wl
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_indexed_chunks(text: str) -> List[Dict[str, str]]:
    chunks = semantic_chunks(text)
    return [{"index": i, "text": c} for i, c in enumerate(chunks)]


