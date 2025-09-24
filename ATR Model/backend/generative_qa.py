from typing import List
import os

try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # type: ignore

_generator = None


def _get_generator():
    global _generator
    if _generator is None and pipeline is not None:
        # Revert to original model configuration
        _generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return _generator


def _truncate_tokens(text: str, max_tokens: int = 480) -> str:
    # Approximate tokenization by whitespace; fast and good enough for clipping
    parts = text.split()
    if len(parts) <= max_tokens:
        return text
    return " ".join(parts[:max_tokens])


def answer_with_context(question: str, contexts: List[str]) -> str:
    ctx = "\n".join(contexts[:5])
    ctx = _truncate_tokens(ctx, max_tokens=480)
    prompt = (
        "Use the context to answer the question in a few sentences with specifics.\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    gen = _get_generator()
    if gen is None:
        # Fallback: join top context snippet
        from textwrap import shorten
        return shorten(ctx, width=400, placeholder="...") or "I don't have enough context."
    out = gen(prompt, max_length=192, do_sample=False, num_beams=4)
    if isinstance(out, list) and out:
        return out[0].get("generated_text", "").strip()
    return ""


def paraphrase_succinct(answer: str) -> str:
    """Paraphrase an answer into 1-2 concise sentences using FLAN-T5 if available."""
    answer = (answer or "").strip()
    if not answer:
        return ""
    # Allow runtime control via env PARAPHRASE_QA
    flag = (os.getenv("PARAPHRASE_QA", "0") or "0").strip().lower() in ("1", "true", "yes", "on")
    if not flag:
        return answer
    gen = _get_generator()
    if gen is None:
        return answer
    prompt = (
        "Rewrite the following answer into 1-2 concise, direct sentences, preserving factual details.\n\n"
        f"Answer: {answer}\n\nRewritten:"
    )
    out = gen(prompt, max_length=128, do_sample=False, num_beams=4)
    if isinstance(out, list) and out:
        rewritten = out[0].get("generated_text", "").strip()
        return rewritten or answer
    return answer


