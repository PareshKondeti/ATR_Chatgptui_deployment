from typing import List
import os

try:
    from transformers import pipeline  # type: ignore
except Exception:
    pipeline = None  # type: ignore

_qa_pipeline = None


def _get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None and pipeline is not None:
        # Use extractive QA per README: deepset/roberta-base-squad2
        model_name = os.getenv("QA_MODEL_NAME", "deepset/roberta-base-squad2")
        _qa_pipeline = pipeline("question-answering", model=model_name)
    return _qa_pipeline


def _truncate_tokens(text: str, max_tokens: int = 480) -> str:
    # Approximate tokenization by whitespace; fast and good enough for clipping
    parts = text.split()
    if len(parts) <= max_tokens:
        return text
    return " ".join(parts[:max_tokens])


def answer_with_context(question: str, contexts: List[str]) -> str:
    # Combine top contexts into a single passage for extractive QA
    ctx = "\n".join(contexts[:5])
    ctx = _truncate_tokens(ctx, max_tokens=480)
    qa = _get_qa_pipeline()
    if qa is None or not ctx.strip():
        from textwrap import shorten
        return shorten(ctx, width=400, placeholder="...") or "I don't have enough context."
    try:
        result = qa({"question": question, "context": ctx})
        # result contains 'answer' and 'score'
        ans = (result or {}).get("answer", "")
        return (ans or "").strip()
    except Exception:
        # Safe fallback to top context
        from textwrap import shorten
        return shorten(ctx, width=400, placeholder="...") or "I don't have enough context."


def paraphrase_succinct(answer: str) -> str:
    """No-op for extractive QA; return as-is. Enable only if a t2t model is configured."""
    return (answer or "").strip()


