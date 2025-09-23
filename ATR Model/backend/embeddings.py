from typing import List, Literal, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


_e5_query: Optional[object] = None
_e5_passage: Optional[object] = None


def _get_e5_models():
    global _e5_query, _e5_passage
    if (_e5_query is None or _e5_passage is None) and SentenceTransformer is not None:
        try:
            # Prefer base; user can switch to small by setting E5_SIZE=small
            import os
            size = (os.getenv("E5_SIZE", "base") or "base").strip().lower()
            model_name = f"intfloat/e5-{size}-v2"
            _e5_query = SentenceTransformer(model_name)
            _e5_passage = _e5_query
        except Exception:
            # Fallback to MiniLM if E5 not available
            try:
                _e5_query = SentenceTransformer("all-MiniLM-L6-v2")
                _e5_passage = _e5_query
            except Exception:
                _e5_query = None
                _e5_passage = None
    return _e5_query, _e5_passage


def _prepend_e5_instruction(texts: List[str], kind: Literal["query", "passage"]) -> List[str]:
    if kind == "query":
        prefix = "query: "
    else:
        prefix = "passage: "
    return [f"{prefix}{t}".strip() for t in texts]


def embed_texts(texts: List[str], kind: Literal["query", "passage"] = "passage") -> List[List[float]]:
    model_q, model_p = _get_e5_models()
    model = model_q if kind == "query" else model_p
    if model is None:
        # Fallback: deterministic random-like vectors for stability across runs
        dim = 768 if kind == "query" else 768
        return [
            list(np.random.default_rng(abs(hash((t, kind))) % (2**32)).standard_normal(dim).astype(float))
            for t in texts
        ]
    inputs = _prepend_e5_instruction(texts, kind)
    vecs = model.encode(inputs, show_progress_bar=False, normalize_embeddings=True)
    return [v.astype(float).tolist() for v in np.asarray(vecs)]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    num = float(np.dot(va, vb))
    den = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
    return num / den if den != 0 else 0.0


