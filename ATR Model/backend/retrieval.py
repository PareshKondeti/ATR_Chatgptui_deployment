from typing import List, Dict, Any, Optional
import json

from .embeddings import embed_texts, cosine_similarity

try:
    from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None  # type: ignore

_reranker: Optional[object] = None


def _get_reranker():
    global _reranker
    if _reranker is None and CrossEncoder is not None:
        try:
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _reranker = None
    return _reranker


def _ensure_vector(vec: Any) -> List[float]:
    # Accept list[float], tuple, numpy array; if string like "[1,2]" parse JSON
    if vec is None:
        return []
    if isinstance(vec, str):
        s = vec.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            return []
        return []
    try:
        return [float(x) for x in list(vec)]
    except Exception:
        return []


def semantic_search(query: str, chunks: List[Dict]) -> List[Dict]:
    if not chunks:
        return []
    texts = [c.get("text", "") for c in chunks]
    # Build embeddings if missing or malformed
    need_embed = False
    for c in chunks:
        emb = _ensure_vector(c.get("embedding"))
        if not emb:
            need_embed = True
            break
    if need_embed:
        vecs = embed_texts(texts, kind="passage")
        for c, v in zip(chunks, vecs):
            c["embedding"] = v
    else:
        # Normalize all embeddings to list[float]
        for c in chunks:
            c["embedding"] = _ensure_vector(c.get("embedding"))

    qv = embed_texts([query], kind="query")[0]
    scored: List[Dict[str, Any]] = []
    for c in chunks:
        score = cosine_similarity(qv, c.get("embedding", []))
        scored.append({**c, "score": float(score)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    # Optional rerank using a lightweight cross-encoder for better relevance
    try:
        reranker = _get_reranker()
        if reranker is not None and scored:
            top_k = min(50, len(scored))
            pairs = [(query, s.get("text", "")) for s in scored[:top_k]]
            ce_scores = reranker.predict(pairs)
            for i, ce in enumerate(ce_scores):
                scored[i]["rerank_score"] = float(ce)
            scored[:top_k] = sorted(scored[:top_k], key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
    except Exception:
        # Fail quietly if reranker unavailable
        pass
    return scored


