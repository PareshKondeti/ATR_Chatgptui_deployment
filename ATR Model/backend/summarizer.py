from typing import Optional, List, Tuple
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
except Exception:
    pipeline = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForSeq2SeqLM = None  # type: ignore

_summarizer = None
_summarizer_kind: Optional[str] = None  # 'flan' or 'bart' or 'led'
_led_summarizer = None
_logger = logging.getLogger("atr.summarization")


def _get_summarizer() -> Tuple[Optional[object], Optional[str]]:
    global _summarizer, _summarizer_kind
    if _summarizer is not None:
        return _summarizer, _summarizer_kind
    if pipeline is None:
        _logger.warning("transformers.pipeline not available; summarization disabled")
        return None, None
    # Force BART as default, ignoring env; download/cache locally under models/bart-large-cnn
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        cache_dir = os.path.join(project_root, "models", "bart-large-cnn")
        os.makedirs(cache_dir, exist_ok=True)
        _logger.info("Summarizer selection: forced=bart cache_dir=%s", cache_dir)
        if AutoTokenizer is not None and AutoModelForSeq2SeqLM is not None:
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=cache_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", cache_dir=cache_dir)
            _summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        else:
            # Fallback: rely on pipeline with cache via model_kwargs if core classes unavailable
            _summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
            )
        _summarizer_kind = 'bart'
        _logger.info("Initialized summarizer model kind=bart (facebook/bart-large-cnn)")
        return _summarizer, _summarizer_kind
    except Exception:
        _logger.exception("Failed to initialize facebook/bart-large-cnn (forced)")
        return None, None


def _get_led_summarizer() -> Optional[object]:
    global _led_summarizer
    if _led_summarizer is not None:
        return _led_summarizer
    if pipeline is None:
        return None
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        cache_dir = os.path.join(project_root, "models", "led-16384")
        os.makedirs(cache_dir, exist_ok=True)
        if AutoTokenizer is not None and AutoModelForSeq2SeqLM is not None:
            tok = AutoTokenizer.from_pretrained("allenai/longformer-encoder-decoder-16384", cache_dir=cache_dir)
            mod = AutoModelForSeq2SeqLM.from_pretrained("allenai/longformer-encoder-decoder-16384", cache_dir=cache_dir)
            _led_summarizer = pipeline("summarization", model=mod, tokenizer=tok)
        else:
            _led_summarizer = pipeline("summarization", model="allenai/longformer-encoder-decoder-16384")
        _logger.info("Initialized LED summarizer (allenai/longformer-encoder-decoder-16384)")
        return _led_summarizer
    except Exception:
        _logger.exception("Failed to initialize LED summarizer")
        _led_summarizer = None
        return None


def _chunk_text(text: str, chunk_size: int = 2200) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        _logger.debug("Chunking: single chunk len_chars=%d (chunk_size=%d)", len(text), chunk_size)
        return [text]
    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # try to break on sentence end
        slice_ = text[start:end]
        pivot = max(slice_.rfind('. '), slice_.rfind('\n'))
        if pivot != -1 and start + pivot + 1 < end:
            end = start + pivot + 1
            slice_ = text[start:end]
        parts.append(slice_.strip())
        start = end
    chunks = [p for p in parts if p]
    _logger.debug("Chunking: produced %d chunks (chunk_size=%d)", len(chunks), chunk_size)
    return chunks


def _looks_extractive(candidate: str, source: str) -> bool:
    """Heuristic: if candidate is largely copied from source (substring or n-gram overlap)."""
    try:
        c = (candidate or "").strip().lower()
        s = (source or "").strip().lower()
        if not c or not s:
            return False
        if c in s:
            return True
        def ngrams(text: str, n: int = 6) -> set:
            words = text.split()
            return {" ".join(words[i:i+n]) for i in range(max(0, len(words) - n + 1))}
        cg = ngrams(c, 6)
        sg = ngrams(s, 6)
        if not cg or not sg:
            return False
        overlap = len(cg & sg) / max(1, len(cg))
        return overlap > 0.5
    except Exception:
        return False


def _summarize_chunk(model, kind: Optional[str], chunk: str, max_len: int, min_len: int) -> Optional[str]:
    _logger.debug(
        "Summarize chunk: kind=%s words=%d max_len=%d min_len=%d",
        kind, len(chunk.split()), max_len, min_len,
    )
    if kind == 'flan':
        # Use a comprehensive summarization prompt for flan-t5
        prompt = (
            "Write one comprehensive paragraph that accurately summarizes the entire passage. "
            "Focus on the central topic, key themes, entities, numbers, and chronology. Avoid missing major points. "
            "Use clear, cohesive prose without bullet points.\n\n"
            f"{chunk}\n\nSummary:"
        )
        out = model(
            prompt,
            max_new_tokens=max_len,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=5,
            repetition_penalty=1.15,
            early_stopping=True,
        )
        if isinstance(out, list) and out:
            result = out[0].get("generated_text", "").strip()
            # Clean up the result - remove the prompt if it's included
            if "Summary:" in result:
                result = result.split("Summary:")[-1].strip()
            return result
        return None
    else:
        out = model(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            num_beams=5,
            no_repeat_ngram_size=5,
            repetition_penalty=1.15,
            early_stopping=True,
        )
        if isinstance(out, list) and out:
            return out[0].get("summary_text", "").strip()
        return None


def _summarize_chunks_parallel(model, kind: Optional[str], chunks: List[str], max_workers: Optional[int] = None) -> List[str]:
    """Summarize chunks in parallel to speed up long transcripts. Preserves order."""
    if not chunks:
        return []
    workers = max_workers or min(8, (os.cpu_count() or 4))
    _logger.info("Parallel summarization starting: chunks=%d workers=%d", len(chunks), workers)
    results: List[Optional[str]] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(_summarize_chunk, model, kind, ch, 220, 90): i
            for i, ch in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                s = future.result() or ""
                results[i] = s
                _logger.debug("Parallel chunk %d/%d summarized len_chars=%d", i + 1, len(chunks), len(s))
            except Exception:
                _logger.exception("Parallel chunk %d failed", i + 1)
                results[i] = ""
    summarized = [s for s in results if (s or "").strip()]
    _logger.info("Parallel summarization complete: ok=%d/%d", len(summarized), len(chunks))
    return summarized


def generate_summary(text: str, max_len: int = 320, min_len: int = 140, length: Optional[str] = None) -> Optional[str]:
    """Generate one comprehensive summary covering the entire transcript.

    The length parameter is ignored; we always produce a single, thorough summary.
    """
    model, kind = _get_summarizer()
    if model is None or not text.strip():
        return None
    input_text = text.strip()
    _logger.info("Generate summary: input_chars=%d (comprehensive mode)", len(input_text))
    try:
        # LED fallback for very long transcripts (approx > 12k chars ~ 3-4k tokens)
        use_led = len(input_text) > 12000
        if use_led:
            led = _get_led_summarizer()
            if led is not None:
                _logger.info("Using LED summarizer for long input: chars=%d", len(input_text))
                try:
                    out = led(
                        input_text,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False,
                        num_beams=4,
                        no_repeat_ngram_size=5,
                        repetition_penalty=1.15,
                        early_stopping=True,
                    )
                    if isinstance(out, list) and out:
                        raw = out[0].get("summary_text", "").strip()
                        return _postprocess_dedupe(raw)
                except Exception:
                    _logger.exception("LED summarization failed, falling back to map-reduce BART")
        # Map-reduce summarization for long inputs
        chunks = _chunk_text(input_text, chunk_size=2200)
        if len(chunks) == 1:
            first = _summarize_chunk(model, kind, chunks[0], max_len, min_len)
            if first and _looks_extractive(first, chunks[0]):
                _logger.debug("Single-chunk output looked extractive; regenerating without leaking prompt text")
                if kind == 'flan':
                    paraphrase_prompt = f"Rewrite this into a cohesive, comprehensive paragraph that covers all major points without copying phrases:\n\n{first}\n\nRewritten:"
                    second = _summarize_chunk(model, kind, paraphrase_prompt, max_len, min_len)
                else:
                    # For BART, avoid instruction prompts; re-summarize the draft to encourage abstraction
                    second = _summarize_chunk(model, kind, first, max_len, min_len)
                first = second or first
            _logger.info("Summary complete (single-chunk) ok=%s len_chars=%d", bool(first), len(first or ""))
            return first
        # Parallelize per-chunk summarization for speed on long transcripts
        interim: List[str] = _summarize_chunks_parallel(model, kind, chunks)
        if not interim:
            _logger.warning("Interim summarization produced no outputs; returning None")
            return None
        combined = "\n".join(interim)
        # Final consolidation with a slightly larger window to capture all key topics
        final = _summarize_chunk(model, kind, combined, max_len=max_len, min_len=min_len)
        if final and _looks_extractive(final, input_text):
            _logger.debug("Map-reduce final looked extractive; regenerating without instruction prompt")
            if kind == 'flan':
                paraphrase_prompt = f"Rewrite this into a cohesive, comprehensive paragraph, preserving all key points while avoiding copied phrases:\n\n{final}\n\nRewritten:"
                retry = _summarize_chunk(model, kind, paraphrase_prompt, max_len, min_len)
            else:
                # For BART, simply re-summarize the final draft to reduce source copying
                retry = _summarize_chunk(model, kind, final, max_len, min_len)
            final = retry or final
        # Final clean-up to remove repeated sentences/phrases
        if final:
            final = _postprocess_dedupe(final)
        _logger.info(
            "Summary complete (map-reduce) chunks=%d final_ok=%s len_chars=%d",
            len(chunks), bool(final), len(final or ""),
        )
        return final
    except Exception:
        _logger.exception("generate_summary failed")
        return None
def _postprocess_dedupe(text: str) -> str:
    """Remove near-duplicate sentences and collapse whitespace to reduce repetition artifacts."""
    try:
        import re
        # Split into sentences (simple heuristic)
        parts = re.split(r'(?:[.!?]\s+)|\n+', text.strip())
        cleaned: List[str] = []
        seen = set()
        def norm(s: str) -> str:
            s2 = re.sub(r'[^a-z0-9\s]', '', s.lower())
            s2 = re.sub(r'\s+', ' ', s2).strip()
            return s2
        for p in parts:
            s = (p or '').strip()
            if not s:
                continue
            key = norm(s)
            # Skip very short or already seen sentences
            if len(key) < 8:
                continue
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(s)
        # Also remove immediate duplicate phrases within each sentence (n-gram like)
        def dedupe_phrases(s: str) -> str:
            tokens = s.split()
            out_tokens: List[str] = []
            last = None
            for tok in tokens:
                if last is not None and tok.lower() == last.lower():
                    continue
                out_tokens.append(tok)
                last = tok
            return ' '.join(out_tokens)
        cleaned = [dedupe_phrases(s) for s in cleaned]
        result = '. '.join(cleaned).strip()
        if result and not result.endswith('.'):
            result += '.'
        return result
    except Exception:
        return text


