import os
from typing import Any, Dict, List, Optional

try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = None  # type: ignore


_SUPABASE_URL = os.getenv("SUPABASE_URL", "")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

TABLE_TRANSCRIPTS = "transcripts"
TABLE_CHUNKS = "transcript_chunks"
TABLE_SUMMARIES = "summaries"


def get_supabase() -> Optional["Client"]:
    # Read environment at call time to reflect any updated values
    url = os.getenv("SUPABASE_URL", _SUPABASE_URL)
    key = os.getenv("SUPABASE_KEY", _SUPABASE_KEY)
    if not url or not key or create_client is None:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def ensure_tables_note() -> str:
    return (
        "Ensure Supabase tables exist:\n"
        "- transcripts: id uuid pk, filename text, text text, created_at timestamptz\n"
        "- transcript_chunks: id uuid pk, transcript_id uuid fk, chunk_index int, text text, embedding vector, created_at timestamptz\n"
        "- summaries: id uuid pk, transcript_id uuid fk, summary text, created_at timestamptz\n"
        "Optionally create RPC for vector search, else client-side cosine is used."
    )


def insert_transcript(supabase: "Client", filename: str, text: str) -> Optional[str]:
    try:
        r = supabase.table(TABLE_TRANSCRIPTS).insert({"filename": filename, "text": text}).execute()
        data = getattr(r, "data", None) or []
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return None


def upsert_transcript(supabase: "Client", filename: str, text: str) -> Optional[str]:
    """Insert or update transcript by filename to avoid duplicates."""
    try:
        # Check if transcript with this filename already exists
        existing = supabase.table(TABLE_TRANSCRIPTS).select("id").eq("filename", filename).execute()
        if existing.data:
            # Update existing transcript
            transcript_id = existing.data[0]["id"]
            supabase.table(TABLE_TRANSCRIPTS).update({"text": text}).eq("id", transcript_id).execute()
            return transcript_id
        else:
            # Insert new transcript
            r = supabase.table(TABLE_TRANSCRIPTS).insert({"filename": filename, "text": text}).execute()
            data = getattr(r, "data", None) or []
            if data:
                return data[0]["id"]
    except Exception:
        pass
    return None


def get_transcript_by_filename(supabase: "Client", filename: str) -> Optional[str]:
    try:
        r = (
            supabase.table(TABLE_TRANSCRIPTS)
            .select("id")
            .eq("filename", filename)
            .limit(1)
            .execute()
        )
        data = getattr(r, "data", None) or []
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return None


def insert_summary(supabase: "Client", transcript_id: str, summary: str) -> Optional[str]:
    """Upsert summary for this transcript: update if exists, else insert."""
    try:
        existing = (
            supabase.table(TABLE_SUMMARIES)
            .select("id")
            .eq("transcript_id", transcript_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if getattr(existing, "data", None):
            sid = existing.data[0]["id"]
            supabase.table(TABLE_SUMMARIES).update({"summary": summary}).eq("id", sid).execute()
            return sid
        r = supabase.table(TABLE_SUMMARIES).insert({"transcript_id": transcript_id, "summary": summary}).execute()
        data = getattr(r, "data", None) or []
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return None


def fetch_transcript_text(supabase: "Client", transcript_id: str) -> Optional[str]:
    try:
        r = (
            supabase.table(TABLE_TRANSCRIPTS)
            .select("text")
            .eq("id", transcript_id)
            .limit(1)
            .execute()
        )
        data = getattr(r, "data", None) or []
        if data:
            return data[0].get("text") or None
    except Exception:
        return None
    return None


def insert_chunks(
    supabase: "Client",
    transcript_id: str,
    chunks: List[Dict[str, Any]],
) -> bool:
    try:
        rows = []
        for c in chunks:
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "chunk_index": c["index"],
                    "text": c["text"],
                    # pgvector accepts list[float]
                    "embedding": c.get("embedding", []),
                }
            )
        supabase.table(TABLE_CHUNKS).insert(rows).execute()
        return True
    except Exception:
        return False


def delete_chunks_for_transcript(supabase: "Client", transcript_id: str) -> bool:
    try:
        supabase.table(TABLE_CHUNKS).delete().eq("transcript_id", transcript_id).execute()
        return True
    except Exception:
        return False


def fetch_chunks_for_transcript(supabase: "Client", transcript_id: str) -> List[Dict[str, Any]]:
    try:
        r = supabase.table(TABLE_CHUNKS).select("id, chunk_index, text, embedding").eq("transcript_id", transcript_id).order("chunk_index").execute()
        return getattr(r, "data", None) or []
    except Exception:
        return []


def fetch_summary_for_transcript(supabase: "Client", transcript_id: str) -> Optional[str]:
    """Return the most recent summary text for a transcript if present."""
    try:
        r = (
            supabase.table(TABLE_SUMMARIES)
            .select("summary, created_at")
            .eq("transcript_id", transcript_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        data = getattr(r, "data", None) or []
        if data:
            return data[0].get("summary") or None
    except Exception:
        return None
    return None


def fetch_latest_transcript_id(supabase: "Client") -> Optional[str]:
    try:
        r = supabase.table(TABLE_TRANSCRIPTS).select("id").order("created_at", desc=True).limit(1).execute()
        data = getattr(r, "data", None) or []
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return None


def fetch_all_chunks(supabase: "Client") -> List[Dict[str, Any]]:
    try:
        r = supabase.table(TABLE_CHUNKS).select("id, transcript_id, chunk_index, text, embedding").order("created_at").execute()
        return getattr(r, "data", None) or []
    except Exception:
        return []


def list_transcripts(supabase: "Client") -> List[Dict[str, Any]]:
    """Return id, filename, created_at for all transcripts, deduplicated by filename."""
    try:
        r = (
            supabase.table(TABLE_TRANSCRIPTS)
            .select("id, filename, created_at")
            .order("created_at", desc=True)
            .execute()
        )
        data = getattr(r, "data", None) or []
        # Deduplicate by filename, keeping the most recent
        seen_filenames = set()
        unique_transcripts = []
        for item in data:
            filename = item.get("filename", "")
            if filename and filename not in seen_filenames:
                seen_filenames.add(filename)
                unique_transcripts.append(item)
        return unique_transcripts
    except Exception:
        return []


