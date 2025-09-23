import os
from supabase import create_client


def _delete_all(sb, table_name: str) -> None:
    # Select IDs then delete in batches to satisfy WHERE requirement
    try:
        res = sb.table(table_name).select("id").execute()
        ids = [row["id"] for row in (getattr(res, "data", None) or [])]
        if not ids:
            print(f"{table_name}: nothing to delete")
            return
        batch = 500
        for i in range(0, len(ids), batch):
            chunk = ids[i:i+batch]
            sb.table(table_name).delete().in_("id", chunk).execute()
        print(f"{table_name}: deleted {len(ids)} rows")
    except Exception as e:
        print(f"{table_name} delete error:", e)


def main() -> None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY env vars")
    sb = create_client(url, key)

    # Delete in dependency order: summaries -> transcript_chunks -> transcripts
    print("Deleting from summaries...")
    _delete_all(sb, "summaries")

    print("Deleting from transcript_chunks...")
    _delete_all(sb, "transcript_chunks")

    print("Deleting from transcripts...")
    _delete_all(sb, "transcripts")

    print("Done. All Supabase tables cleared.")


if __name__ == "__main__":
    main()


