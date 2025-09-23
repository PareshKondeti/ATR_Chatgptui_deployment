from supabase import create_client
import os


def main() -> None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

    sb = create_client(url, key)

    ins = sb.table("transcripts").insert({"filename": "test.wav", "text": "hello world"}).execute()
    print("insert:", getattr(ins, "data", None))

    sel = (
        sb.table("transcripts")
        .select("id, filename, created_at")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    print("select:", getattr(sel, "data", None))


if __name__ == "__main__":
    main()


