"""Download poetry metadata and accessible TXT texts from Project Ben-Yehuda.

Requires: pandas, requests, and a private api_secrets.py containing API_KEY.
All works are preserved as raw text. Splitting/cleaning comes later.
"""

from pathlib import Path
import time

import pandas as pd
import requests

from api_secrets import API_KEY

# (Project Ben-Yehuda authority ID, canonical label in the ML dataset)
AUTHORS = (
    (89, "חיים נחמן ביאליק"),
    (57, "שאול טשרניחובסקי"),
    (141, "רחל בלובשטיין"),
    (372, "אברהם שלונסקי"),
    (450, "דן אלמגור"),
    (161, "יהודה הלוי"),
    (180, "שלמה אבן גבירול"),
    (577, "רחל שפירא"),
)

SEARCH_URL = "https://benyehuda.org/api/v1/search"
OUTPUT_DIR = Path("poetry_data")
INDEX_CSV = OUTPUT_DIR / "poetry_index.csv"
DATASET_CSV = OUTPUT_DIR / "poetry_dataset.csv"
REQUEST_INTERVAL = 3  # Conservative rate: at most ~20 requests/minute
CHECKPOINT_EVERY = 20
MAX_WORKS_PER_AUTHOR = None  # Set to 2 for a quick test; None = all works

session = requests.Session()
_last_request_time = 0.0


def fetch(method, url, **kwargs):
    """Throttled request with limited retries; never print the API key."""
    global _last_request_time
    for attempt in range(5):
        wait = REQUEST_INTERVAL - (time.monotonic() - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()
        try:
            response = session.request(method, url, timeout=60, **kwargs)
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(60, 3 * 2 ** attempt))
            continue
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt == 4:
                response.raise_for_status()
            try:
                retry_after = float(response.headers.get("Retry-After", "0"))
            except ValueError:
                retry_after = 0
            time.sleep(max(retry_after, min(60, 3 * 2 ** attempt)))
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("Request failed after retries")


def collect_index():
    """Search all pages for each author's poetry; keep works as separate rows."""
    rows = []
    for author_id, author_name in AUTHORS:
        search_after = None
        seen_pages = set()
        n_before = len(rows)

        while True:
            payload = {
                "key": API_KEY,
                "author_ids": [author_id],
                "genres": ["poetry"],
                "view": "metadata",
                "file_format": "txt",
            }
            if search_after is not None:
                payload["search_after"] = search_after

            result = fetch("POST", SEARCH_URL, json=payload).json()
            for work in result.get("data", []):
                metadata = work.get("metadata") or {}
                ids = metadata.get("author_ids") or []
                rows.append({
                    "Post_ID": int(work["id"]),  # Same column as the old FB dataset
                    "Work_ID": int(work["id"]),
                    "PageName": author_name,
                    "Author_ID": author_id,
                    "Title": metadata.get("title", ""),
                    "Genre": metadata.get("genre", ""),
                    "Author_String": metadata.get("author_string", ""),
                    "Author_IDs": ",".join(map(str, ids)),
                    "Original_Language": metadata.get("orig_lang", ""),
                    "Rights": metadata.get("intellectual_property", ""),
                    "Source_URL": work.get("url", ""),
                    "Download_URL": work.get("download_url", ""),
                })

            search_after = result.get("next_page_search_after")
            if not search_after:
                break
            cursor = tuple(search_after)
            if cursor in seen_pages:
                raise RuntimeError(f"Repeated pagination cursor for {author_name}")
            seen_pages.add(cursor)

        print(f"{author_name}: {len(rows) - n_before} poetry records")
        pd.DataFrame(rows).drop(columns=["Download_URL"], errors="ignore").to_csv(
            INDEX_CSV, index=False, encoding="utf-8-sig"
        )

    index_df = pd.DataFrame(rows).drop_duplicates(["Author_ID", "Work_ID"])
    index_df.drop(columns=["Download_URL"], errors="ignore").to_csv(
        INDEX_CSV, index=False, encoding="utf-8-sig"
    )
    return index_df


def collect_poetry():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_df = collect_index()
    if index_df.empty:
        raise RuntimeError("The API returned no poetry works. Check the key and query.")

    # Resume: preserve already-downloaded texts after an interrupted run.
    saved = {}
    if DATASET_CSV.exists():
        previous = pd.read_csv(DATASET_CSV, keep_default_na=False)
        saved = {
            (int(row.Author_ID), int(row.Work_ID)): (row.Data, row.download_error)
            for row in previous.itertuples(index=False)
        }

    rows = []
    for _, index_row in index_df.iterrows():
        row = index_row.to_dict()
        key = (int(row["Author_ID"]), int(row["Work_ID"]))
        row["Data"], row["download_error"] = saved.get(key, ("", ""))
        rows.append(row)

    if MAX_WORKS_PER_AUTHOR is not None:
        keep = {tuple(x) for x in index_df.groupby("Author_ID").head(MAX_WORKS_PER_AUTHOR)[["Author_ID", "Work_ID"]].itertuples(index=False, name=None)}
    else:
        keep = None

    remaining = 0
    for n, row in enumerate(rows, start=1):
        key = (int(row["Author_ID"]), int(row["Work_ID"]))
        if keep is not None and key not in keep:
            continue
        if row["Data"]:
            continue

        try:
            if not row["Download_URL"]:
                raise ValueError("Missing TXT download URL")
            response = fetch("GET", row["Download_URL"])
            content = response.content.decode("utf-8-sig")
            if content.lstrip().lower().startswith(("<!doctype html", "<html")):
                raise ValueError("Received HTML instead of TXT")
            row["Data"] = content
            row["download_error"] = ""
        except (requests.RequestException, ValueError, UnicodeError) as exc:
            # Do not record exception URLs: some APIs embed credentials in URLs.
            status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else ""
            row["download_error"] = f"{type(exc).__name__} {status}".strip()
            print(f"Download failed: {row['PageName']} / {row['Work_ID']} ({row['download_error']})")
        remaining += 1

        if remaining % CHECKPOINT_EVERY == 0:
            save_dataset(rows)
            print(f"Downloaded/attempted {remaining} new texts; checkpoint saved")

    return save_dataset(rows)


def save_dataset(rows):
    poems_df = pd.DataFrame(rows)
    poems_df["text_length"] = poems_df["Data"].str.len()
    # Do not create split here: later split by WHOLE WORK, not text fragments.
    poems_df["valid"] = poems_df["text_length"] > 0
    # Do not store possibly authenticated download URLs in saved CSVs.
    poems_df = poems_df.drop(columns=["Download_URL"], errors="ignore")
    poems_df.to_csv(DATASET_CSV, index=False, encoding="utf-8-sig")
    return poems_df


if __name__ == "__main__":
    poems_df = collect_poetry()
    print("\nDataset:", DATASET_CSV)
    print("Index:", INDEX_CSV)
    print(poems_df.groupby("PageName").agg(
        works=("Work_ID", "count"),
        downloaded=("valid", "sum"),
        total_characters=("text_length", "sum"),
    ).to_string())
