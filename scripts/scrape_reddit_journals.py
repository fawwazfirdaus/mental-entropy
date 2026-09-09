"""Scrape journal-like posts from r/offmychest using Reddit's public JSON API."""
from __future__ import annotations

import csv
import time
import urllib.request
import urllib.error
import json
from pathlib import Path

SUBREDDIT = "offmychest"
BASE_URL = f"https://www.reddit.com/r/{SUBREDDIT}"
USER_AGENT = "MentalEntropyResearch/1.0 (academic; contact: research@example.com)"
MIN_LENGTH = 100
TARGET = 500
DELAY = 1.5  # seconds between requests

# Sort options to maximize unique posts
SORT_OPTIONS = [
    ("top", {"t": "all"}),
    ("top", {"t": "year"}),
    ("top", {"t": "month"}),
    ("hot", {}),
    ("new", {}),
    ("top", {"t": "week"}),
    ("rising", {}),
]

OUTPUT_PATH = Path(
    "/Users/karanpatil/conductor/workspaces/mental-entropy/vancouver"
    "/data/additional_journal_data_cleaned_v2.csv"
)

SKIP_TEXTS = {"", "[removed]", "[deleted]"}


def fetch_page(sort: str, params: dict[str, str], after: str | None = None) -> dict:
    """Fetch one page of Reddit listings."""
    query_params = {"limit": "100", "raw_json": "1"}
    query_params.update(params)
    if after:
        query_params["after"] = after

    query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
    url = f"{BASE_URL}/{sort}.json?{query_string}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} for {url}")
        return {}
    except Exception as e:
        print(f"  Error: {e}")
        return {}


def extract_posts(data: dict) -> tuple[list[tuple[str, str]], str | None]:
    """Extract (post_id, selftext) pairs and the 'after' cursor."""
    posts = []
    if not data or "data" not in data:
        return posts, None

    listing = data["data"]
    for child in listing.get("children", []):
        post = child.get("data", {})
        # Only self-posts
        if not post.get("is_self", False):
            continue
        selftext = (post.get("selftext") or "").strip()
        post_id = post.get("id", "")
        if selftext in SKIP_TEXTS:
            continue
        if len(selftext) < MIN_LENGTH:
            continue
        posts.append((post_id, selftext))

    after = listing.get("after")
    return posts, after


def main() -> None:
    seen_ids: set[str] = set()
    collected: list[str] = []

    for sort, params in SORT_OPTIONS:
        label = f"{sort}({','.join(f'{k}={v}' for k, v in params.items())})" if params else sort
        print(f"\n--- Fetching: {label} ---")
        after = None
        page = 0

        while True:
            page += 1
            data = fetch_page(sort, params, after)
            posts, after = extract_posts(data)

            new_count = 0
            for pid, text in posts:
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    collected.append(text)
                    new_count += 1

            print(f"  Page {page}: {len(posts)} valid posts, {new_count} new | Total: {len(collected)}")

            if len(collected) >= TARGET:
                print(f"  Reached target of {TARGET}+")
                break

            if not after:
                print("  No more pages.")
                break

            # Max 10 pages per sort to avoid excessive requests
            if page >= 10:
                print("  Hit page limit (10).")
                break

            time.sleep(DELAY)

        if len(collected) >= TARGET:
            break

        time.sleep(DELAY)

    # Save to CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["full_text"])
        for text in collected:
            writer.writerow([text])

    print(f"\nDone! Saved {len(collected)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
