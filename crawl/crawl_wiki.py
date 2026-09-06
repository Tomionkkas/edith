#!/usr/bin/env python3
"""Marvel SLM - Wikipedia (Tier A) crawler.

Walks the Marvel category tree on EN Wikipedia recursively, then fetches each
article's wikitext in batches of 50. Cleaning happens later via curate.py,
exactly like the Fandom side.

Note on `extracts`: Wikipedia refuses to batch whole-article extracts -- it
answers "exlimit was too large for a whole article extracts request, lowered
to 1" and returns text for a single page, empty for the rest. Wikitext has no
such limit, so we take wikitext and strip markup ourselves.

Runs completely independently of the Fandom crawler: different host, different
state file, different output file. Safe to run at the same time.

Modes:
  py crawl_wiki.py             # enumerate (cached) + fetch, resumable
  py crawl_wiki.py --status    # progress
  py crawl_wiki.py --stop      # graceful stop at the next batch boundary
  py crawl_wiki.py --walk-only # just build the category tree, fetch nothing
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)
STATE_DIR = RAW_DIR / "state"
STATE_DIR.mkdir(exist_ok=True)
STOP_FILE = RAW_DIR / "STOP_WIKI"
OUT_FILE = RAW_DIR / "wiki_marvel.jsonl"
STATE_FILE = STATE_DIR / "wiki_marvel.json"
TREE_FILE = STATE_DIR / "wiki_marvel_tree.json"

API = "https://en.wikipedia.org/w/api.php"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MIN_INTERVAL = 1.0
FETCH_RETRIES = 3
ARTICLE_BATCH = 50        # MediaWiki multi-title limit
MAX_DEPTH = 3
MIN_CHARS = 500           # below this it is a stub/disambig, not worth keeping

ROOTS = [
    "Category:Marvel Comics characters",
    "Category:Marvel Comics storylines",
    "Category:Marvel Comics titles",
    "Category:Characters in Marvel Comics alternate realities",
    "Category:Marvel Comics crossovers",
    "Category:Comics by Marvel Comics",
]

# Wikipedia's category graph leaks into the whole encyclopedia within a few
# hops (Marvel Comics -> American culture -> ...). Only descend into
# subcategories that still look Marvel/comics related.
SUBCAT_ALLOW = re.compile(
    r"marvel|comic|spider-man|x-men|avengers|fantastic four|hulk|iron man|"
    r"thor|captain america|daredevil|wolverine|mutant|superhero|supervillain|"
    r"asgard|wakanda|earth-\d+|multiverse",
    re.I)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def check_stop() -> bool:
    return STOP_FILE.exists()


def http_json(params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    if len(qs) <= 7000:
        req = urllib.request.Request(f"{API}?{qs}", headers={"User-Agent": UA})
    else:
        req = urllib.request.Request(API, data=qs.encode("utf-8"),
                                     headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def api_retry(params: dict):
    """http_json with 429/5xx backoff. Returns None after exhausting retries."""
    for attempt in range(FETCH_RETRIES):
        try:
            return http_json(params)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(60, int(e.headers.get("Retry-After", 5) or 5)))
                continue
            if 500 <= e.code < 600:
                time.sleep(2 ** attempt * 2)
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2 ** attempt * 2)
            continue
        except Exception:
            return None
    return None


# ------------------------------------------------------------- category walk

def walk_categories(fetch_members, roots, max_depth: int = MAX_DEPTH,
                    allow=SUBCAT_ALLOW):
    """BFS the category tree -> (sorted unique page titles, walked categories).

    `fetch_members(cat)` returns [(title, "page"|"subcat"), ...].
    `allow` is an optional compiled regex gating which subcategories we descend
    into; pass None to follow every subcategory.

    The real category graph contains cycles, so every visited category is
    recorded in `seen` and never expanded twice; without that this does not
    terminate.
    """
    seen: set[str] = set()
    pages: set[str] = set()
    queue = [(r, 0) for r in roots]
    while queue:
        cat, depth = queue.pop(0)
        if cat in seen or depth > max_depth:
            continue
        seen.add(cat)
        for title, kind in fetch_members(cat):
            if kind == "page":
                pages.add(title)
            elif kind == "subcat" and depth < max_depth:
                if title in seen:
                    continue
                if allow is not None and not allow.search(title):
                    continue
                queue.append((title, depth + 1))
    return sorted(pages), sorted(seen)


def live_fetch_members(cat: str):
    """All members of one category, following continuation."""
    out, cont = [], None
    while True:
        params = {"action": "query", "list": "categorymembers", "cmtitle": cat,
                  "cmlimit": "500", "cmtype": "page|subcat", "format": "json"}
        if cont:
            params.update(cont)
        data = api_retry(params)
        if data is None:
            break
        for m in data.get("query", {}).get("categorymembers", []):
            ns = m.get("ns")
            if ns == 0:
                out.append((m["title"], "page"))
            elif ns == 14:
                out.append((m["title"], "subcat"))
        cont = data.get("continue")
        if not cont:
            break
        time.sleep(MIN_INTERVAL)
    return out


# ------------------------------------------------------------ article fetch

_fetch_batch = None


def _batcher():
    """Reuse the tested batch fetcher from the Fandom crawler."""
    p = Path(__file__).resolve().parent / "crawl.py"
    spec = importlib.util.spec_from_file_location("crawl_core", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.fetch_wikitext_batch


def fetch_articles_batch(titles: list) -> tuple[dict, dict]:
    """Wikitext for up to ARTICLE_BATCH articles in one request.

    Returns ({requested_title: wikitext}, {requested_title: error}), keyed by
    the exact strings we asked for (MediaWiki normalises titles).
    """
    global _fetch_batch
    if _fetch_batch is None:
        _fetch_batch = _batcher()
    pages, errors = _fetch_batch(API, list(titles))
    for t, wt in list(pages.items()):
        if len(wt) < MIN_CHARS:
            errors[t] = "too-short"
            del pages[t]
    return pages, errors


# ------------------------------------------------------------------- state

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"name": "wiki_marvel", "done_titles": [], "errors": [], "titles": [],
            "titles_total": None}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def show_status() -> None:
    st = load_state()
    done, errs = len(st.get("done_titles", [])), len(st.get("errors", []))
    total = st.get("titles_total") or "?"
    print(f"{'file':<16}{'done':>9}{'errors':>9}{'total':>10}")
    print(f"{'wiki_marvel':<16}{done:>9}{errs:>9}{str(total):>10}")
    if OUT_FILE.exists():
        mb = OUT_FILE.stat().st_size / 1e6
        age = (time.time() - OUT_FILE.stat().st_mtime) / 60
        print(f"  {OUT_FILE.name}: {mb:,.1f} MB, last write {age:.0f} min ago")
    if STOP_FILE.exists():
        print("STOP_WIKI present - crawler will exit cleanly.")


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--walk-only", action="store_true")
    ap.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    args = ap.parse_args()

    if args.status:
        show_status()
        return 0
    if args.stop:
        STOP_FILE.write_text(time.strftime("%Y-%m-%d %H:%M:%S stop requested\n"))
        print("Stop file created. The wiki crawler will exit at the next batch.")
        return 0

    t_start = time.time()
    state = load_state()

    if state.get("titles"):
        titles = state["titles"]
        log(f"resuming with saved tree ({len(titles)} pages, "
            f"{len(state['done_titles'])} done)")
    else:
        log(f"walking {len(ROOTS)} root categories (max depth {args.max_depth})...")
        titles, cats = walk_categories(live_fetch_members, ROOTS, args.max_depth)
        TREE_FILE.write_text(json.dumps({"categories": cats, "pages": titles}),
                             encoding="utf-8")
        state["titles"] = titles
        state["titles_total"] = len(titles)
        save_state(state)
        log(f"walked {len(cats)} categories -> {len(titles)} unique articles")

    if args.walk_only:
        return 0

    done = set(state["done_titles"])
    pending = [t for t in titles if t not in done]
    log(f"{len(pending)} articles pending in "
        f"{(len(pending) + ARTICLE_BATCH - 1) // ARTICLE_BATCH} batches")

    buf: list[dict] = []
    total_done = total_errs = last_progress = 0

    def flush():
        if buf:
            with OUT_FILE.open("a", encoding="utf-8") as fh:
                for rec in buf:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            buf.clear()

    for i in range(0, len(pending), ARTICLE_BATCH):
        if check_stop():
            log("Stop requested - exiting cleanly.")
            break
        chunk = pending[i:i + ARTICLE_BATCH]
        t_req = time.time()
        got, errs = fetch_articles_batch(chunk)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for title, text in got.items():
            buf.append({"file": "wiki_marvel", "title": title, "api": API,
                        "fetched_at": stamp, "wikitext": text})
            state["done_titles"].append(title)
            total_done += 1
        for title, err in errs.items():
            state["errors"].append({"title": title, "error": err})
            if err in ("missing", "too-short", "invalid"):   # never worth retrying
                state["done_titles"].append(title)
            total_errs += 1
        if len(buf) >= ARTICLE_BATCH:
            flush()
        if total_done - last_progress >= 100:
            last_progress = total_done
            el = time.time() - t_start
            log(f"progress: {total_done} ok / {total_errs} skipped in {el/60:.0f}m "
                f"({(total_done + total_errs) / max(el, 1):.1f} pages/s)")
        if (i // ARTICLE_BATCH) % 10 == 0:
            save_state(state)
        time.sleep(max(0.0, MIN_INTERVAL - (time.time() - t_req)))

    flush()
    save_state(state)
    STOP_FILE.unlink(missing_ok=True)
    el = time.time() - t_start
    log(f"wiki crawl finished after {el/60:.1f} min. "
        f"{total_done} articles, {total_errs} skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
