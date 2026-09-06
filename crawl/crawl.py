#!/usr/bin/env python3
"""Marvel SLM — Fandom harvest crawler (production, full crawl).

Crawls every target category on Marvel Fandom + a few backstop Wikipedia
categories, saves raw wikitext as JSONL, resumable.

Pages are fetched 50-at-a-time (MediaWiki's multi-title limit), so one HTTP
round-trip yields 50 pages instead of 1 at the same polite request rate.

Modes:
  python crawl.py              # full crawl (resumes from state)
  python crawl.py --status     # show per-file progress from state files
  python crawl.py --stop       # create stop file (crawling run exits cleanly)
  python crawl.py --test N     # legacy: stop after N total pages

Graceful stop (crawling finishes the current page, flushes all state, exits 0):
  1. `python crawl.py --stop`  -> creates <RAW_DIR>/STOP (works from any shell)
  2. SIGINT / SIGTERM on the process (Ctrl+C, taskkill)
  3. `touch raw/STOP` directly
Resume by re-running: done titles are skipped via state files.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import signal
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
STOP_FILE = RAW_DIR / "STOP"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MIN_INTERVAL = 1.0        # seconds between requests (Fandom rate-limits anyway)
FETCH_RETRIES = 3
PROGRESS_EVERY = 100      # log a progress line every N pages

# full-crawl targets: name -> (api_base, category_title)
FANDOM = "https://marvel.fandom.com/api.php"
ENWIKI = "https://en.wikipedia.org/w/api.php"
TARGETS = [
    ("characters",  FANDOM, "Category:Characters"),
    ("comics",      FANDOM, "Category:Comics"),
    ("teams",       FANDOM, "Category:Teams"),
    ("events",      FANDOM, "Category:Events"),
    ("story_arcs",  FANDOM, "Category:Story Arcs"),
    # Items and Locations were missing from the first crawl, and the gap was
    # only visible from the answers: nothing in 187,584 records owned the name
    # "Infinity Stones", and Wakanda existed only as a Wikipedia article.
    # 3,655 + 10,647 pages, flat category members like every other target.
    ("items",       FANDOM, "Category:Items"),
    ("locations",   FANDOM, "Category:Locations"),
    ("mw_character_backstop", ENWIKI, "Category:Comics characters"),
    ("mw_marvel_backstop",    ENWIKI, "Category:Marvel Comics characters"),
]

_stop_requested = False


def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    sys.stderr.write("\n[crawl] stop signal received — finishing current page, flushing state...\n")
    sys.stderr.flush()


def stop_file_exists() -> bool:
    return STOP_FILE.exists()


def check_stop() -> bool:
    return _stop_requested or stop_file_exists()


def http_get_json(url: str, params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{qs}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def fetch_wikitext(api: str, title: str) -> tuple[str | None, str | None]:
    """Return (wikitext, error). Retry w/ backoff on transient failures."""
    params = {"action": "query", "titles": title, "prop": "revisions",
              "rvprop": "content", "format": "json"}
    for attempt in range(FETCH_RETRIES):
        try:
            data = http_get_json(api, params)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if "missing" in page:
                    return None, "missing"
                if "invalid" in page:
                    return None, "invalid"
                return page.get("revisions", [{}])[0].get("*"), None
            return None, "no-page"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(30, int(e.headers.get("Retry-After", 5) or 5))
                time.sleep(wait)
                continue
            if 500 <= e.code < 600:
                time.sleep(2 ** attempt)
                continue
            return None, f"http-{e.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            time.sleep(2 ** attempt)
            continue
        except Exception as e:
            return None, f"parse-{type(e).__name__}"
    return None, "retries-exhausted"


BATCH_SIZE = 50           # MediaWiki caps anonymous multi-title queries at 50
MAX_QS_BYTES = 7000       # switch to POST before we risk a URL-length 414


def http_json(url: str, params: dict) -> dict:
    """GET, or POST automatically when the query string gets long."""
    qs = urllib.parse.urlencode(params)
    if len(qs) <= MAX_QS_BYTES:
        req = urllib.request.Request(f"{url}?{qs}", headers={"User-Agent": UA})
    else:
        req = urllib.request.Request(url, data=qs.encode("utf-8"),
                                     headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def enumerate_titles_page(api: str, cat: str, limit: int = 500, cont: dict | None = None):
    """One page of category members -> (titles, continue-token-or-None)."""
    params = {"action": "query", "list": "categorymembers", "cmtitle": cat,
              "cmtype": "page", "cmlimit": str(limit), "format": "json"}
    if cont:
        params.update(cont)
    data = http_json(api, params)
    titles = [m["title"] for m in data.get("query", {}).get("categorymembers", [])
              if not m["title"].startswith(("File:", "Image:"))]
    return titles, data.get("continue")


def fetch_wikitext_batch(api: str, titles: list) -> tuple[dict, dict]:
    """Fetch up to BATCH_SIZE pages in ONE request.

    Returns ({requested_title: wikitext}, {requested_title: error}).

    MediaWiki silently normalises titles (Iron_Man -> Iron Man) and answers
    under the normalised name. Results are mapped back onto the exact strings
    we asked for, because those are the keys the resume state is built from --
    get this wrong and finished pages never look finished.
    """
    titles = list(titles)
    if not titles:
        return {}, {}
    params = {"action": "query", "titles": "|".join(titles),
              "prop": "revisions", "rvprop": "content", "rvslots": "main",
              "format": "json", "formatversion": "2"}

    data = None
    last_err = "retries-exhausted"
    for attempt in range(FETCH_RETRIES):
        try:
            data = http_json(api, params)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(30, int(e.headers.get("Retry-After", 5) or 5)))
                last_err = "http-429"
                continue
            if 500 <= e.code < 600:
                time.sleep(2 ** attempt)
                last_err = f"http-{e.code}"
                continue
            last_err = f"http-{e.code}"
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2 ** attempt)
            last_err = "network"
            continue
        except Exception as e:
            last_err = f"parse-{type(e).__name__}"
            break

    if data is None:
        # transient: leave every title un-done so a re-run retries it
        return {}, {t: last_err for t in titles}

    query = data.get("query", {})
    # normalised/converted name -> the string we actually asked for
    back = {}
    for key in ("normalized", "converted"):
        for n in query.get(key, []):
            back[n["to"]] = n["from"]

    pages, errors = {}, {}
    for page in query.get("pages", []):
        name = page.get("title", "")
        requested = back.get(name, name)
        if page.get("missing"):
            errors[requested] = "missing"
            continue
        if page.get("invalid"):
            errors[requested] = "invalid"
            continue
        try:
            wt = page["revisions"][0]["slots"]["main"]["content"]
        except (KeyError, IndexError, TypeError):
            errors[requested] = "no-content"
            continue
        if wt:
            pages[requested] = wt
        else:
            errors[requested] = "empty"

    # anything the API never mentioned at all
    for t in titles:
        if t not in pages and t not in errors:
            errors[t] = "no-page"
    return pages, errors


def enumerate_titles(api: str, cat: str) -> tuple[list[str], int]:
    titles: list[str] = []
    cont = None
    api_errs = 0
    while True:
        params = {"action": "query", "list": "categorymembers",
                  "cmtitle": cat, "cmtype": "page", "cmlimit": "500", "format": "json"}
        if cont:
            params.update(cont)
        try:
            data = http_get_json(api, params)
        except Exception as e:
            api_errs += 1
            if api_errs > 20:
                break
            time.sleep(3)
            continue
        api_errs = 0
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            t = m["title"]
            if t.startswith("File:") or t.startswith("Image:"):
                continue
            titles.append(t)
        cont = data.get("continue")
        if not cont:
            break
        time.sleep(MIN_INTERVAL)
    return titles, api_errs


def load_state(name: str) -> dict:
    p = STATE_DIR / f"{name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"name": name, "done_titles": [], "errors": [], "titles_total": None}


def save_state(state: dict) -> None:
    p = STATE_DIR / f"{state['name']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, p)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def show_status() -> None:
    print(f"{'file':<24} {'done':>8} {'errors':>8} {'total':>9}")
    grand = 0
    for st in sorted(STATE_DIR.glob("*.json")):
        try:
            d = json.loads(st.read_text(encoding="utf-8"))
        except Exception:
            continue
        n = len(d.get("done_titles", []))
        grand += n
        tot = d.get("titles_total") or "?"
        print(f"{d.get('name', st.stem):<24} {n:>8} {len(d.get('errors', [])):>8} {str(tot):>9}")
    print(f"{'TOTAL':<24} {grand:>8}")
    if STOP_FILE.exists():
        print("STOP file present — crawler will exit (or has exited) cleanly.")
    # rough age of latest data file
    for f in sorted(RAW_DIR.glob("*.jsonl")):
        if f.parent == RAW_DIR and f.name != "STOP":
            age = (time.time() - f.stat().st_mtime) / 60
            print(f"  last write to {f.name}: {age:.0f} min ago")


def main() -> int:
    global _stop_requested
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0, help="stop after N total pages (debug)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true", help="create stop file and exit")
    args = ap.parse_args()

    if args.status:
        show_status()
        return 0
    if args.stop:
        STOP_FILE.write_text(time.strftime("%Y-%m-%d %H:%M:%S stop requested\n"))
        print("Stop file created. The crawler will exit cleanly at the next page boundary.")
        return 0

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    t_start = time.time()
    log(f"Full crawl starting. Raw dir: {RAW_DIR}")
    log(f"Graceful stop: `python crawl.py --stop`, SIGINT/SIGTERM, or touch raw/STOP")

    buf: list[dict] = []
    all_states: dict[str, dict] = {}
    total_done = 0
    total_errs = 0
    last_progress = 0

    def flush() -> None:
        nonlocal total_done
        if buf:
            for rec in buf:
                p = RAW_DIR / f"{rec['file']}.jsonl"
                with p.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            buf.clear()

    for name, api, cat in TARGETS:
        if check_stop():
            log("Stop requested — skipping remaining targets.")
            break
        state = load_state(name)
        all_states[name] = state
        done = set(state.get("done_titles", []))
        if state.get("titles_total") and state.get("titles"):
            titles = state["titles"]
            log(f"{name}: resuming with saved titles ({len(titles)} total, {len(done)} done)")
        else:
            log(f"{name}: enumerating {cat} ...")
            titles, api_errs = enumerate_titles(api, cat)
            # de-dupe preserving order
            seen = set()
            titles = [t for t in titles if not (t in seen or seen.add(t))]
            state["titles"] = titles
            state["titles_total"] = len(titles)
            save_state(state)
            log(f"{name}: {len(titles)} unique titles enumerated (api_errs={api_errs})")

        pending = [t for t in titles if t not in done]
        if pending:
            nbatch = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
            log(f"{name}: {len(pending)} pages pending in {nbatch} batches of {BATCH_SIZE}")
        for i in range(0, len(pending), BATCH_SIZE):
            if check_stop():
                log("Stop requested — exiting current target cleanly.")
                break
            chunk = pending[i:i + BATCH_SIZE]
            t_req = time.time()
            got, errs = fetch_wikitext_batch(api, chunk)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            for title, wt in got.items():
                buf.append({"file": name, "title": title, "api": api,
                            "fetched_at": stamp, "wikitext": wt})
                done.add(title)
                state["done_titles"].append(title)
                total_done += 1
            for title, err in errs.items():
                state.setdefault("errors", []).append({"title": title, "error": err})
                total_errs += 1
            if len(buf) >= BATCH_SIZE or (check_stop() and buf):
                flush()
            if total_done - last_progress >= PROGRESS_EVERY:
                last_progress = total_done
                elapsed = time.time() - t_start
                rate = (total_done + total_errs) / max(elapsed, 1)
                log(f"progress: {total_done} ok / {total_errs} err in {elapsed/60:.0f}m "
                    f"({rate:.1f} pages/s) — current file: {name}")
            # checkpoint every 5 batches (~250 pages), same cadence as before
            if (i // BATCH_SIZE) % 5 == 0:
                save_state(state)
            # pace by REQUEST, counting the fetch itself against the interval
            time.sleep(max(0.0, MIN_INTERVAL - (time.time() - t_req)))

        save_state(state)
        flush()
        if args.test and total_done >= args.test:
            log(f"--test {args.test} reached; stopping after this file.")
            break

    # final cleanup
    flush()
    for st in all_states.values():
        save_state(st)
    if STOP_FILE.exists():
        STOP_FILE.unlink(missing_ok=True)
    elapsed = time.time() - t_start
    if check_stop():
        log(f"Stopped cleanly after {elapsed/60:.1f} min. "
            f"Total: {total_done} pages fetched, {total_errs} errors. Re-run to resume.")
    else:
        log(f"crawl finished after {elapsed/60:.1f} min. Total: {total_done} pages, {total_errs} errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())