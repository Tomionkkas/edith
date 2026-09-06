"""Test the issue curation path on real issues fetched live from Fandom.

Fetches a handful of known issues (classic + modern), runs curate_issue,
prints stats and samples. Does NOT touch crawl state or raw/ files.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("curate", Path(__file__).resolve().parent / "curate.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
API = "https://marvel.fandom.com/api.php"

TEST_TITLES = [
    "Amazing Spider-Man Vol 1 300",      # Venom debut
    "X-Men Vol 1 1",                      # classic first issue
    "Fantastic Four Vol 1 1",             # first comic
    "The Avengers Vol 1 1",               # classic
    "Amazing Spider-Man Vol 1 299",       # symbiote arrival
    "Ultimate Spider-Man Vol 1 1",        # modern run
    "Doomsday Vol 1 1",                   # crossover
    "Avengers vs. X-Men Vol 1 1",         # modern crossover
]


def fetch(title: str):
    qs = urllib.parse.urlencode({
        "action": "query", "titles": title, "prop": "revisions",
        "rvprop": "content", "format": "json",
    })
    req = urllib.request.Request(f"{API}?{qs}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    for page in data.get("query", {}).get("pages", {}).values():
        if "missing" in page or "invalid" in page:
            return None
        revs = page.get("revisions") or [{}]
        return revs[0].get("*")
    return None


def main():
    out = []
    for i, t in enumerate(TEST_TITLES):
        wt = fetch(t)
        if wt is None:
            print(f"MISSING: {t}")
            continue
        rec = {"title": t, "wikitext": wt}
        text = cu.curate_issue(rec)
        n_chars = len(text)
        n_syn = text.count("Synopsis") if text else 0
        has_writers = "written by" in text
        print(f"[{i+1}/{len(TEST_TITLES)}] {t}: raw={len(wt)} clean={n_chars} "
              f"synopsis={n_syn>0} writers={has_writers}")
        out.append({"title": t, "clean": text, "raw_chars": len(wt)})
        if i < 2:
            print("=" * 70)
            print(text[:1500])
            print("=" * 70)
        time.sleep(1.2)

    ok = [o for o in out if o["clean"]]
    if ok:
        total = sum(len(o["clean"]) for o in ok)
        sizes = sorted(len(o["clean"]) for o in ok)
        print(f"\nSUMMARY: {len(ok)}/{len(TEST_TITLES)} produced text, "
              f"total={total} chars, median={sizes[len(sizes)//2]}")
    # save samples for review
    sd = Path(__file__).resolve().parent.parent / "curated" / "samples"
    sd.mkdir(parents=True, exist_ok=True)
    for o in out[:4]:
        safe = o["title"].replace(" ", "_").replace("/", "-")
        (sd / f"issue_{safe}.txt").write_text(o["clean"], encoding="utf-8")
    print(f"samples saved to {sd}")


if __name__ == "__main__":
    main()