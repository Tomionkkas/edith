"""One-shot: seed raw/state/characters.json from the 500 existing test records
so the production crawl does not re-fetch pages already in characters.jsonl."""
import json
from pathlib import Path

raw = Path(__file__).resolve().parent.parent / "raw"
state_dir = raw / "state"
state_dir.mkdir(exist_ok=True)
p = raw / "characters.jsonl"

files = {}
titles = []
for line in p.open(encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    files[r.get("file")] = files.get(r.get("file"), 0) + 1
    if r.get("title"):
        titles.append(r["title"])

print("file fields in characters.jsonl:", files)
print("total records:", len(titles), "unique titles:", len(set(titles)))

st = state_dir / "characters.json"
d = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {"name": "characters", "done_titles": [], "errors": []}
have = set(d["done_titles"])
new = [t for t in titles if t not in have]
d["done_titles"].extend(new)
st.write_text(json.dumps(d), encoding="utf-8")
print("seeded:", len(new), "| now total done:", len(d["done_titles"]))