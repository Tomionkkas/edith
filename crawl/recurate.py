import importlib.util, sys, time
from pathlib import Path
s = importlib.util.spec_from_file_location("cu", str(Path(__file__).resolve().parent / "curate.py"))
cu = importlib.util.module_from_spec(s); s.loader.exec_module(cu)
# Fandom phases only. wiki_marvel.jsonl is Wikipedia prose, not Marvel Database
# infoboxes, so it needs its own cleaning path (section trimming) -- not this one.
PHASES = ["characters", "comics", "teams", "events", "story_arcs",
          "items", "locations",
          "mw_character_backstop", "mw_marvel_backstop", "patch"]
t0 = time.time()
for p in PHASES:
    cu.run_phase(p)
    sys.stdout.flush()
# One wiki page can be filed under two categories, so two phases curate it.
# run_phase() cannot see that; this pass can, and it runs once every phase
# has had its say.
dropped = cu.dedupe_pages()
print(f"deduped {dropped} page(s) curated more than once")
print(f"\nre-curated all Fandom phases in {(time.time()-t0)/60:.1f} min")
