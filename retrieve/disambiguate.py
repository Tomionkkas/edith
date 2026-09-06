"""Which Spider-Man did you mean?

The corpus is faithful to Marvel, and Marvel is ambiguous: 18 records are
headlined exactly "Spider-Man" (Peter Parker, Norman Osborn, Ai Apaec...) and
15 exactly "Venom" (the symbiote, Mary Jane Watson, Angelo Fortunato...). They
are different characters who used the same name, not duplicates.

Ranking already puts the flagship first, via page size and the Earth-616
convention. What was missing is telling the reader WHICH one they got, and
offering the others. That happens here, in the harness - deterministic, no
generated tokens, and so no chance of inventing a variant that does not exist.

A menu is only worth blocking on when there is no clear winner; otherwise the
flagship is answered and the alternatives are a footer.
"""
from __future__ import annotations

# Below this relative gap between the top two, no candidate is clearly the
# flagship and asking beats guessing.
NEWLINE = chr(10)
# Narrative sections; the fielded head stops here (mirrors context.py).
STOP_SECTIONS = ("History:", "Synopsis")
PRIORITY_LABELS = frozenset((
    "Full name", "Created by", "First appearance", "Reality", "Powers",
    "Abilities", "Species / origin", "Affiliation", "Occupation",
))
AMBIGUOUS_MARGIN = 0.015
MAX_CANDIDATES = 8

# Candidates sharing a headline whose scores are this close are effectively
# tied on relevance, and page size decides. Notability reaches the BM25 score
# as 1.5*log1p(size), which compressed Steve Rogers' 92,106-char page against
# Sam Wilson's 87,554 into 0.08 points - less than BM25 noise, so "captain
# america" answered as Sam Wilson. Among equals, the most written-about
# holder of a shared name is the one people mean.
FLAGSHIP_MARGIN = 0.05


def _field(record: str, label: str) -> str:
    prefix = label + ": "
    for line in record.split("\n"):
        if line.startswith(prefix):
            return line[len(prefix):].strip()
        if line.startswith(("History:", "Synopsis")):
            break
    return ""


def _field_count(record: str) -> int:
    """How many priority fields the record actually carries.

    A prose record (Wikipedia) has none, and cannot answer "who created X"
    however long it is.
    """
    n = 0
    for line in record.split(NEWLINE)[1:]:
        if line.startswith(STOP_SECTIONS):
            break
        label = line.split(": ", 1)[0] if ": " in line else ""
        if label in PRIORITY_LABELS:
            n += 1
    return n


def identity(record: str) -> tuple:
    """What makes two records the same entity.

    Headline alone is not enough - 18 records share "Spider-Man". Headline
    plus real name plus reality separates Peter Parker from Norman Osborn
    while still collapsing a record with itself.
    """
    head = record.split("\n")[0].strip()
    return (head, _field(record, "Full name"), _field(record, "Reality"))


def describe(record: str) -> str:
    """A one-line label a reader can choose between.

    Wikipedia records carry no fielded head, so they fall back to their first
    line of prose rather than rendering as an empty label.
    """
    bits = []
    full = _field(record, "Full name")
    if full and full.lower() not in ("inapplicable", "unknown"):
        bits.append(full)
    # Species is NOT a fallback for a missing real name: Deadpool's record has
    # no Full name and rendered as "Human mutate after being a De...", which
    # reads like a different character than the headline promises.
    reality = _field(record, "Reality")
    if reality:
        bits.append(reality)
    first = _field(record, "First appearance")
    if first:
        bits.append(first)
    if bits:
        return " \u00b7 ".join(bits)
    for line in record.split("\n")[1:]:
        line = line.strip()
        if len(line) > 30:
            return line[:90]
    return record.split("\n")[0].strip()


def candidates(index, query: str, k: int = MAX_CANDIDATES) -> list:
    """[(doc_id, headline, description, score)], best first, one per entity.

    Over-fetches because near-duplicate records of the same entity collapse,
    and a caller asking for 5 choices should get 5 distinct ones.
    """
    seen, out = set(), []
    for doc_id, score in index.search(query, k=k * 3):
        record = index.text(doc_id)
        key = identity(record)
        if key in seen:
            continue
        seen.add(key)
        out.append((doc_id, index.headlines[doc_id], describe(record), score))
        if len(out) >= k:
            break
    return promote_flagship(out, index)


def promote_flagship(cands, index, margin: float = FLAGSHIP_MARGIN):
    """Among near-tied candidates sharing a headline, the biggest page wins.

    Only reorders within a scoring tie, so a genuinely better match is never
    displaced by a merely longer article.
    """
    if len(cands) < 2:
        return cands
    top = cands[0][3]
    if top <= 0:
        return cands
    head = cands[0][1]
    tied = [c for c in cands
            if c[1] == head and (top - c[3]) / abs(top) < margin]
    if len(tied) < 2:
        return cands
    # Field richness first, page size only as a tie-break. Page size alone
    # promoted the Wikipedia article for "doctor doom": the biggest page by
    # far, and prose - no Created by, no Reality, nothing a factual answer can
    # be copied from. The model then filled the gaps from memory, which is the
    # exact failure retrieval exists to prevent.
    best = max(tied, key=lambda c: (_field_count(index.text(c[0])),
                                    len(index.text(c[0]))))
    if best is cands[0]:
        return cands
    rest = [c for c in cands if c is not best]
    return [best] + rest


def is_ambiguous(cands, margin: float = AMBIGUOUS_MARGIN) -> bool:
    """True when no candidate is clearly the flagship.

    Compares the top two scores relatively, not absolutely: BM25 scores scale
    with query length, so a fixed threshold would ask constantly on long
    questions and never on short ones.
    """
    if len(cands) < 2:
        return False
    top, second = cands[0][3], cands[1][3]
    if top <= 0:
        return True
    return (top - second) / abs(top) < margin


def alternatives_line(cands, limit: int = 4) -> str:
    """The footer: other entities sharing this name, most notable first."""
    others = cands[1:]
    if not others:
        return ""
    shown = [c[2].split(" \u00b7 ")[0] for c in others[:limit]]
    more = len(others) - len(shown)
    line = " \u00b7 ".join(shown)
    return line + (f" \u00b7 {more} more" if more > 0 else "")
