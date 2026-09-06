#!/usr/bin/env python3
"""Honest accuracy at scale: ask the corpus about itself.

Hand-picked test cases measure the cases you thought of. This takes a random
sample of records, asks a question whose answer is written in that very
record, and checks the system comes back to it. Nothing is curated, so
nothing is flattered - a wrong answer here is a wrong answer a user would get.

Four probes, each a different way a real person asks:

    headline      "who created Iron Man"           the name on the page
    real name     "who is Anthony Edward Stark"    the name in the field
    profile       "tell me about Iron Man"         no field named
    field         "what powers does Iron Man have" a specific field

    py infer/evaluate.py --n 400
    py infer/evaluate.py --n 400 --show 40      # print the failures
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def answerable(index, resolve, doc_id: int) -> bool:
    """A record a user could reasonably expect an answer from."""
    record = index.text(doc_id)
    return (resolve.field_count(record) >= resolve.MIN_FIELDS
            and len(record) > 3000
            and resolve.is_main_continuity(record))


def sample(index, resolve, names, n: int, seed: int) -> list:
    """Records whose name belongs to nobody else.

    Every probe here is built from the record's own headline, so a ranking
    that prefers exact matches scores higher by construction - the harness
    would recommend the change it is biased towards. Where several records
    share a name the "right" one is a judgement (people asking about the Hood
    mean Parker Robbins, not the 1,986-char Golden Age stub that owns the bare
    name), and judgement belongs in the hand-written flagship set, not here.
    So ambiguous names are excluded and this measures one thing only: given a
    name that means exactly one record, do you return that record.
    """
    rnd = random.Random(seed)
    ids = list(range(len(index.docs)))
    rnd.shuffle(ids)
    out = []
    for doc_id in ids:
        if not answerable(index, resolve, doc_id):
            continue
        key = resolve.norm(bare(index.headlines[doc_id], resolve))
        if len(names.get(key, ())) != 1:
            continue
        out.append(doc_id)
        if len(out) >= n:
            break
    return out


def bare(headline: str, resolve) -> str:
    return resolve.REALITY_SUFFIX.sub("", headline).strip()


def probes(index, resolve, doc_id: int):
    """(label, question) pairs whose right answer is this record."""
    record = index.text(doc_id)
    name = bare(index.headlines[doc_id], resolve)
    out = [("headline", f"who created {name}"),
           ("profile", f"tell me about {name}"),
           ("field", f"what powers does {name} have")]
    for line in record.split("\n")[1:12]:
        if line.startswith("Full name: "):
            full = line[11:].strip().strip('"')
            if full and full.lower() != name.lower() and len(full) < 40:
                out.append(("real name", f"who is {full}"))
            break
    return out


EXACT = 2        # tier 2 is "the name IS the query"

RANKINGS = {
    "main, primary, size, tier (the original)":
        lambda m, p, s, t, d, n=False: (m, p, s, t, -d),
    "main, primary, tier, size":
        lambda m, p, s, t, d, n=False: (m, p, t, s, -d),
    "main, tier, primary, size":
        lambda m, p, s, t, d, n=False: (m, t, p, s, -d),
    "main, exact, primary, size, tier":
        lambda m, p, s, t, d, n=False: (m, t == EXACT, p, s, t, -d),
    "main, primary, exact, size, tier":
        lambda m, p, s, t, d, n=False: (m, p, t == EXACT, s, t, -d),
    "exact, main, primary, size, tier":
        lambda m, p, s, t, d, n=False: (t == EXACT, m, p, s, t, -d),
    # Size CLASS first, exactness within it. A 24 KB article outranks a 2 KB
    # stub whatever the stub is called, but among records of the same weight
    # the one the query actually names wins.
    "main, size class, exact, primary, size":
        lambda m, p, s, t, d, n=False: (m, decade(s), t == EXACT, p, s, t, -d),
    "main, primary, size class, exact, size":
        lambda m, p, s, t, d, n=False: (m, p, decade(s), t == EXACT, s, t, -d),
    "main, size class, primary, exact, size":
        lambda m, p, s, t, d, n=False: (m, decade(s), p, t == EXACT, s, t, -d),
    # Exact first, then SIZE rather than primary-ness. Both Hoods own the bare
    # name exactly - one as a headline, one as an alias - so what separates
    # them is which is the real article.
    "main, exact, size, primary, tier":
        lambda m, p, s, t, d, n=False: (m, t == EXACT, s, p, t, -d),
    "main, exact, size, tier, primary":
        lambda m, p, s, t, d, n=False: (m, t == EXACT, s, t, p, -d),
    # The mechanism, rather than a proxy for it. Tier 1 is "this name is only
    # a FRAGMENT of what you asked" - "War" inside "War Fist" - and that is the
    # match that ruins the long tail. Tier 0 ("thor" inside "thor odinson") is
    # the one the flagships need. So demote tier 1 specifically and leave the
    # rest of the key alone.
    "main, not-fragment, primary, size, tier":
        lambda m, p, s, t, d, n=False: (m, t != 1, p, s, t, -d),
    "main, primary, not-fragment, size, tier":
        lambda m, p, s, t, d, n=False: (m, p, t != 1, s, t, -d),
    "not-fragment, main, primary, size (no oracle)":
        lambda m, p, s, t, d, n=False: (t != 1, m, p, s, t, -d),
    # The notability oracle. `m or n` rather than a slot of its own: the key
    # must stay positionally uniform or a famous candidate's primary-ness gets
    # compared against another candidate's continuity. Fame reaching the key
    # at all is gated in build() by NOTABLE_SIZE_RATIO - measured without that
    # gate, this ordering scored 34/40, losing every flagship to an alternate
    # continuity.
    "not-fragment, main-or-famous, primary, size (now)":
        lambda m, p, s, t, d, n=False: (t != 1, m or n, p, s, t, -d),
    "not-fragment, famous, main, primary, size":
        lambda m, p, s, t, d, n=False: (t != 1, n, m, p, s, t, -d),
    "not-fragment, main, famous, primary, size":
        lambda m, p, s, t, d, n=False: (t != 1, m, n, p, s, t, -d),
    # Miles Morales is Earth-1610 and Eddie Brock's page is not the one whose
    # Full name matches exactly, so main-continuity-before-size handed both to
    # obscure Earth-616 records that happen to own the name.
    "not-fragment, primary, size, main":
        lambda m, p, s, t, d, n=False: (t != 1, p, s, m, t, -d),
    "not-fragment, size, main, primary":
        lambda m, p, s, t, d, n=False: (t != 1, s, m, p, t, -d),
    "not-fragment, main, size, primary":
        lambda m, p, s, t, d, n=False: (t != 1, m, s, p, t, -d),
}


def decade(size: int) -> int:
    """Order of magnitude: a stub, an article, or a flagship."""
    return len(str(max(size, 1)))



# The other half of the measurement. The round-trip above only uses names that
# mean exactly one record; these are the ones where several records share a
# name and the answer is a judgement about who people mean. A ranking has to
# satisfy both - optimising either alone produced a worse terminal.
#
# KEYED ON THE `Page:` TITLE, NOT THE HEADLINE (2026-09-04). This list used to
# hold headline strings and `flagship_score` compared `index.headlines[got]`,
# which cannot tell two records apart: the headline `Doctor Doom` belongs to
# five doppelgangers, clones and LMDs and NEVER to Victor von Doom, who is
# headlined `Emperor Doom` because curation takes the story-current
# CurrentAlias as the display name. So the old list scored PASS on a clone for
# months, and scored FAIL the moment 4.10 fixed it. `Beast` did the same for
# Krahllak over Henry McCoy, and `Power Man` for Erik Josten over Luke Cage.
# 4.9 re-keyed run_cases onto `Page:` for exactly this reason; this was the
# last place still comparing headlines.
#
# Every entry below was adjudicated by hand against the record the question
# should return, not copied from what the engine happened to answer.
FLAGSHIPS = [
    ('thor', 'Thor Odinson (Earth-616)'),
    ('who created thor', 'Thor Odinson (Earth-616)'),
    ('tell me about thor in detail', 'Thor Odinson (Earth-616)'),
    ('the hood', 'Parker Robbins (Earth-616)'),
    ('who created the hood', 'Parker Robbins (Earth-616)'),
    ('who is the Hood, explain in detail his powers', 'Parker Robbins (Earth-616)'),
    ('what powers does the hood have', 'Parker Robbins (Earth-616)'),
    ('venom', 'Venom (Symbiote) (Earth-616)'),
    ('who created venom', 'Venom (Symbiote) (Earth-616)'),
    ('spider-man', 'Peter Parker (Earth-616)'),
    ('who created spider-man', 'Peter Parker (Earth-616)'),
    ('wolverine', 'James Howlett (Earth-616)'),
    ('storm', 'Ororo Munroe (Earth-616)'),
    ('what powers does storm have', 'Ororo Munroe (Earth-616)'),
    ('captain america', 'Steven Rogers (Earth-616)'),
    ('moon knight', 'Marc Spector (Earth-616)'),
    ('who created moon knight', 'Marc Spector (Earth-616)'),
    ('iron man', 'Anthony Stark (Earth-616)'),
    ('doctor doom', 'Victor von Doom (Earth-616)'),
    ('who created doctor doom', 'Victor von Doom (Earth-616)'),
    ('hulk', 'Bruce Banner (Earth-616)'),
    ('black panther', "T'Challa (Earth-616)"),
    ('daredevil', 'Matthew Murdock (Earth-616)'),
    ('magneto', 'Max Eisenhardt (Earth-616)'),
    ('thanos', 'Thanos (Earth-616)'),
    ('deadpool', 'Wade Wilson (Earth-616)'),
    ('the thing', 'Benjamin Grimm (Earth-616)'),
    ('who is power man', 'Lucas Cage (Earth-616)'),
    ('rogue', 'Anna Marie LeBeau (Earth-616)'),
    ('silver surfer', 'Norrin Radd (Earth-616)'),
    ('scarlet witch', 'Wanda Maximoff (Earth-616)'),
    ('groot', 'Groot (Earth-616)'),
    ('who created loki', 'Loki Laufeyson (Earth-616)'),
    ('cyclops', 'Scott Summers (Earth-616)'),
    ('jean grey', 'Jean Grey (Earth-616)'),
    ('gambit', 'Remy LeBeau (Earth-616)'),
    ('vision', 'Vision (Earth-616)'),
    # Scott Lang, not Henry Pym: the corpus HEADLINES Scott Lang "Ant-Man",
    # which is its own statement that he IS the name, while Hank Pym carries
    # it as one codename among several. 4.10 moved this to Hank Pym on size
    # alone and 4.11 moved it back on identity; the second is the principled
    # one. Decided by the user 2026-09-05. Both men are Ant-Man - this is a
    # judgement recorded, not a defect fixed.
    ('ant-man', 'Scott Lang (Earth-616)'),
    ('nightcrawler', 'Kurt Wagner (Earth-616)'),
    ('beast', 'Henry McCoy (Earth-616)'),
]


def flagship_score(index, engine, facts, resolve):
    """Score on record IDENTITY, not on the headline string.

    `Page:` is what makes two records distinguishable. Comparing headlines
    scored a Doctor Doom clone as a pass and Victor von Doom as a failure -
    see the note above FLAGSHIPS. Wikipedia prose and comic issues carry no
    `Page:` line, so the headline is the best identity they have, which is
    the same fallback `terminal.emit_trace` uses.
    """
    hits = []
    for question, want in FLAGSHIPS:
        got = engine.resolved_doc(question, index, facts, resolve)
        page = None
        if got is not None:
            page = resolve.page_title(index.text(got)) or index.headlines[got]
        hits.append((page == want, question, want, page))
    return hits



# ---------------------------------------------------------------- by kind

CHARACTER_MARKS = ("Species / origin", "Identity status", "Gender")


def schema(record: str) -> set:
    """Field labels a record carries, stopping where the prose starts."""
    out = set()
    for line in record.split("\n")[1:]:
        if line.startswith(("History:", "Synopsis")):
            break
        label, sep, value = line.partition(": ")
        if sep and value.strip() and len(label) < 30 and " " not in label[:1]:
            out.add(label.strip())
    return out


def kind_of(index, doc_id: int) -> str:
    """What sort of thing this record is about.

    The corpus does not label its records, so this reads the shape: a
    character carries Gender or Species; an issue announces itself in its own
    headline; everything else with a schema is an event, a team, a location or
    an object; and 38% of the corpus carries no schema at all, which is what
    makes it unreachable by name.
    """
    head = index.headlines[doc_id]
    if " Vol " in head and " is a Marvel comic" in head:
        return "issue"
    marks = schema(index.text(doc_id))
    if marks & set(CHARACTER_MARKS):
        return "character"
    if len(marks) >= 3:
        return "event/team/other"
    return "no schema"


KIND_PROBES = {
    "character": ("who created {n}", "tell me about {n}",
                  "what powers does {n} have", "who is {n}"),
    "event/team/other": ("what is {n}", "tell me about {n}",
                         "who created {n}", "when did {n} first appear"),
    "issue": ("what is {n}", "tell me about {n}", "who wrote {n}"),
    "no schema": ("what is {n}", "tell me about {n}"),
}


def by_kind(index, resolve, engine, facts, per_kind: int, seed: int) -> None:
    """How well each sort of question is answered, sampled at random."""
    rnd = random.Random(seed)
    ids = list(range(len(index.docs)))
    rnd.shuffle(ids)
    buckets = {k: [] for k in KIND_PROBES}
    for doc_id in ids:
        k = kind_of(index, doc_id)
        if len(buckets[k]) < per_kind and len(index.text(doc_id)) > 600:
            buckets[k].append(doc_id)
        if all(len(v) >= per_kind for v in buckets.values()):
            break

    print(f"{'kind':<20}{'sampled':>8}{'to ITSELF':>11}{'owns the name':>15}   "
          f"example failure")
    for kind, docs in buckets.items():
        found = right = total = owner = 0
        example = ""
        for doc_id in docs:
            name = bare(index.headlines[doc_id], resolve)
            if " Vol " in name and " is a Marvel comic" in name:
                name = name.split(" is a Marvel comic")[0]
            for shape in KIND_PROBES[kind]:
                question = shape.format(n=name)
                got = engine.resolved_doc(question, index, facts, resolve)
                total += 1
                found += got is not None
                if got == doc_id:
                    right += 1
                    owner += 1
                else:
                    # A name shared by twelve records makes "the exact record
                    # I sampled" an unfair target: "what is Scarlet Witch"
                    # returning a different Scarlet Witch is not a failure.
                    if got is not None and owns(index, resolve, got, question):
                        owner += 1
                    if not example:
                        head = index.headlines[got] if got is not None else None
                        example = f"{question[:42]!r} -> {str(head)[:30]!r}"
        print(f"{kind:<20}{len(docs):>8}{right / max(total, 1):>10.0%}"
              f"{owner / max(total, 1):>15.0%}   {example}")


def variant_check(index, resolve, engine, facts, n: int, seed: int) -> None:
    """Does a "what are X's variants" question list the records it should?"""
    rnd = random.Random(seed)
    counts = {}
    for doc_id in range(len(index.docs)):
        key = resolve.norm(bare(index.headlines[doc_id], resolve))
        if key:
            counts.setdefault(key, []).append(doc_id)
    shared = [ids for ids in counts.values() if len(ids) >= 3]
    rnd.shuffle(shared)
    listed = named = 0
    for ids in shared[:n]:
        name = bare(index.headlines[ids[0]], resolve)
        doc_id = engine.resolved_doc(f"what are the variants of {name}",
                                     index, facts, resolve)
        answer = (facts.render_variants(index, resolve, doc_id)
                  if doc_id is not None else "")
        if answer and "records" in answer:
            listed += 1
            if name.split()[0].lower() in answer.lower():
                named += 1
    print(f"\nvariants: {listed}/{min(n, len(shared))} names with 3+ records "
          f"answered with a LIST, {named} of them naming the right character")


def owns(index, resolve, doc_id: int, question: str) -> bool:
    """Does this record legitimately answer to the name in the question?"""
    key = resolve.query_key(question)
    if not key:
        return False
    primary, alias = resolve.names_of(index.headlines[doc_id],
                                      index.text(doc_id))
    return any(set(key) <= set(name) or set(name) <= set(key)
               for name in primary | alias if name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="records to sample")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=12, help="failures to print")
    ap.add_argument("--kinds", action="store_true",
                    help="score events, issues, teams and variants too")
    ap.add_argument("--rank", action="store_true",
                    help="sweep whole ranking keys instead of scoring one")
    ap.add_argument("--picker", action="store_true",
                    help="sweep how often each confidence rule would ask")
    args = ap.parse_args()

    theme = _load("theme", "infer/theme.py")
    theme.use_utf8()
    search = _load("search", "retrieve/search.py")
    resolve = _load("resolve", "retrieve/resolve.py")
    facts = _load("facts", "infer/facts.py")
    engine = _load("engine", "infer/engine.py")
    index = search.Index.load()
    engine._NAMES = resolve.load() or {}
    resolve.token_index(engine._NAMES)

    if args.kinds:
        by_kind(index, resolve, engine, facts, args.n, args.seed)
        variant_check(index, resolve, engine, facts, args.n, args.seed)
        return 0

    docs = sample(index, resolve, engine._NAMES, args.n, args.seed)
    print(f"{len(docs):,} records sampled from {len(index.docs):,}\n")

    if args.rank:
        # Whole orderings, measured against the same probes. Hand-picked
        # cases chose the current key and were wrong about it; a thousand
        # probes is the only argument worth making here.
        for label, key in RANKINGS.items():
            # The lambdas below were written when this slot held a boolean
            # `primary`. Phase 4.11 made it a 0/1/2 provenance, and production's
            # rank() collapses it with `provenance > 0`. Passing the raw int here
            # made every row a different key from its label - the row labelled
            # "(now)" stopped being production and reproduced 4.11's first failed
            # attempt, 39/40 with `beast` lost. Coerce once, here.
            resolve.rank = lambda m, p, s, t, d, n=False, k=key: k(m, p > 0, s, t, d, n)
            hits = tot = 0
            for doc_id in docs:
                want = index.headlines[doc_id]
                for _lab, question in probes(index, resolve, doc_id):
                    got = engine.resolved_doc(question, index, facts,
                                              resolve)
                    tot += 1
                    hits += (got is not None
                             and index.headlines[got] == want)
            flags = flagship_score(index, engine, facts, resolve)
            fw = sum(1 for ok, *_ in flags if ok)
            print(f"  {label:<38}"
                  f"round-trip {hits / max(tot, 1):6.1%}"
                  f"   flagships {fw:>2}/{len(flags)}")
        return 0

    if args.picker:
        # Picker RATE is the number that decides this. A threshold that
        # passes every answer case can still ask on half of all questions,
        # and nothing in the suite would say so - which is exactly how
        # NOTABLE_SIZE_RATIO got in.
        #
        # Both signals are measured on facts.entity_text(question) - the
        # SAME intent-stripped string engine.resolved_doc() actually
        # resolves (engine.py:136), not the raw probe question. FINDING
        # 2026-09-02: sweeping the raw question measures a string production
        # never resolves. "what powers does moon knight have" keeps the
        # token "power" - deliberately not in QUERY_NOISE, since it's part
        # of real names like Power Man - which demotes the record's own name
        # to a tier-1 fragment match and invites closer competition from
        # unrelated candidates that would never reach it stripped. Measured:
        # moon knight confidence 5.9 raw vs. 38.7 stripped, while genuinely
        # contested spider-man does not move (1.9 either way). rivals()
        # mirrors resolve()'s candidate loop identically, so it has the same
        # mismatch and gets the same fix.
        # known_words=index.postings, the CORPUS vocabulary - the same one
        # infer/engine.py's wants_choice() passes confidence() in production
        # (see resolve.confidence's docstring; FINDING 2026-09-02). Without
        # it this sweep measures a guard production no longer runs: judged
        # against the NAME vocabulary instead, a real corpus word that names
        # nobody wrongly evicts tier-1 candidates and reads falsely confident.
        probes_asked = []
        for doc_id in docs:
            for label, question in probes(index, resolve, doc_id):
                probes_asked.append((label, facts.entity_text(question)))
        total = len(probes_asked)
        rivals_vals = [resolve.rivals(engine._NAMES, p, index.postings)
                       for _l, p in probes_asked]
        conf_vals = [resolve.confidence(engine._NAMES, p, index.postings)
                     for _l, p in probes_asked]

        print(f"  {'rule':<28}{'asks':>8}{'rate':>9}")
        for threshold in (2, 3, 5, 8):
            asked = sum(1 for v in rivals_vals if v >= threshold)
            print(f"  {'rivals >= ' + str(threshold):<28}{asked:>8}"
                  f"{asked / max(total, 1):>8.1%}")
        # The count signal saturates (build() caps every name's entries at
        # 8) and its uncapped form still cannot separate "answer" from
        # "offer" - emma frost, galactus and taskmaster all have dozens to
        # hundreds of same-named rivals and are not ambiguous to a reader.
        # confidence() is the spec's other candidate: the size margin
        # between the best record and the runner-up, measured 2026-09-02 to
        # separate the two hand-picked groups cleanly (see resolve.py).
        for ratio in (1.5, 2, 3, 5, 10):
            asked = sum(1 for v in conf_vals if v < ratio)
            print(f"  {'confidence < ' + str(ratio):<28}{asked:>8}"
                  f"{asked / max(total, 1):>8.1%}")

        # The pooled rate hides which probe SHAPE drives it. Break
        # confidence down per probe label - headline, real name, profile,
        # field - so a threshold choice can see whether one shape (field,
        # per the finding above) is doing all the asking.
        labels = sorted({label for label, _p in probes_asked})
        by_label = {lab: [] for lab in labels}
        for (label, _p), v in zip(probes_asked, conf_vals):
            by_label[label].append(v)
        print()
        print("  confidence < R, by probe label")
        for ratio in (1.5, 2, 3, 5, 10):
            print(f"    confidence < {ratio}")
            for label in labels:
                vals = by_label[label]
                asked = sum(1 for v in vals if v < ratio)
                print(f"      {label:<20}{asked:>5}/{len(vals):<6}"
                      f"{asked / max(len(vals), 1):>7.1%}")

        # FINDING 2026-09-02 (Important): the corpus-wide rate above is NOT
        # what chose CONFIDENCE_TO_ASK - ROADMAP.md's Phase 4.7 entry and the
        # comment above CONFIDENCE_TO_ASK both quote a hand-typed table
        # nothing here reproduced, so a stranger could not re-derive or
        # challenge it. This block IS that table, generated by calling
        # engine.wants_choice() itself - the exact function production
        # calls, never a re-implementation of its decision - against the
        # sets that represent real questions: FLAGSHIPS (40 judgement calls)
        # and infer/answer_cases.py's hand-written CASES (must and must not
        # offer a picker).
        answer_cases = _load("answer_cases", "infer/answer_cases.py")

        def resolve_case(turns):
            """Thread `previous` across every turn but the last, exactly as
            the terminal does, and return (doc_id, last_question) for the
            turn wants_choice() judges."""
            previous = None
            for q in turns[:-1]:
                previous = engine.resolved_doc(q, index, facts, resolve,
                                               previous)
            last = turns[-1]
            return engine.resolved_doc(last, index, facts, resolve,
                                       previous), last

        flag_docs = [(q, engine.resolved_doc(q, index, facts, resolve))
                     for q, _want in FLAGSHIPS]
        case_docs = []                    # (last_question, doc_id, must_ask)
        for case in answer_cases.CASES:
            doc_id, last = resolve_case(case["ask"])
            case_docs.append((last, doc_id, case.get("expect") == "picker"))
        not_ask_total = sum(1 for *_ , must_ask in case_docs if not must_ask)
        must_ask_total = sum(1 for *_, must_ask in case_docs if must_ask)

        print()
        print("  representative-set sweep - the table CONFIDENCE_TO_ASK's "
              "value actually rests on")
        print(f"  ({len(flag_docs)} flagships, {not_ask_total} answer-cases "
              f"that must NOT offer, {must_ask_total} that must)")
        print(f"  {'threshold':<14}{'flagships':>14}{'must-not-ask':>17}"
              f"{'must-ask':>13}")
        saved_threshold = engine.CONFIDENCE_TO_ASK
        try:
            for threshold in (1.5, 2, 3, 5, 10):
                engine.CONFIDENCE_TO_ASK = threshold
                flag_asks = sum(
                    engine.wants_choice(q, index, facts, resolve, d)
                    for q, d in flag_docs)
                not_ask_asks = sum(
                    engine.wants_choice(last, index, facts, resolve, d)
                    for last, d, must_ask in case_docs if not must_ask)
                must_ask_hits = sum(
                    engine.wants_choice(last, index, facts, resolve, d)
                    for last, d, must_ask in case_docs if must_ask)
                print(f"  < {threshold:<12}"
                      f"{flag_asks:>3}/{len(flag_docs):<3}"
                      f"{flag_asks / len(flag_docs):>7.0%}"
                      f"{not_ask_asks:>6}/{not_ask_total:<3}"
                      f"{not_ask_asks / max(not_ask_total, 1):>7.0%}"
                      f"{must_ask_hits:>6}/{must_ask_total}")
        finally:
            engine.CONFIDENCE_TO_ASK = saved_threshold
        return 0

    totals, wins, owners = Counter(), Counter(), Counter()
    failures = []
    for doc_id in docs:
        want = index.headlines[doc_id]
        for label, question in probes(index, resolve, doc_id):
            got = engine.resolved_doc(question, index, facts, resolve)
            totals[label] += 1
            head = index.headlines[got] if got is not None else None
            if head == want:
                wins[label] += 1
                owners[label] += 1
            else:
                # "who is David Haller" returns Legion, and the harness wanted
                # an Earth-6160 variant. Legion is the right answer to that
                # question. So a second, fairer score: does the record we
                # returned actually own the name that was asked about?
                if got is not None and owns(index, resolve, got, question):
                    owners[label] += 1
                failures.append((label, question, want, head))

    print(f"{'probe':<12}{'exact record':>14}{'owns the name':>16}")
    for label in ("headline", "real name", "profile", "field"):
        if totals[label]:
            print(f"{label:<12}{wins[label] / totals[label]:>13.1%}"
                  f"{owners[label] / totals[label]:>16.1%}")
    total, win, own = (sum(totals.values()), sum(wins.values()),
                       sum(owners.values()))
    print(f"{'ALL':<12}{win / max(total, 1):>13.1%}{own / max(total, 1):>16.1%}"
          f"   ({total:,} probes)")

    flags = flagship_score(index, engine, facts, resolve)
    fw = sum(1 for ok, *_ in flags if ok)
    print(f"\n{'flagships':<12}{fw:>5}/{len(flags):<4}{'':>4}"
          f"{fw / len(flags):6.1%}   (ambiguous names, judgement calls)")
    for ok, question, want, head in flags:
        if not ok:
            print(f"  {question!r} -> {head!r}, wanted {want!r}")

    if failures and args.show:
        print(f"\n{len(failures)} failures; first {min(args.show, len(failures))}:")
        for label, question, want, head in failures[:args.show]:
            print(f"  [{label}] {question!r}")
            print(f"       want {want!r}")
            print(f"       got  {head!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
