"""The BM25 rewrite, checked against a transcription of what it replaced.

`search()` was rewritten from dict-of-dicts postings and a per-document
Python loop into flat arrays and numpy vector ops. That is a change with no
intended effect on any answer, which is the most dangerous kind: a ranking
that shifts by a point looks like nothing until the wrong Spider-Man comes
back. So the old algorithm is kept here, literally, and the new one is
required to agree with it exactly - scores included, not just ordering.

Run: py -m pytest retrieve/test_search_equivalence.py
"""
import importlib.util
import math
import random
import unittest
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("search", HERE / "search.py")
se = importlib.util.module_from_spec(spec)
spec.loader.exec_module(se)


def reference_search(records, query, k=3):
    """search() as it was before the rewrite: dicts, and a loop per document.

    Self-contained on purpose - it builds its own index from the records, so
    it cannot accidentally share a precomputed table with the code under test
    and agree with it for the wrong reason.
    """
    docs = [r.strip() for r in records if r.strip()]
    headlines = [d.split("\n", 1)[0] for d in docs]
    is_entity = [any(ln.startswith("Reality:") for ln in d.split("\n"))
                 for d in docs]
    postings, head_postings, lengths = defaultdict(dict), defaultdict(set), []
    for i, d in enumerate(docs):
        toks = se.tokenize(d)
        lengths.append(len(toks))
        for term, tf in Counter(toks).items():
            postings[term][i] = tf
        for term in set(se.tokenize(headlines[i])):
            head_postings[term].add(i)
    avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0

    terms = se.tokenize(query)
    if not terms or not docs:
        return []
    n = len(docs)
    scores = defaultdict(float)
    for term in set(terms):
        posting = postings.get(term)
        if not posting:
            continue
        df = len(posting)
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        heads = head_postings.get(term, ())
        for doc_id, tf in posting.items():
            norm = 1.0 - se.B + se.B * (lengths[doc_id] / max(avg_len, 1e-9))
            scores[doc_id] += idf * (tf * (se.K1 + 1.0)) / (tf + se.K1 * norm)
            if doc_id in heads:
                scores[doc_id] += idf * se.HEADLINE_BOOST
    entity = {se.singular(t) for t in se.strip_question(tuple(terms))}
    if len(entity) > 1:
        cutoff = n * se.ENTITY_DF_RATIO
        rare = {t for t in entity if len(postings.get(t, ())) <= cutoff}
        entity = rare or entity
    asked_reality = any(se.REALITY_RE.fullmatch(t) for t in terms)
    for doc_id in list(scores):
        head_toks = {se.singular(t) for t in se.tokenize(headlines[doc_id])}
        if entity and entity <= head_toks:
            scores[doc_id] += se.FULL_HEAD_BONUS
            scores[doc_id] -= se.BREVITY_PENALTY * max(0, len(head_toks) - len(entity))
            scores[doc_id] += se.NOTABILITY_WEIGHT * math.log1p(lengths[doc_id])
        if not asked_reality and "(" not in headlines[doc_id]:
            scores[doc_id] += se.MAIN_CONTINUITY_BONUS
        if is_entity[doc_id]:
            scores[doc_id] += se.ENTITY_PAGE_BONUS
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


NAMES = ("Spider-Man Storm Venom Thor Iron Man Doctor Doom Power Man "
         "Moon Knight Black Panther The Thing Captain America").split()
BODY = ("radioactive spider bite New York symbiote asgard mutant weather "
        "goddess doom castle latveria wall crawling powers strength").split()


def fuzz_records(seed, n=60):
    """Records that collide the way the corpus does - shared names, shared
    words, sizes spanning three orders of magnitude, and duplicate headlines
    (18 records are headlined exactly "Spider-Man")."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        head = " ".join(rnd.sample(NAMES, rnd.randint(1, 2)))
        if rnd.random() < 0.35:
            head += f" (Earth-{rnd.choice((616, 1610, 199999))})"
        lines = [head]
        if rnd.random() < 0.8:
            lines.append(f"Reality: Earth-{rnd.choice((616, 1610))}")
        lines.append("Created by: " + " ".join(rnd.sample(BODY, 2)))
        if rnd.random() < 0.5:
            lines.append("Powers: " + " ".join(rnd.sample(BODY, 3)))
        lines.append("History:")
        lines.append(" ".join(rnd.choices(BODY + NAMES, k=rnd.randint(5, 900))))
        out.append("\n".join(lines))
    return out


def fuzz_queries(seed, n=60):
    rnd = random.Random(seed)
    scaffold = ("who created", "what powers does", "tell me about",
                "where is", "", "who is", "real name of")
    out = []
    for _ in range(n):
        body = " ".join(rnd.sample(NAMES + BODY, rnd.randint(1, 3)))
        out.append(f"{rnd.choice(scaffold)} {body}".strip())
    out += ["earth-616 storm", "who is", "the a of in", "s.h.i.e.l.d.",
            "spider-man", "storms", "nothing matches this query xyzzy"]
    return out


class ScoringEquivalence(unittest.TestCase):
    def test_scores_match_the_previous_implementation_exactly(self):
        for seed in range(4):
            records = fuzz_records(seed)
            index = se.Index.build(records)
            for q in fuzz_queries(seed):
                want = reference_search(records, q, k=5)
                got = index.search(q, k=5)
                self.assertEqual([d for d, _ in got], [d for d, _ in want],
                                 f"ranking differs for {q!r}")
                for (_, a), (_, b) in zip(got, want):
                    self.assertAlmostEqual(a, b, places=9, msg=q)

    def test_survives_a_save_load_round_trip(self):
        import tempfile
        records = fuzz_records(9)
        a = se.Index.build(records)
        p = Path(tempfile.mkdtemp()) / "ix.pkl"
        a.save(p)
        b = se.Index.load(p)
        for q in fuzz_queries(9, n=20):
            self.assertEqual(a.search(q, k=5), b.search(q, k=5), q)

    def test_an_index_from_an_older_layout_is_refused_not_misread(self):
        """A stale pickle must say so. Silently scoring against half a table
        is the failure mode that looks like a bad model."""
        import pickle, tempfile
        p = Path(tempfile.mkdtemp()) / "old.pkl"
        with p.open("wb") as fh:
            pickle.dump({"docs": ["Spider-Man\nReality: Earth-616"],
                         "headlines": ["Spider-Man"]}, fh)
        with self.assertRaises(ValueError) as cm:
            se.Index.load(p)
        self.assertIn("--build", str(cm.exception))

    def test_empty_corpus_is_safe(self):
        self.assertEqual(se.Index.build([]).search("spider-man"), [])

    def test_document_frequency_is_reported_from_the_spans(self):
        index = se.Index.build(["Storm\nReality: Earth-616\nweather",
                                "Thor\nReality: Earth-616\nweather"])
        self.assertEqual(index.df("weather"), 2)
        self.assertEqual(index.df("storm"), 1)
        self.assertEqual(index.df("nonexistent"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
