#!/usr/bin/env python3
"""BM25 retrieval over the curated Marvel corpus.

The model supplies fluency; this supplies truth. A 250M model cannot memorise
261 MB of facts -- that is lossy compression, not storage -- so facts are
looked up in the corpus and handed to the model as context instead.

The corpus is unusually easy to retrieve from because curation kept the field
schema (`Created by:`, `First appearance:`, `Reality:`) and put the continuity
in the headline, so an entity name plus a reality id pins down exactly one of
the several Spider-Men.

  py retrieve/search.py --build              # index curated/ -> retrieve/index.pkl
  py retrieve/search.py "who created Galactus"
"""
from __future__ import annotations
import argparse
import math
import pickle
import re
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
CURATED = BASE_DIR / "curated"
INDEX_PATH = Path(__file__).resolve().parent / "index.pkl"
SEPARATOR = "=" * 60

# One token per: dotted acronym (S.H.I.E.L.D.), hyphenated name or reality id
# (Spider-Man, Earth-616), or plain word. Splitting these would break exactly
# the lookups the corpus exists to answer.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)+\.?|[a-z0-9]+(?:-[a-z0-9]+)+|[a-z0-9]+")

K1 = 1.5            # BM25 term-frequency saturation
B = 0.75            # BM25 length normalisation
HEADLINE_BOOST = 3.0
MAX_CONTEXT_CHARS = 1800

# Tuned against live queries that returned the wrong entity: "who created
# Moon Knight" surfaced an obscure Venomverse variant, and "who created
# Spider-Man" surfaced "List of actors who have played Spider-Man".
FULL_HEAD_BONUS = 12.0        # every entity term appears in the headline
BREVITY_PENALTY = 2.4         # per surplus headline term, so the tightest wins
MAIN_CONTINUITY_BONUS = 5.0   # Earth-616 pages carry no suffix by convention
ENTITY_PAGE_BONUS = 6.0       # a page about a being, not about a publication
NOTABILITY_WEIGHT = 1.5       # log(page size): the canonical entity has the biggest page

# Carry no entity signal. Stripped only for headline matching -- BM25 still
# scores the full query, so "Created by" still helps rank the body.
QUESTION_WORDS = frozenset("""
who what when where which why how whose whom is are was were be been am do does did
the a an of in on at to for by from with and or as tell me about show give
create created creates make made makes name named call called
detail details explain describe please you your know more anything rundown
info information he she they him her his their them have has had can could
would should want need much many some any this that these those there here
""".split())

# NOT in the list: "power", "powers", "man", "woman", "girl", "boy". They look
# like filler and are not - Power Man, Iron Man, Spider-Woman. Stripping them
# would break the very lookups the corpus exists to answer. The df filter in
# search() handles them instead, without guessing.

# A name is rare and filler is common, so document frequency separates them
# without a hand-maintained list. Terms above this share of the corpus are
# dropped from the entity test.
ENTITY_DF_RATIO = 0.15


REALITY_RE = re.compile(r"earth-\d+")

# Typecode matching np.int32. "i" is 4 bytes everywhere CPython runs, but
# np.frombuffer would silently misread every posting if it were not.
INT32 = "i" if array("i").itemsize == 4 else "l"


def tokenize(text: str) -> list:
    """Lowercase tokens; compound names also yield their parts.

    `Spider-Man` -> ['spider-man', 'spider', 'man'] so a query of either form
    reaches the record.
    """
    if not text:
        return []
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        out.append(t)
        # "daredevils real name" must reach the Daredevil record; without this
        # it matched "The Daredevils", a different entity. Question words are
        # excluded: stripping "does" gives "doe", which is rare enough to look
        # like a name and matched a record called "Doe Eyes".
        if (len(t) > 3 and t.endswith("s") and t not in QUESTION_WORDS
                and not t.endswith(("ss", "us", "is"))):
            out.append(t[:-1])
        if "-" in t:
            out.extend(p for p in t.split("-") if p)
    return out


def singular(token: str) -> str:
    """Collapse a simple plural, for comparing a query against a headline.

    tokenize() emits both "daredevils" and "daredevil" so BM25 can match
    either. That broke the headline test, which asks whether every entity term
    appears in the headline: {daredevils, daredevil} is not a subset of
    {daredevil}, so the Daredevil record lost its bonus to "The Daredevils".
    Comparing singular forms on both sides fixes it.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def strip_question(tokens: tuple) -> tuple:
    """Drop interrogatives and filler so only the entity terms remain.

    'who created Galactus' -> ('galactus',). Falls back to the original tokens
    if stripping would leave nothing.
    """
    kept = tuple(t for t in tokens if t not in QUESTION_WORDS)
    return kept or tuple(tokens)


def entity_query(index, question: str) -> str:
    """The question reduced to terms rare enough to name an entity.

    BM25 scores every term, so wording shifts the ranking even when the entity
    is obvious: "what powers does storm have" found Ororo Munroe while "what is
    storm skilled in" found Of-Storm, because "skilled" is a real term that
    matches real records. A name is rare and question words are common, so
    document frequency separates them - the same test that made conversational
    phrasing work.

    Falls back to the original question when nothing is discriminative enough,
    since a query of no terms retrieves nothing at all.
    """
    terms = list(dict.fromkeys(strip_question(tuple(tokenize(question)))))
    if not terms:
        return question
    n = max(len(index.docs), 1)
    cutoff = n * ENTITY_DF_RATIO
    rare = [t for t in terms if index.df(t) <= cutoff]
    return " ".join(rare) if rare else " ".join(terms)



FORMAT = 2       # stored layout; load() refuses anything else


def _csr(per_term: dict, order: list):
    """{term: array("i")} -> (term -> (start, end), one flat int32 array).

    One array per term costs a Python object and a dict lookup per posting.
    Flattening them puts every posting in one buffer, which is what lets the
    scorer add a whole term's contribution as a single vector operation.
    """
    spans, parts, pos = {}, [], 0
    for term in order:
        a = np.frombuffer(per_term[term], dtype=np.int32)
        spans[term] = (pos, pos + len(a))
        pos += len(a)
        parts.append(a)
    flat = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
    return spans, flat


class Index:
    """In-memory inverted index with BM25 scoring and entity resolution.

    Postings are stored flat - one int32 array of doc ids, one of term
    frequencies, and a term -> (start, end) span map - not as a dict per term.
    A query took 353 ms in the dict form, and three quarters of that was
    re-tokenising 187,584 headlines to test each against the query. Headline
    tokens are precomputed at build time and scoring is vectorised, which is
    the whole of the difference; the arithmetic is unchanged, and
    test_search_equivalence.py holds it to the old implementation's scores.
    """

    def __init__(self):
        self.docs: list[str] = []
        self.headlines: list[str] = []
        self.is_entity = np.empty(0, dtype=bool)
        # term -> (start, end) into post_docs / post_tf
        self.postings: dict[str, tuple] = {}
        self.post_docs = np.empty(0, dtype=np.int32)
        self.post_tf = np.empty(0, dtype=np.int32)
        # the same, over headline tokens only
        self.head_postings: dict[str, tuple] = {}
        self.head_docs = np.empty(0, dtype=np.int32)
        # and over DE-PLURALISED headline tokens, which is what the entity
        # test compares against. Both are kept because they genuinely differ:
        # tokenize() leaves question words alone, so a headline containing
        # "does" yields the singular "doe" here and not there.
        self.head_singular: dict[str, tuple] = {}
        self.singular_docs = np.empty(0, dtype=np.int32)
        self.head_len = np.empty(0, dtype=np.int32)     # distinct singular tokens
        self.has_paren = np.empty(0, dtype=bool)        # "(" in the headline
        self.lengths = np.empty(0, dtype=np.int32)
        self.avg_len: float = 0.0
        self._derive()

    def _derive(self) -> None:
        """Tables that depend on K1/B/avg_len, so they are never stored.

        Baking the length normalisation into the index file would let a change
        to K1 or B silently disagree with a stale one.
        """
        length = self.lengths.astype(np.float64)
        self.k1_norm = K1 * (1.0 - B + B * (length / max(self.avg_len, 1e-9)))
        self.log1p_len = np.log1p(length)

    def __len__(self):
        return len(self.docs)

    def text(self, doc_id: int) -> str:
        return self.docs[doc_id]

    def df(self, term: str) -> int:
        """How many records contain `term`."""
        span = self.postings.get(term)
        return span[1] - span[0] if span else 0

    @classmethod
    def build(cls, records) -> "Index":
        ix = cls()
        docs, headlines, is_entity, lengths = [], [], [], []
        head_len, has_paren = [], []
        post_docs = defaultdict(lambda: array(INT32))
        post_tf = defaultdict(lambda: array(INT32))
        head_post = defaultdict(lambda: array(INT32))
        sing_post = defaultdict(lambda: array(INT32))
        for rec in records:
            rec = rec.strip()
            if not rec:
                continue
            doc_id = len(docs)
            docs.append(rec)
            head = rec.split("\n", 1)[0]
            headlines.append(head)
            has_paren.append("(" in head)
            # A Reality: line means the record is about a being or team, not
            # about a comic issue. Queries are nearly always about the former,
            # and issue titles ("Nick Fury: Director of S.H.I.E.L.D.") otherwise
            # outrank the character they are named after on pure word overlap.
            is_entity.append(any(ln.startswith("Reality:")
                                 for ln in rec.split("\n")))
            toks = tokenize(rec)
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                post_docs[term].append(doc_id)
                post_tf[term].append(tf)
            head_toks = tokenize(head)
            for term in set(head_toks):
                head_post[term].append(doc_id)
            singles = {singular(t) for t in head_toks}
            head_len.append(len(singles))
            for term in singles:
                sing_post[term].append(doc_id)
        ix.docs, ix.headlines = docs, headlines
        ix.is_entity = np.array(is_entity, dtype=bool)
        ix.has_paren = np.array(has_paren, dtype=bool)
        ix.head_len = np.array(head_len, dtype=np.int32)
        ix.lengths = np.array(lengths, dtype=np.int32)
        ix.avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        order = sorted(post_docs)
        ix.postings, ix.post_docs = _csr(post_docs, order)
        _, ix.post_tf = _csr(post_tf, order)
        ix.head_postings, ix.head_docs = _csr(head_post, sorted(head_post))
        ix.head_singular, ix.singular_docs = _csr(sing_post, sorted(sing_post))
        ix._derive()
        return ix

    def search(self, query: str, k: int = 3) -> list:
        """Return [(doc_id, score)] best first."""
        terms = tokenize(query)
        if not terms or not self.docs:
            return []
        n = len(self.docs)
        scores = np.zeros(n, dtype=np.float64)
        for term in set(terms):
            span = self.postings.get(term)
            if span is None:
                continue
            start, end = span
            df = end - start
            # BM25 idf: rare terms carry the signal, terms in every record none
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            ids = self.post_docs[start:end]
            tf = self.post_tf[start:end]
            scores[ids] += idf * (tf * (K1 + 1.0)) / (tf + self.k1_norm[ids])
            head = self.head_postings.get(term)
            if head is not None:
                # the entity you named is almost always the record you want,
                # not every page that happens to mention it
                scores[self.head_docs[head[0]:head[1]]] += idf * HEADLINE_BOOST
        # Every contribution above is strictly positive, so a non-zero score
        # is exactly "this record matched something".
        hit = np.flatnonzero(scores)
        if hit.size == 0:
            return []
        # Entity resolution, on top of BM25 relevance. BM25 alone ranks by word
        # overlap, which cannot tell the character "Moon Knight" from a page
        # that merely mentions him a lot.
        entity = {singular(t) for t in strip_question(tuple(terms))}
        # The headline test below is all-or-nothing: one stray term and the
        # entity keeps neither FULL_HEAD_BONUS nor its notability score. For
        # "tell me about spider-man in detail" the survivor "detail" cost
        # Spider-Man ~30 points and handed the query to a comic issue. Keep
        # only terms discriminative enough to be part of a name.
        if len(entity) > 1:
            cutoff = n * ENTITY_DF_RATIO
            rare = {t for t in entity if self.df(t) <= cutoff}
            entity = rare or entity
        if entity:
            named = self._headlines_containing(entity)
            named = named[scores[named] > 0]
            scores[named] += FULL_HEAD_BONUS
            # the tightest headline containing the whole query is the entity
            # itself, not a list or a comic issue named after it
            scores[named] -= BREVITY_PENALTY * np.maximum(
                0, self.head_len[named] - len(entity))
            # Notability. 18 Earth-616 records are headlined exactly
            # "Spider-Man" -- Peter Parker plus an android, an LMD, a Skrull
            # impostor. Page size is what separates them (125,811 chars vs a
            # stub), and BM25 length normalisation penalises precisely the
            # one we want, so it is added back here.
            scores[named] += NOTABILITY_WEIGHT * self.log1p_len[named]
        if not any(REALITY_RE.fullmatch(t) for t in terms):
            scores[hit[~self.has_paren[hit]]] += MAIN_CONTINUITY_BONUS
        scores[hit[self.is_entity[hit]]] += ENTITY_PAGE_BONUS
        return self._top(scores, hit, k)

    def _headlines_containing(self, entity: set):
        """Doc ids whose headline holds every term in `entity`."""
        found = None
        for term in entity:
            span = self.head_singular.get(term)
            if span is None:
                return np.empty(0, dtype=np.int32)
            ids = self.singular_docs[span[0]:span[1]]
            found = ids if found is None else np.intersect1d(found, ids,
                                                            assume_unique=True)
            if found.size == 0:
                break
        return found

    @staticmethod
    def _top(scores, hit, k: int) -> list:
        """The k best of `hit`, ties broken by ascending doc id.

        Sorting every match cost more than scoring them. Partitioning finds
        the k-th score in one pass and only the records that reach it are
        sorted - `>=`, not `>`, or a tie at the cut would drop a record the
        full sort would have kept.
        """
        s = scores[hit]
        if s.size > k:
            cut = -np.partition(-s, k - 1)[k - 1]
            keep = np.flatnonzero(s >= cut)
            hit, s = hit[keep], s[keep]
        order = np.argsort(-s, kind="stable")[:k]   # stable: doc id ascending
        return [(int(hit[i]), float(s[i])) for i in order]

    # Plain data is serialised, never the class itself: pickling the object
    # ties the index file to this module's import path, so it breaks whenever
    # search.py is imported under a different name.
    def save(self, path=INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "format": FORMAT,
            "docs": self.docs,
            "headlines": self.headlines,
            "is_entity": self.is_entity,
            "has_paren": self.has_paren,
            "postings": self.postings,
            "post_docs": self.post_docs,
            "post_tf": self.post_tf,
            "head_postings": self.head_postings,
            "head_docs": self.head_docs,
            "head_singular": self.head_singular,
            "singular_docs": self.singular_docs,
            "head_len": self.head_len,
            "lengths": self.lengths,
            "avg_len": self.avg_len,
        }
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @staticmethod
    def load(path=INDEX_PATH) -> "Index":
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        if blob.get("format") != FORMAT:
            raise ValueError(
                f"{path} was written by an older index layout "
                f"(format {blob.get('format')}, this build wants {FORMAT}). "
                f"Rebuild it: py retrieve/search.py --build")
        ix = Index()
        for field in ("docs", "headlines", "is_entity", "has_paren",
                      "postings", "post_docs", "post_tf", "head_postings",
                      "head_docs", "head_singular", "singular_docs",
                      "head_len", "lengths", "avg_len"):
            setattr(ix, field, blob[field])
        ix._derive()
        return ix


def format_context(index: Index, query: str, k: int = 3,
                   max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Retrieved records as the `Context:` block the model is trained to read.

    Delegates to retrieve/context.py, the one implementation shared with
    stage-3 data generation. These had drifted: this side sent up to three
    FULL records including History:, while training used one record trimmed
    to 700 chars. On "who created moon knight" that was 11,200 tokens against
    a 1,024-token window.
    """
    hits = index.search(query, k=k)
    if not hits:
        return ""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "context", Path(__file__).resolve().parent / "context.py")
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    return ctx.build_context([index.text(doc_id) for doc_id, _ in hits],
                             total_chars=max_chars)


def read_records(path: Path):
    buf = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(SEPARATOR):
                rec = "".join(buf).strip()
                if rec:
                    yield rec
                buf = []
            else:
                buf.append(line)
    rec = "".join(buf).strip()
    if rec:
        yield rec


def corpus_records():
    for f in sorted(CURATED.glob("*.txt")):
        if f.stat().st_size > 1000:
            yield from read_records(f)


def build_cli() -> int:
    import time
    t0 = time.time()
    records = list(corpus_records())
    print(f"read {len(records):,} records in {time.time()-t0:.1f}s", flush=True)
    t1 = time.time()
    ix = Index.build(records)
    print(f"indexed {len(ix):,} docs, {len(ix.postings):,} terms "
          f"in {time.time()-t1:.1f}s", flush=True)
    ix.save()
    print(f"saved {INDEX_PATH} ({INDEX_PATH.stat().st_size/1e6:.0f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="search the corpus")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()
    if args.build:
        return build_cli()
    if not args.query:
        ap.print_help()
        return 0
    if not INDEX_PATH.exists():
        raise SystemExit(f"no index â€” run: py retrieve/search.py --build")
    ix = Index.load()
    q = " ".join(args.query)
    for rank, (doc_id, score) in enumerate(ix.search(q, k=args.k), 1):
        print(f"\n--- {rank}. score {score:.2f} ---")
        print(ix.text(doc_id)[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())


