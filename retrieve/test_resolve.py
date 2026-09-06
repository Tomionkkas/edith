"""Tests for two-stage entity resolution.

BM25 answers "which documents contain these words". That is a different
question from "which character is this", and conflating them cost five
successive ranking patches that each helped and each broke something.
"""
import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("resolve", HERE / "resolve.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class FakeIndex:
    def __init__(self, records):
        self.docs = list(range(len(records)))
        self.headlines = [r.split("\n")[0] for r in records]
        self._records = records

    def text(self, doc_id):
        return self._records[doc_id]

    def __len__(self):
        return len(self.docs)


def rec(headline, full="", aliases="", reality="Earth-616", pad=10, extra=""):
    lines = [headline]
    if "Page: " not in extra:
        # Every genuine Marvel Database character record carries a `Page:`
        # line (has_page_title()'s whole reason to exist), even when it just
        # repeats the headline. Omitting it made these fixtures look like
        # page-less Wikipedia articles once build() started capping a
        # page-less record's identity provenance - a fixture gap the cap
        # exposed, not a real distinction these tests meant to draw.
        lines.append(f"Page: {headline}")
    if full:
        lines.append(f"Full name: {full}")
    if aliases:
        lines.append(f"Other aliases: {aliases}")
    if reality:
        lines.append(f"Reality: {reality}")
    lines.append("Created by: Someone; Someone Else")
    lines.append("First appearance: A Comic Vol 1 1")
    if extra:
        lines.append(extra)
    lines.append("History:")
    lines.append("body " * pad)
    return "\n".join(lines)


def wiki_rec(title, pad=200):
    """A Wikipedia-tier record: prose, no `Page:` line, no Reality.

    It still reaches names_of(), because field_count() counts any line with
    ": " and encyclopedic prose is full of them - a real article reports 51
    fields. That is why the tier can win a name at all, and why the guard
    has to be the `Page:` line rather than a field count.
    """
    nl = chr(10)
    return nl.join([
        title,
        "Marvel Comics fictional character",
        "Created by: Stan Lee and Jack Kirby",
        "Publisher: Marvel Comics",
        "First issue: September 1963",
        "body " * pad,
    ])


class Normalisation(unittest.TestCase):
    def test_plurals_collapse(self):
        self.assertEqual(R.norm("Wolverines"), R.norm("Wolverine"))

    def test_case_collapses(self):
        self.assertEqual(R.norm("MOON KNIGHT"), R.norm("moon knight"))

    def test_a_hyphenated_name_is_indexed_under_every_form(self):
        """The corpus writes Spider-Man; people type "spiderman" and
        "spider man". Keeping the hyphen meant "what are spidermans
        variants" could not reach Spider-Man at all - it found an obscure
        1940s character called The Spiderman instead."""
        self.assertEqual(R.norm("Spider-Man"), ("spiderman",))
        self.assertIn(("spiderman",), R.keys_of("Spider-Man"))
        self.assertIn(("spider", "man"), R.keys_of("Spider-Man"))

    def test_short_s_words_are_not_stripped(self):
        self.assertEqual(R.norm("Gus"), ("gus",))


class PrimaryVersusAlias(unittest.TestCase):
    """Beta Ray Bill is also called Thor. That must not win "thor"."""

    def setUp(self):
        self.ix = FakeIndex([
            rec("Thor Odinson (Earth-616)", full="Thor Odinson", pad=4000),
            rec("Beta Ray Bill (Earth-616)", full="Beta Ray Bill",
                aliases="Thor", pad=800),
        ])
        self.names = R.build(self.ix)

    def test_primary_name_beats_an_alias(self):
        got = R.resolve(self.names, "thor")
        self.assertEqual(self.ix.headlines[got], "Thor Odinson (Earth-616)")

    def test_alias_still_resolves_when_it_is_the_only_match(self):
        got = R.resolve(self.names, "beta ray bill")
        self.assertEqual(self.ix.headlines[got], "Beta Ray Bill (Earth-616)")


class Ranking(unittest.TestCase):
    def test_bigger_page_beats_a_smaller_exact_name(self):
        """A 4 KB record headlined "Thor" must not beat a 116 KB Thor Odinson."""
        ix = FakeIndex([
            rec("Thor", full="Thor Variant", pad=40),
            rec("Thor Odinson (Earth-616)", full="Thor Odinson", pad=4000),
        ])
        got = R.resolve(R.build(ix), "thor")
        self.assertEqual(ix.headlines[got], "Thor Odinson (Earth-616)")

    def test_main_continuity_beats_an_adaptation(self):
        ix = FakeIndex([
            rec("Thor (Marvel Cinematic Universe)", reality="", pad=1500),
            rec("Thor Odinson (Earth-616)", full="Thor Odinson", pad=1000),
        ])
        got = R.resolve(R.build(ix), "thor")
        self.assertEqual(ix.headlines[got], "Thor Odinson (Earth-616)")

    def test_prose_records_are_excluded(self):
        """A page with no fields cannot answer "who created X"."""
        ix = FakeIndex([
            "Doctor Doom\n" + "prose about doom. " * 400,
            rec("Doctor Doom", full="Victor von Doom", pad=50),
        ])
        got = R.resolve(R.build(ix), "doctor doom")
        self.assertIn("Full name: Victor von Doom", ix.text(got))


class Matching(unittest.TestCase):
    def setUp(self):
        self.ix = FakeIndex([
            rec("Moon Knight", full="Marc Spector", pad=2000),
            rec("Wolverine", full="James Howlett", pad=3000),
            rec("Wolverines", full="", pad=60),
        ])
        self.names = R.build(self.ix)

    def test_name_inside_a_longer_question(self):
        got = R.resolve(self.names, "who created moon knight")
        self.assertEqual(self.ix.headlines[got], "Moon Knight")

    def test_possessive_reaches_the_character_not_the_team_book(self):
        got = R.resolve(self.names, "wolverines")
        self.assertEqual(self.ix.headlines[got], "Wolverine")

    def test_unknown_name_resolves_to_nothing(self):
        self.assertIsNone(R.resolve(self.names, "zyxthaloraxian"))

    def test_empty_query(self):
        self.assertIsNone(R.resolve(self.names, ""))

    def test_empty_index(self):
        self.assertIsNone(R.resolve({}, "wolverine"))

    def test_a_much_longer_name_does_not_match_a_short_query(self):
        """"doom" must not reach "Doctor Doom's Fearsome Foes"."""
        ix = FakeIndex([rec("Doctor Doom's Fearsome Foes Assemble", pad=900)])
        self.assertIsNone(R.resolve(R.build(ix), "doom"))


class QueryNormalisation(unittest.TestCase):
    """Articles and question scaffolding must not block a name match."""

    def setUp(self):
        self.ix = FakeIndex([
            rec("Lord Hood", full="Parker Davis Robbins",
                aliases="Lord Hood", pad=900),
            rec("Hood", full="Jim Torrence", pad=60),
        ])
        self.names = R.build(self.ix)

    def test_article_does_not_block_a_match(self):
        """"the hood" keyed as ("the","hood") is not a subset of
        ("lord","hood"), so Parker Robbins was unreachable."""
        got = R.resolve(self.names, "the hood")
        self.assertEqual(self.ix.headlines[got], "Lord Hood")

    def test_question_words_are_ignored(self):
        """Scaffolding that resolve() strips for itself.

        Intent words like "powers" are deliberately NOT in its noise list -
        "power" is part of Power Man - and are removed upstream by
        facts.entity_text() before resolve() is called.
        """
        for q in ("who created the hood", "where is the hood from",
                  "the hood", "tell me about the hood"):
            got = R.resolve(self.names, q)
            self.assertEqual(self.ix.headlines[got], "Lord Hood", q)


    def test_a_query_of_only_noise_does_not_crash(self):
        R.resolve(self.names, "what is the")

    def test_articles_inside_names_still_match(self):
        ix = FakeIndex([rec("The Thing", full="Benjamin Grimm", pad=500)])
        got = R.resolve(R.build(ix), "who created the thing")
        self.assertEqual(ix.headlines[got], "The Thing")

    def test_a_demonstrative_keys_empty_like_its_pronoun_siblings(self):
        """"this" is scaffolding, the same class of word as "he"/"it"/"they" -
        left in the key it blocks the fallback-to-previous path in
        engine.resolved_doc(): "who is this" keyed as ("this",) instead of
        () and matched on the literal word "this" rather than falling back
        to whoever was already being discussed. Its siblings are already in
        QUERY_NOISE and already key empty; the demonstratives must behave
        the same way, not just happen to be empty on their own."""
        for word in ("this", "that", "these", "those"):
            self.assertEqual(R.query_key(f"who is {word}"), (), word)
        for word in ("he", "it", "him", "its", "they"):
            self.assertEqual(R.query_key(f"who is {word}"), (), word)


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

def reference_resolve(names, query, known_words=None):
    """The exhaustive scan, kept as the oracle for the indexed version.

    This is what resolve() used to be: a loop over all 84,734 names, building
    a set per name. Correctness is defined as "the same answer this gives",
    so it stays here rather than being deleted.
    """
    if not names:
        return None
    key = tuple(t for t in R.norm(query) if t not in R.QUERY_NOISE) or R.norm(query)
    if not key:
        return None
    terms = set(key)
    vocab = R.vocabulary(names)
    best = None
    for name, entries in names.items():
        name_set = set(name)
        if name == key:
            tier = 2
        elif name_set <= terms:
            tier = 1
        elif terms <= name_set and len(name) - len(key) <= R.EXTRA_TOKENS:
            tier = 0
        else:
            continue
        known = vocab if known_words is None else known_words
        if tier == 1 and any(t not in known for t in terms - name_set):
            continue
        doc_id, size, main, primary, _page, famous = entries[0]
        cand = R.rank(main, primary, size, tier, doc_id, famous)
        if best is None or cand > best[0]:
            best = (cand, doc_id)
    return best[1] if best else None


def scanned_names(names, query):
    """Every name the oracle assigns a tier to - what enumeration must find."""
    key = tuple(t for t in R.norm(query) if t not in R.QUERY_NOISE) or R.norm(query)
    terms = set(key)
    out = set()
    for name in names:
        name_set = set(name)
        if (name == key or name_set <= terms
                or (terms <= name_set
                    and len(name) - len(key) <= R.EXTRA_TOKENS)):
            out.add(name)
    return out


WORDS = ("thor storm venom hood knight moon iron man doctor doom captain "
         "america spider woman black panther power thing king strange").split()


def fuzz_corpus(seed=7, n=140):
    """A name index with the shape that makes enumeration hard: shared tokens.

    Names built from a small pool collide heavily - "iron man", "iron",
    "man thing" - which is exactly the corpus's own shape and the case where
    a union over tokens could return too little.
    """
    import random
    rnd = random.Random(seed)
    records = []
    for i in range(n):
        head = " ".join(rnd.sample(WORDS, rnd.randint(1, 3)))
        if rnd.random() < 0.3:
            head += f" (Earth-{rnd.choice((616, 1610, 199999))})"
        alias = " ".join(rnd.sample(WORDS, rnd.randint(1, 2)))
        records.append(rec(head, full=" ".join(rnd.sample(WORDS, 2)),
                           aliases=alias, pad=rnd.randint(5, 400),
                           reality=rnd.choice(("Earth-616", "Earth-1610"))))
    return FakeIndex(records)


def fuzz_queries(seed=11, n=200):
    import random
    rnd = random.Random(seed)
    noise = ("who", "created", "what", "is", "tell", "me", "about", "the")
    out = []
    for _ in range(n):
        q = rnd.sample(WORDS, rnd.randint(1, 3))
        for _ in range(rnd.randint(0, 2)):
            q.insert(rnd.randrange(len(q) + 1), rnd.choice(noise))
        out.append(" ".join(q))
    return out


class CandidateEnumeration(unittest.TestCase):
    """The scan over every name was the entire cost of resolution.

    All three tiers require the name to share at least one token with the
    query: tier 2 is equality, tier 1 needs the name's tokens inside the
    query, tier 0 needs the query's tokens inside the name. So the union of
    the per-token name lists is a COMPLETE candidate set, and every name
    outside it would have hit `continue`. That argument is what these pin -
    an enumeration that is merely usually complete loses characters silently,
    and looks like a retrieval bug rather than an indexing one.
    """

    @classmethod
    def setUpClass(cls):
        cls.ix = fuzz_corpus()
        cls.names = R.build(cls.ix)

    def test_candidates_include_every_name_the_scan_would_score(self):
        for q in fuzz_queries():
            key = (tuple(t for t in R.norm(q) if t not in R.QUERY_NOISE)
                   or R.norm(q))
            got = set(R.candidates(self.names, set(key)))
            self.assertLessEqual(scanned_names(self.names, q), got, q)

    def test_resolution_matches_the_exhaustive_scan(self):
        for q in fuzz_queries():
            self.assertEqual(R.resolve(self.names, q),
                             reference_resolve(self.names, q), q)

    def test_matches_the_scan_with_a_corpus_vocabulary_too(self):
        """The unknown-word guard reads `known_words`; enumeration must not
        change which names reach it."""
        known = {w for name in self.names for w in name} - {"doom", "storm"}
        for q in fuzz_queries(seed=13, n=120):
            self.assertEqual(R.resolve(self.names, q, known),
                             reference_resolve(self.names, q, known), q)

    def test_candidates_are_a_small_slice_of_the_index(self):
        """The point of the exercise: a query must not touch every name."""
        key = tuple(t for t in R.norm("who created moon knight")
                    if t not in R.QUERY_NOISE)
        self.assertLess(len(R.candidates(self.names, set(key))),
                        len(self.names) / 2)

    def test_a_token_in_no_name_yields_no_candidates(self):
        self.assertEqual(list(R.candidates(self.names, {"zyxthaloraxian"})), [])


class NotabilityOracle(unittest.TestCase):
    """Wikipedia covers the characters that matter; Fandom covers all 103,811.

    So a name with a Wikipedia article in the corpus is a famous name, and
    that is the one signal the ranking never had. It settles the case where
    an obscure Earth-616 record and a famous one share a name: there really
    is an Earth-616 character called Miles Morales, aliased `Ultimatum`, and
    at 6,329 characters he beat the 72,760-character Earth-1610 Spider-Man
    for no better reason than being main continuity.

    It changes only which entry a NAME keeps, never how two names are
    compared - `rank()` is untouched, because the two candidates here sit
    under the same name and the ordering between names was never the problem.
    """

    def entries(self, notable):
        records = [
            rec("Ultimatum", pad=20, extra="Page: Miles Morales (Earth-616)"),
            rec("Spider-Man", reality="Earth-1610", pad=900,
                extra="Page: Miles Morales (Earth-1610)"),
        ]
        names = R.build(FakeIndex(records), notable=notable)
        return names[R.norm("Miles Morales")]

    def test_without_the_oracle_main_continuity_still_wins(self):
        """The old behaviour, kept for every name Wikipedia never heard of."""
        self.assertEqual(self.entries(frozenset())[0][0], 0)

    def test_a_notable_name_prefers_the_bigger_record(self):
        self.assertEqual(
            self.entries(frozenset({R.norm("Miles Morales")}))[0][0], 1)

    def test_notable_names_include_the_bare_form_of_a_disambiguated_title(self):
        """Wikipedia files him under `Venom (character)`; people type Venom."""
        notable = R.notable_names(["Venom (character)"])
        self.assertIn(R.norm("Venom"), notable)

    def test_notable_names_include_the_full_title_too(self):
        notable = R.notable_names(["Jean Grey"])
        self.assertIn(R.norm("Jean Grey"), notable)

    def test_an_empty_title_contributes_no_name(self):
        self.assertEqual(R.notable_names(["", "   "]), frozenset())

    def test_the_oracle_never_becomes_the_answer(self):
        """A Wikipedia article says a name is famous. It does not say the
        article is the record to answer from.

        Letting it take the name's first entry cost five flagships at once:
        prose carries no Reality, so the entry went main=False, and `rank()`
        - which still puts main continuity first - handed the query to a
        main-continuity record sitting under a DIFFERENT name. "beast" went
        to `Man-Beast`. A Fandom record is identified by its `Page:` line;
        prose has none.
        """
        records = [
            rec("Beast", pad=90, extra="Page: Krahllak (Earth-616)"),
            wiki_rec("Beast", pad=4000),
        ]
        names = R.build(FakeIndex(records),
                        notable=frozenset({R.norm("Beast")}))
        self.assertEqual(names[R.norm("Beast")][0][0], 0)

    def test_a_notable_name_still_prefers_the_bigger_fandom_record(self):
        """The prose guard must not undo what the oracle is for."""
        records = [
            rec("Ultimatum", pad=20, extra="Page: Miles Morales (Earth-616)"),
            rec("Spider-Man", reality="Earth-1610", pad=900,
                extra="Page: Miles Morales (Earth-1610)"),
            wiki_rec("Miles Morales", pad=4000),
        ]
        names = R.build(FakeIndex(records),
                        notable=frozenset({R.norm("Miles Morales")}))
        self.assertEqual(names[R.norm("Miles Morales")][0][0], 1)

    def test_a_famous_off_continuity_record_beats_a_main_continuity_neighbour(self):
        """build() and rank() have to express the SAME rule.

        Demoting main continuity inside a notable name while rank() still led
        with it cost six flagships: `doctor doom` went to `Avengers (Doctor
        Doom) (Earth-616)` because the real Doom's chosen entry was now
        off-continuity and lost the very first comparison to a main-continuity
        record sitting under a different name. So `main` is satisfied by
        fame - a famous record stops being penalised for its reality, and
        everything after it decides as before.
        """
        records = [
            rec("Ultimatum", pad=20, extra="Page: Miles Morales (Earth-616)"),
            rec("Spider-Man", reality="Earth-1610", pad=900,
                extra="Page: Miles Morales (Earth-1610)"),
            # a main-continuity record under a longer, unremarkable name
            rec("Morales Gang", pad=60, extra="Page: Miles Morales Gang (Earth-616)"),
        ]
        names = R.build(FakeIndex(records),
                        notable=frozenset({R.norm("Miles Morales")}))
        self.assertEqual(R.resolve(names, "who is miles morales", ), 1)

    def test_a_slightly_bigger_alternate_does_not_displace_main_continuity(self):
        """Fame alone is not enough, and measuring proved it.

        Letting fame outrank continuity outright cost six flagships, and
        every one failed the same way: to an ALTERNATE continuity. Famous
        characters have famous alternates and their pages are long, so
        `doctor doom` went to Earth-534834, `ant-man` to the MCU, `beast` to
        the film. Continuity still decides; fame only overrules it when the
        gap is not a matter of degree - a famous record that dwarfs the
        main-continuity one is a different character with the same name.
        """
        records = [
            rec("Doctor Doom", pad=400, extra="Page: Victor von Doom (Earth-616)"),
            rec("Doctor Doom", reality="Earth-534834", pad=700,
                extra="Page: Victor von Doom (Earth-534834)"),
        ]
        names = R.build(FakeIndex(records),
                        notable=frozenset({R.norm("Doctor Doom")}))
        self.assertEqual(names[R.norm("Doctor Doom")][0][0], 0)

    def test_rank_treats_fame_as_satisfying_main_continuity(self):
        notable = R.rank(main=False, provenance=R.PROV_IDENTITY, size=100,
                         tier=2, doc_id=0, notable=True)
        plain = R.rank(main=True, provenance=R.PROV_IDENTITY, size=100,
                       tier=2, doc_id=0)
        self.assertEqual(notable[1], plain[1])

    def test_rank_is_unchanged_for_a_name_nobody_has_heard_of(self):
        self.assertEqual(
            R.rank(main=True, provenance=R.PROV_IDENTITY, size=10, tier=2,
                   doc_id=3),
            R.rank(main=True, provenance=R.PROV_IDENTITY, size=10, tier=2,
                   doc_id=3, notable=False))

    def test_the_oracle_does_not_disturb_an_unambiguous_name(self):
        """One record, notable or not, is still the answer."""
        records = [rec("Wolverine", full="James Howlett", pad=500,
                       extra="Page: James Howlett (Earth-616)")]
        names = R.build(FakeIndex(records),
                        notable=frozenset({R.norm("Wolverine")}))
        self.assertEqual(R.resolve(names, "wolverine"), 0)


class PageTitleIsAPrimaryName(unittest.TestCase):
    """The `Page:` line curation now emits is a name, and a primary one.

    Fandom headlines a page by its CurrentAlias, which is right - it is what
    people type. But `Jean Grey (Earth-616)` is headlined `Phoenix` and its
    Full name is the married form `Jean Elaine Grey-Summers`: four tokens
    against a two-token query, past what EXTRA_TOKENS will bridge. Her 224 KB
    record could not be reached by "jean grey" at all, and a 6 KB Earth-616
    stub called `Jean Grey I` answered instead.

    Primary, not alias, and for the reason Beta Ray Bill established: an alias
    loses to a primary name, so indexing the title as an alias would leave the
    stub winning.
    """

    def test_page_title_becomes_a_primary_name(self):
        r = rec("Phoenix", full="Jean Elaine Grey-Summers",
                extra="Page: Jean Grey (Earth-616)")
        primary, _ = R.names_of("Phoenix", r)
        self.assertIn(R.norm("Jean Grey"), primary)

    def test_page_title_is_not_merely_an_alias(self):
        r = rec("Phoenix", full="Jean Elaine Grey-Summers",
                extra="Page: Jean Grey (Earth-616)")
        _, alias = R.names_of("Phoenix", r)
        self.assertNotIn(R.norm("Jean Grey"), alias)

    def test_reality_suffix_is_stripped_from_the_page_title(self):
        r = rec("Spider-Man", full="Miles Gonzalo Morales",
                reality="Earth-1610", extra="Page: Miles Morales (Earth-1610)")
        primary, _ = R.names_of("Spider-Man (Earth-1610)", r)
        self.assertIn(R.norm("Miles Morales"), primary)
        self.assertNotIn(R.norm("Miles Morales Earth 1610"), primary)

    def test_a_disambiguator_is_not_part_of_the_name(self):
        """Fandom titles are `Name (disambiguator) (Reality)`, and the middle
        parenthetical is filing, not naming. Taking it as a name made
        `Sasquatch (Beast) (Earth-616)` answer to "beast" - at tier 0, where
        it then outweighed the record actually called Beast on size."""
        r = rec("Sasquatch", full="Walter Langkowski",
                extra="Page: Sasquatch (Beast) (Earth-616)")
        primary, _ = R.names_of("Sasquatch", r)
        self.assertIn(R.norm("Sasquatch"), primary)
        self.assertNotIn(R.norm("Sasquatch Beast"), primary)

    def test_a_symbiote_page_still_yields_the_bare_name(self):
        """The same rule, wanted rather than feared: the 162 KB
        `Venom (Symbiote) (Earth-616)` should answer to "venom"."""
        r = rec("Venom", extra="Page: Venom (Symbiote) (Earth-616)")
        primary, _ = R.names_of("Venom", r)
        self.assertIn(R.norm("Venom"), primary)

    def test_a_record_without_a_page_line_is_unaffected(self):
        r = rec("Wolverine", full="James Howlett")
        primary, _ = R.names_of("Wolverine", r)
        self.assertEqual(primary, {R.norm("Wolverine"), R.norm("James Howlett")})

    def test_the_big_record_now_wins_its_own_name(self):
        """End to end: the stub owned "jean grey" only because she did not."""
        records = [
            rec("Jean Grey I", pad=40),
            rec("Phoenix", full="Jean Elaine Grey-Summers", pad=4000,
                extra="Page: Jean Grey (Earth-616)"),
        ]
        names = R.build(FakeIndex(records))
        self.assertEqual(R.resolve(names, "tell me about jean grey"), 1)


class KindReading(unittest.TestCase):
    """`unknown` is the load-bearing case: curated/ is gitignored, so someone
    can be holding a corpus with no Kind lines, and it must merely be
    un-improved rather than broken."""

    def test_reads_the_kind(self):
        r = rec("Wakanda", extra="Kind: location")
        self.assertEqual(R.kind_of(r), "location")

    def test_a_record_without_a_kind_is_unknown(self):
        self.assertEqual(R.kind_of(rec("Wolverine")), "unknown")

    def test_it_stops_at_the_history(self):
        """A History paragraph mentioning "Kind: " is prose, not a field."""
        r = rec("Wolverine") + chr(10) + "Kind: location"
        self.assertEqual(R.kind_of(r), "unknown")

    def test_an_empty_record_is_unknown(self):
        self.assertEqual(R.kind_of(""), "unknown")


class Rivals(unittest.TestCase):
    """How many records contend for the name a query resolves to.

    The cheapest confidence signal: "spider-man" has hundreds behind it,
    "spider-man 2099" has one. Phase 4.7's sweep measures whether cheap is
    also good enough.
    """

    def setUp(self):
        self.ix = FakeIndex([
            rec("Spider-Man", full="Peter Parker", pad=4000),
            rec("Spider-Man", reality="Earth-1610", full="Miles Morales", pad=900),
            rec("Spider-Man", reality="Earth-928", full="Miguel O'Hara", pad=800),
            rec("Moon Knight", full="Marc Spector", pad=2000),
        ])
        self.names = R.build(self.ix)

    def test_a_contested_name_counts_its_rivals(self):
        self.assertGreaterEqual(R.rivals(self.names, "spider-man"), 3)

    def test_an_uncontested_name_has_one(self):
        self.assertEqual(R.rivals(self.names, "moon knight"), 1)

    def test_a_name_nobody_has_is_zero(self):
        self.assertEqual(R.rivals(self.names, "zyxthaloraxian"), 0)

    def test_it_counts_the_WINNING_name_not_every_candidate(self):
        """"who created moon knight" must count Moon Knight's rivals, not
        every name the question brushes against."""
        self.assertEqual(R.rivals(self.names, "who created moon knight"), 1)

    def test_an_empty_query_is_zero(self):
        self.assertEqual(R.rivals(self.names, ""), 0)


class RivalsUnknownWordGuard(unittest.TestCase):
    """rivals() must refuse exactly where resolve() refuses.

    resolve() rejects a tier-1 candidate when the query contains a word the
    corpus vocabulary has never seen - Matching.test_unknown_name_resolves_to_nothing
    covers the plain case, "zyxthaloraxian" alone. rivals() had no equivalent
    guard, so a query pairing a real, tier-1-eligible name with an
    out-of-vocabulary word could still "win" a name and report a nonzero
    rival count for a query resolve() itself refuses outright.
    """

    def setUp(self):
        self.ix = FakeIndex([
            rec("Moon Knight", full="Marc Spector", aliases="Fist", pad=2000),
        ])
        self.names = R.build(self.ix)

    def test_an_unknown_word_beside_a_real_tier_one_name_is_zero(self):
        """"zyxthaloraxian" appears in no name in this index; "moon knight"
        is a real, tier-1-eligible name inside the query. resolve() refuses
        this query outright (see Matching.test_unknown_name_resolves_to_nothing
        for the same guard on a bare unknown name); rivals() must too."""
        self.assertEqual(
            R.rivals(self.names, "who created zyxthaloraxian moon knight"), 0)

    def test_an_ordinary_in_vocabulary_tier_one_query_still_counts(self):
        """The fix must not be "return 0 more often": an extra query word
        that IS in the corpus vocabulary - here, Moon Knight's own alias -
        still reaches a real tier-1 match and reports its real count."""
        self.assertEqual(R.rivals(self.names, "moon knight fist"), 1)


class Confidence(unittest.TestCase):
    """The size margin between the best candidate record and the runner-up.

    The spec's first candidate signal, and the one the sweep kept: `rivals()`
    saturates at the build()-imposed cap of 8 and cannot separate "answer"
    from "offer" even uncapped (measured 2026-09-02 - emma frost 115 rivals,
    galactus 211, neither ambiguous; spider-man 907, which is). The margin
    between the top two records does separate them.
    """

    def test_two_evenly_matched_records_give_a_ratio_near_one(self):
        """Same size, same continuity: nothing picks one over the other."""
        ix = FakeIndex([
            rec("Venom", full="Eddie Brock", pad=900),
            rec("Venom", full="Eddie Brock Two", pad=900),
        ])
        got = R.confidence(R.build(ix), "venom")
        self.assertAlmostEqual(got, 1.0, delta=0.05)

    def test_a_dominant_record_gives_a_large_ratio(self):
        ix = FakeIndex([
            rec("Galactus", pad=5000),
            rec("Galactus", reality="Earth-1610", pad=50),
        ])
        got = R.confidence(R.build(ix), "galactus")
        self.assertGreater(got, 10)

    def test_a_single_candidate_is_maximum_confidence(self):
        """One record is the MAXIMUM confidence, not the minimum - the
        opposite of what an unguarded division would do."""
        ix = FakeIndex([rec("Moon Knight", full="Marc Spector", pad=2000)])
        got = R.confidence(R.build(ix), "moon knight")
        self.assertEqual(got, float("inf"))

    def test_nothing_resolving_is_maximum_confidence(self):
        ix = FakeIndex([rec("Moon Knight", full="Marc Spector", pad=2000)])
        names = R.build(ix)
        self.assertEqual(R.confidence(names, "zyxthaloraxian"), float("inf"))

    def test_an_empty_query_is_maximum_confidence(self):
        names = R.build(FakeIndex([rec("Moon Knight", pad=200)]))
        self.assertEqual(R.confidence(names, ""), float("inf"))

    def test_an_empty_index_is_maximum_confidence(self):
        self.assertEqual(R.confidence({}, "wolverine"), float("inf"))

    def test_a_record_reachable_under_two_names_is_not_compared_to_itself(self):
        """FINDING 2026-09-02: emma frost measured 1.0 before dedup, 30.0
        after. A record is reachable under several names - its headline and
        its `Page:` title, for instance - so a query can match TWO different
        name buckets that each resolve, on their own, to the SAME doc. Taking
        the raw top two across buckets then compares that record to itself
        and reads as maximally ambiguous. It must dedupe by doc id first.

        Built the way the real case happens: one record's headline and its
        `Full name` normalise to two different name keys ("Emma Frost Prime"
        and "Emma Frost Omega"), both of which are legitimate, undemoted
        (tier 0) matches for the query "emma frost" on their own - so without
        dedup this one record supplies BOTH of the top two slots.
        """
        ix = FakeIndex([
            rec("Emma Frost Prime", full="Emma Frost Omega", pad=900),
            rec("Emma Frost Beta", pad=50),
        ])
        names = R.build(ix)
        # Sanity: doc 0 really is reachable under two distinct name keys.
        primary, _ = R.names_of(ix.headlines[0], ix.text(0))
        self.assertIn(R.norm("Emma Frost Prime"), primary)
        self.assertIn(R.norm("Emma Frost Omega"), primary)

        got = R.confidence(names, "emma frost")
        # Without dedup this reads 1.0 (doc 0 vs itself). Deduped, it is
        # doc 0 (900 chars) against the genuinely different doc 1 (50 chars).
        self.assertGreater(got, 5)


class ConfidenceKnownWordsParameter(unittest.TestCase):
    """FINDING 2026-09-02: confidence() judged its tier-1 unknown-word guard
    against the NAME vocabulary while production judges the same guard - via
    resolve() - against the CORPUS vocabulary (resolve.py:509,
    `index.postings`). A query word that names nobody but IS a real corpus
    word wrongly evicted every tier-1 candidate for the contested name,
    which can only push the ratio UP - silently suppressing the picker.
    Measured on the real index 2026-09-02: "who is venom symbiotic" -
    "symbiotic" is in no name but is in the corpus - read confidence() as
    `inf` (name vocabulary) and 1.54 (corpus vocabulary, same as bare
    "venom"). confidence() needs the same `known_words` escape hatch
    resolve() already has.
    """

    def setUp(self):
        self.ix = FakeIndex([
            rec("Venom", full="Eddie Brock", pad=900),
            rec("Venom", full="Eddie Brock Two", pad=880),
        ])
        self.names = R.build(self.ix)

    def test_default_still_reads_the_name_vocabulary(self):
        """"symbiotic" is in no name here, so with no known_words passed the
        guard behaves exactly as before: it rejects both tier-1 "venom"
        candidates outright, leaving fewer than two to compare."""
        self.assertNotIn("symbiotic", R.vocabulary(self.names))
        self.assertEqual(R.confidence(self.names, "venom symbiotic"),
                          float("inf"))

    def test_a_corpus_known_word_restores_the_real_contest(self):
        """Judged against a vocabulary that HAS seen "symbiotic" - the
        corpus, in production - the query must not read as MORE confident
        than "venom" alone. Silently suppressing the picker here is exactly
        the bug: two same-named, near-equal-size Venom records are a real
        contest that "symbiotic" does nothing to resolve."""
        corpus_vocab = R.vocabulary(self.names) | {"symbiotic"}
        got = R.confidence(self.names, "venom symbiotic", corpus_vocab)
        bare = R.confidence(self.names, "venom")
        self.assertAlmostEqual(got, bare, delta=0.05)


class RivalsKnownWordsParameter(unittest.TestCase):
    """The same fix rivals() needs, for the same reason - see
    ConfidenceKnownWordsParameter above. rivals() had no known_words
    parameter of its own before this; production does not call rivals(), but
    infer/evaluate.py's --picker sweep does, and it must measure the same
    guard production's confidence() now uses or the sweep is not measuring
    what ships.
    """

    def setUp(self):
        self.ix = FakeIndex([
            rec("Venom", full="Eddie Brock", pad=900),
            rec("Venom", full="Eddie Brock Two", pad=880),
        ])
        self.names = R.build(self.ix)

    def test_default_still_reads_the_name_vocabulary(self):
        self.assertEqual(R.rivals(self.names, "venom symbiotic"), 0)

    def test_a_corpus_known_word_restores_the_real_count(self):
        corpus_vocab = R.vocabulary(self.names) | {"symbiotic"}
        self.assertEqual(
            R.rivals(self.names, "venom symbiotic", corpus_vocab),
            R.rivals(self.names, "venom"))


class LoadLayoutGuard(unittest.TestCase):
    """FINDING 6 2026-09-02: this branch grew the name-index entry tuple from
    4 fields to 6 (`_page`, `famous`). A stale `names.pkl` built by the old
    layout now fails as a bare unpack ValueError in the middle of a query
    (`resolve()`'s `doc_id, size, main, primary, _page, famous = entries[0]`),
    instead of failing clearly at load time. `retrieve/search.py`'s
    `Index.load()` already guards its own format this way; `resolve.load()`
    needs the same guard."""

    def _pickle(self, names):
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        with open(tmp.name, "wb") as fh:
            pickle.dump(names, fh, protocol=4)
        return Path(tmp.name)

    def test_a_stale_four_tuple_layout_is_refused_with_valueerror(self):
        # The old layout, before this branch added `_page` and `famous`:
        # (doc_id, size, main_continuity, is_primary).
        path = self._pickle({("wolverine",): [(0, 12345, True, True)]})
        self.assertRaises(ValueError, R.load, path)

    def test_the_error_names_the_rebuild_command(self):
        path = self._pickle({("wolverine",): [(0, 12345, True, True)]})
        with self.assertRaises(ValueError) as cm:
            R.load(path)
        self.assertIn("py retrieve/build_names.py", str(cm.exception))

    def test_the_current_six_tuple_layout_still_loads(self):
        path = self._pickle({("wolverine",): [(0, 12345, True, True, True, False)]})
        names = R.load(path)
        self.assertEqual(names, {("wolverine",): [(0, 12345, True, True, True, False)]})

    def test_a_missing_file_still_returns_none(self):
        """Not the layout guard's job: an absent file is a first run, not a
        stale one, and load() already handles it before this fix."""
        self.assertIsNone(R.load(HERE / "no-such-names-file.pkl"))


class PageTitle(unittest.TestCase):
    """The `Page:` line is a record's identity - two records are headlined
    `Beast` and only one of them is Henry McCoy. The answer harness keys on
    it, so reading it needs to be a function rather than a regex written
    twice."""

    def test_reads_the_page_line(self):
        record = ("Beast\n"
                  "Page: Henry McCoy (Earth-616)\n"
                  "Created by: Stan Lee\n"
                  "History: Hank was born in Illinois.\n")
        self.assertEqual(R.page_title(record),
                         "Henry McCoy (Earth-616)")

    def test_empty_when_there_is_no_page_line(self):
        record = ("Spider-Man\n"
                  "Wikipedia prose about the character.\n")
        self.assertEqual(R.page_title(record), "")

    def test_stops_where_the_prose_starts(self):
        # A History section can quote anything, including a line that looks
        # like a schema field. Identity is only claimed above the prose.
        record = ("Beast\n"
                  "Created by: Stan Lee\n"
                  "History: he read aloud, 'Page: Not This (Earth-9)'.\n")
        self.assertEqual(R.page_title(record), "")

    def test_ignores_the_headline(self):
        # has_page_title() skips line 0 and so must this, or a record whose
        # headline happened to start `Page: ` would name itself.
        record = "Page: Not A Field\nCreated by: Stan Lee\n"
        self.assertEqual(R.page_title(record), "")

    def test_the_first_page_line_wins(self):
        record = ("Beast\n"
                  "Page: Henry McCoy (Earth-616)\n"
                  "Page: Something Later\n")
        self.assertEqual(R.page_title(record),
                         "Henry McCoy (Earth-616)")


class CodenamesArePrimary(unittest.TestCase):
    """A codename is what a record IS CALLED, the same status as its headline
    and its Page: title. `Beastmaster` and `Uncle Luke` are things a character
    is ALSO known as, and stay aliases.

    The distinction is not cosmetic: build() sorts (-main, -primary, -size)
    and resolve() reads entries[0] only, so an alias sorts below every
    primary regardless of size. Krahllak owns `beast` as a primary, so a
    codename indexed as an alias would leave `who is beast` exactly as broken
    as it was."""

    def record(self, *lines):
        return "\n".join(lines) + "\nHistory:\nprose\n"

    def test_a_codename_is_primary(self):
        rec = self.record("Chairman", "Page: Henry McCoy (Earth-616)",
                          "Codename: Beast")
        primary, alias = R.names_of("Chairman", rec)
        self.assertIn(("beast",), primary)
        self.assertNotIn(("beast",), alias)

    def test_several_codenames_are_all_primary(self):
        rec = self.record("Prowler", "Page: Aaron Davis (Earth-41940)",
                          "Codename: Prowler; Spider-Men")
        primary, _ = R.names_of("Prowler", rec)
        self.assertIn(("prowler",), primary)
        self.assertIn(("spider", "men"), primary)

    def test_other_aliases_are_still_aliases(self):
        rec = self.record("Chairman", "Page: Henry McCoy (Earth-616)",
                          "Codename: Beast",
                          "Other aliases: Beastmaster; Kreature")
        primary, alias = R.names_of("Chairman", rec)
        self.assertIn(("beast",), primary)
        self.assertIn(("beastmaster",), alias)
        self.assertIn(("kreature",), alias)

    def test_also_known_as_is_still_an_alias(self):
        rec = self.record("Chairman", "Page: Henry McCoy (Earth-616)",
                          "Also known as: Chairman",
                          "Codename: Beast")
        primary, alias = R.names_of("Chairman", rec)
        self.assertIn(("beast",), primary)
        self.assertNotIn(("beast",), alias)

    def test_a_codename_below_the_prose_is_ignored(self):
        # names_of() stops at STOP_SECTIONS. A History section can quote
        # anything, including a line that looks like a schema field.
        rec = ("Chairman\nPage: Henry McCoy (Earth-616)\n"
               "History:\nhe wrote 'Codename: Not This'.\n")
        primary, alias = R.names_of("Chairman", rec)
        self.assertNotIn(("not", "this"), primary)
        self.assertNotIn(("not", "this"), alias)

    def test_no_codename_line_changes_nothing(self):
        rec = self.record("Spider-Man", "Page: Peter Parker (Earth-616)",
                          "Other aliases: Web-Head")
        primary, alias = R.names_of("Spider-Man", rec)
        self.assertIn(("spiderman",), primary)
        self.assertIn(("webhead",), alias)


class BetaRayBillStaysDemoted(unittest.TestCase):
    """The reason the primary/alias split exists. If a record carries another
    character's famous codename in `Codename:`, promoting it to primary puts
    this bug straight back."""

    def test_an_alias_thor_does_not_become_primary(self):
        rec = ("Beta Ray Bill\nPage: Beta Ray Bill (Earth-616)\n"
               "Other aliases: Thor\nHistory:\nprose\n")
        primary, alias = R.names_of("Beta Ray Bill", rec)
        self.assertIn(("thor",), alias)
        self.assertNotIn(("thor",), primary)


class Provenance(unittest.TestCase):
    """Where a name came from decides whether it may outrank an exact match.

    `thor odinson` is what the record IS; `spider lizard` is one of the many
    things Peter Parker has been CALLED. names_of()'s primary set holds both
    - 4.10 put codenames there on purpose, and that is what made `who is
    beast` reach Henry McCoy - so primary is too coarse to tell them apart."""

    def record(self, *lines):
        return "\n".join(lines) + "\nHistory:\nprose\n"

    def test_the_headline_is_an_identity(self):
        rec = self.record("Wolverine", "Page: James Howlett (Earth-616)")
        ident, code, alias = R.names_by_provenance("Wolverine", rec)
        self.assertIn(("wolverine",), ident)

    def test_the_page_title_is_an_identity(self):
        rec = self.record("Phoenix", "Page: Jean Grey (Earth-616)")
        ident, code, alias = R.names_by_provenance("Phoenix", rec)
        self.assertIn(("jean", "grey"), ident)

    def test_a_full_name_is_an_identity(self):
        rec = self.record("Star-Lord", "Page: Peter Quill (Earth-616)",
                          "Full name: Peter Jason Quill")
        ident, _c, _a = R.names_by_provenance("Star-Lord", rec)
        self.assertIn(("peter", "jason", "quill"), ident)

    def test_a_codename_is_a_codename(self):
        rec = self.record("Katherine Pryde (Earth-616)",
                          "Page: Katherine Pryde (Earth-616)",
                          "Codename: Shadowcat; Star-Lord")
        ident, code, _a = R.names_by_provenance("Katherine Pryde (Earth-616)", rec)
        self.assertIn(("star", "lord"), code)
        self.assertNotIn(("star", "lord"), ident)

    def test_an_also_known_as_is_an_alias(self):
        rec = self.record("Beta Ray Bill", "Page: Beta Ray Bill (Earth-616)",
                          "Other aliases: Thor")
        _i, code, alias = R.names_by_provenance("Beta Ray Bill", rec)
        self.assertIn(("thor",), alias)
        self.assertNotIn(("thor",), code)

    def test_a_name_takes_its_STRONGEST_provenance(self):
        """Peter Quill is headlined Star-Lord AND carries it as an alias.
        It is an identity, and it must not also appear lower down."""
        rec = self.record("Star-Lord", "Page: Peter Quill (Earth-616)",
                          "Also known as: Star-Lord",
                          "Codename: Legendary Star-Lord")
        ident, code, alias = R.names_by_provenance("Star-Lord", rec)
        self.assertIn(("star", "lord"), ident)
        self.assertNotIn(("star", "lord"), code)
        self.assertNotIn(("star", "lord"), alias)

    def test_names_of_is_unchanged_by_all_this(self):
        """The wrapper must return exactly what it returned before: primary
        is identity plus codename, alias is unchanged. 17 call sites and 15
        existing tests depend on it."""
        rec = self.record("Chairman", "Page: Henry McCoy (Earth-616)",
                          "Codename: Beast", "Other aliases: Beastmaster")
        ident, code, alias = R.names_by_provenance("Chairman", rec)
        primary, al = R.names_of("Chairman", rec)
        self.assertEqual(primary, ident | code)
        self.assertEqual(al, alias)
        self.assertIn(("beast",), primary)


class _FakeIndexForBuild:
    """Just enough index for build(): records in, text and headlines out."""

    def __init__(self, records):
        self._records = list(records)
        self.docs = list(range(len(self._records)))
        self.headlines = [r.split("\n")[0] for r in self._records]
        # Stand-in for the real index's corpus vocabulary (resolve()'s
        # `known_words` guard). Empty is fine here: every case this fake
        # feeds into resolve() resolves at tier 0 or tier 2, and that guard
        # only ever fires on tier 1.
        self.postings = {}

    def text(self, doc_id):
        return self._records[doc_id]

    def __len__(self):
        return len(self.docs)


class TheIndexStoresProvenance(unittest.TestCase):
    def test_entries_carry_an_int_not_a_bool(self):
        """A stale names.pkl would put True/False here, and True == 1 ==
        PROV_CODENAME would misrank silently. The type is the guard."""
        ix = _FakeIndexForBuild([
            "Star-Lord\nKind: character\nPage: Peter Quill (Earth-616)\n"
            "Full name: Peter Jason Quill\nGender: Male\nRole: Hero\n"
            "History:\nprose\n"])
        names = R.build(ix)
        for entries in names.values():
            for e in entries:
                self.assertIsInstance(e[3], int)
                self.assertNotIsInstance(e[3], bool)
                self.assertIn(e[3], (R.PROV_ALIAS, R.PROV_CODENAME,
                                     R.PROV_IDENTITY))

    def test_identity_sorts_above_codename_WITHIN_a_size_class(self):
        """star lord: Peter Quill is HEADLINED it; Kitty Pryde carries it as
        one of seven codenames. The bigger record must not win."""
        ix = _FakeIndexForBuild([
            "Katherine Pryde (Earth-616)\nKind: character\n"
            "Page: Katherine Pryde (Earth-616)\n"
            "Codename: Shadowcat; Star-Lord\nGender: Female\nRole: Hero\n"
            "History:\n" + "x" * 7000 + "\n",
            "Star-Lord\nKind: character\nPage: Peter Quill (Earth-616)\n"
            "Gender: Male\nRole: Hero\nHistory:\n" + "y" * 6000 + "\n"])
        names = R.build(ix)
        winner = names[R.norm("star lord")][0]
        self.assertEqual(winner[3], R.PROV_IDENTITY)
        self.assertIn("Peter Quill", ix.text(winner[0]))

    def test_a_much_bigger_record_beats_provenance_the_BEAST_canary(self):
        """4.10's result, and the case that killed two earlier attempts.

        17 main-continuity records are literally headlined "Beast"
        (identity). Henry McCoy carries "Beast" only as a Codename, on a
        record sixteen times longer. build() keeps just 8 entries per name,
        so sorting provenance above size drops him past the cap and DELETES
        him - `who is beast` then answers from a 6,596-character alien. An
        order of magnitude is not something provenance may overturn."""
        ix = _FakeIndexForBuild([
            "Beast\nKind: character\nPage: Krahllak (Earth-616)\n"
            "Gender: Male\nRole: Villain\nHistory:\n" + "x" * 6000 + "\n",
            "Chairman\nKind: character\nPage: Henry McCoy (Earth-616)\n"
            "Codename: Beast\nGender: Male\nRole: Hero\n"
            "History:\n" + "y" * 100000 + "\n"])
        names = R.build(ix)
        winner = names[R.norm("beast")][0]
        self.assertIn("Henry McCoy", ix.text(winner[0]))
        self.assertEqual(winner[3], R.PROV_CODENAME)


class ALongerCodenameDoesNotSwallowAnExactName(unittest.TestCase):
    """`who is Lizard` returned Peter Parker. `spider lizard` is one of his
    codenames and sits on a 126,398-character record; `lizard` is Curtis
    Connors's actual name on a 25,189-character one. rank() puts size before
    tier, so the big record won.

    Size before tier is deliberate and stays: it is how `thor` reaches the
    116 KB Thor Odinson instead of a 4 KB record headlined exactly `Thor`.
    What changes is that only an IDENTITY may use it."""

    def index(self):
        return _FakeIndexForBuild([
            "Spider-Man\nKind: character\nPage: Peter Parker (Earth-616)\n"
            "Codename: Spider-Lizard\nGender: Male\nRole: Hero\n"
            "History:\n" + "x" * 40000 + "\n",
            "Curtis Connors (Earth-616)\nKind: character\n"
            "Page: Curtis Connors (Earth-616)\nCodename: The Lizard\n"
            "Gender: Male\nRole: Villain\nHistory:\n" + "y" * 4000 + "\n"])

    def test_a_longer_codename_loses_to_the_exact_name(self):
        ix = self.index()
        names = R.build(ix)
        got = R.resolve(names, "lizard", ix.postings)
        self.assertIn("Curtis Connors", ix.text(got))

    def test_a_longer_IDENTITY_still_wins_on_size(self):
        """The Thor case, which must not regress: a small record headlined
        exactly `Thor` must not beat the huge `Thor Odinson`."""
        ix = _FakeIndexForBuild([
            "Thor Odinson (Earth-616)\nKind: character\n"
            "Page: Thor Odinson (Earth-616)\nGender: Male\nRole: Hero\n"
            "History:\n" + "x" * 40000 + "\n",
            "Thor\nKind: character\nPage: Thor (Cat) (Earth-616)\n"
            "Gender: Male\nRole: Pet\nHistory:\nshort\n"])
        names = R.build(ix)
        got = R.resolve(names, "thor", ix.postings)
        self.assertIn("Thor Odinson", ix.text(got))


class AWikipediaArticleMustNotOutrankTheRecordItDescribes(unittest.TestCase):
    """`who is spider-man 2099` regressed to the Wikipedia article once
    provenance existed. The article has no `Page:` line but plenty of
    fields, so it clears MIN_FIELDS and is headlined exactly the name -
    which made its headline an IDENTITY, tied for size class with Miguel
    O'Hara's own record, and an identity beats a codename on provenance.
    Before 4.11 both were merely "primary" and size settled it, which is
    how 4.10 made Miguel win; a page-less record's title must stay capped
    below a real character record's own name."""

    def test_the_article_does_not_beat_the_character(self):
        ix = _FakeIndexForBuild([
            "Spider-Man 2099\nKind: article\nPublisher: Marvel Comics\n"
            "First issue: November 1992\nCreated by: Peter David\n"
            "History:\n" + "x" * 40000 + "\n",
            "Miguel O'Hara\nKind: character\n"
            "Page: Miguel O'Hara (Earth-928)\nCodename: Spider-Man 2099\n"
            "Gender: Male\nRole: Hero\nHistory:\n" + "y" * 45000 + "\n"])
        names = R.build(ix)
        got = R.resolve(names, "spider-man 2099", ix.postings)
        self.assertIn("Miguel O'Hara", ix.text(got))


class AnAliasMustNeverBeatAPrimaryHoweverBigTheRecord(unittest.TestCase):
    """A size class above provenance let a giant record's ALIAS beat a tiny
    record's own PRIMARY name: `who is fred` resolved to Peter Parker, who
    answers to "Fred" only as an alias printed on a record an order of
    magnitude bigger than Fred (Psychic Fish)'s own. Primary-vs-alias was
    ABSOLUTE before 4.11 (`(-main, -primary, -size)`) and must stay so - a
    size class may only settle ties WITHIN primary, never promote an alias
    over it, however much bigger the record carrying the alias is."""

    def test_a_tiny_primary_beats_a_much_bigger_alias(self):
        ix = _FakeIndexForBuild([
            "Fred (Psychic Fish) (Earth-616)\nKind: character\n"
            "Page: Fred (Psychic Fish) (Earth-616)\nGender: Male\n"
            "Role: Pet\nHistory:\n" + "x" * 4000 + "\n",
            "Spider-Man\nKind: character\nPage: Peter Parker (Earth-616)\n"
            "Other aliases: Fred\nGender: Male\nRole: Hero\n"
            "History:\n" + "y" * 90000 + "\n"])
        names = R.build(ix)
        winner = names[R.norm("fred")][0]
        self.assertIn("Fred (Psychic Fish)", ix.text(winner[0]))
        self.assertEqual(winner[3], R.PROV_IDENTITY)
