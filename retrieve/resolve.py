"""Resolve a query to a character, before any document scoring.

The old path scored the whole query with BM25 and layered bonus rules on top,
which conflates "documents containing these words" with "which character is
this". Five successive patches to those rules each helped and each broke
something else, so this replaces the approach rather than adding a sixth.

Two facts about the corpus make a name index necessary:

**Two naming conventions.** Some characters are headlined by alias
("Wolverine", "Spider-Man"); others by real name with the alias only in a
field ("Thor Odinson (Earth-616)", "Loki Laufeyson (Earth-616)"). Matching
headlines alone cannot find the second kind, which is why "who created thor"
returned a 4 KB variant while a 116 KB Thor Odinson record sat unreached.

**Many records share a name.** 12 records are headlined exactly "Wolverine",
15 "Venom", 597 contain it. Page size picks the canonical one, with a field
floor to exclude prose pages. Field COUNT must not lead: Venom's Mary Jane
record has 9 fields against the symbiote's 7, so ranking by fields answers
symbiote questions with Mary Jane.
"""
from __future__ import annotations

import pickle
import importlib.util
import re
from collections import defaultdict
from pathlib import Path

# The data directory, loaded by file path like bootstrap.py itself is - see
# install.py. Constants only, so re-executing it per module costs nothing.
_paths_spec = importlib.util.spec_from_file_location("edith_paths", Path(__file__).resolve().parent.parent / "paths.py")
paths = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(paths)

NAMES_PATH = paths.NAMES

REALITY_SUFFIX = re.compile(r"\s*\((?:Earth|Reality)[-\w]*\)\s*$", re.I)
PAREN_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")
QUOTED = re.compile(r'"([^"]{2,30})"')
TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)+\.?|[a-z0-9]+(?:-[a-z0-9]+)+|[a-z0-9]+")

# A record needs some structure to answer a factual question. Prose pages
# (Wikipedia articles) carry none and must never win the name.
MIN_FIELDS = 3

# How many tokens longer than the query a name may be and still match.
# "thor" -> "thor odinson" is one; "doctor doom" -> a four-word team name
# is not, and matching those resolved Doom to the Human Torch.
EXTRA_TOKENS = 1
NAME_FIELDS = ("Also known as", "Full name", "Other aliases")
STOP_SECTIONS = ("History:", "Synopsis")


# Articles carry no identity and block matching: the query "the hood" keyed as
# ("the", "hood") is not a subset of ("lord", "hood"), so Parker Robbins - who
# is headlined "Lord Hood" and aliased the same - was unreachable by the name
# everyone actually calls him.
ARTICLES = frozenset(("the", "a", "an"))

# Question scaffolding. Left in the key it blocks the "query inside name"
# match: "what powers does the hood have" could not be a subset of
# ("lord", "hood"), so it fell to a lesser Hood record. NOT stripped from
# names, only from queries - a record may legitimately be called "The Thing".
QUERY_NOISE = frozenset("""
who what when where which why how whose whom is are was were be been am
do doe did done have ha had can could would should will
of in on at to for from with and or as by about
tell me my you your show give know more please
created create creates creator creators made wrote drew designed
appeared appear appears debut debuted
detail details explain explains describe describes rundown
happen happened happens occur occurred occurs
variant variants version versions incarnation incarnations
he she it they him her them his hers their theirs its this that these those
anything everything something really actually exactly just like
""".split())

# "him" IS Adam Warlock and "he" IS the High Evolutionary, so stripping them
# looks reckless. It is not: the key falls back to the unstripped query when
# stripping empties it, so "who is Him" still resolves - while "explain his
# powers" stops keying on ("his",) and reaching whoever owns that alias.

# NOT in the list: power, powers, name, man, woman, thing. They look like
# scaffolding and are parts of real names - Power Man, The Thing - and
# stripping them would break the lookups the corpus exists to answer.


def norm(text: str) -> tuple:
    """A name as a comparable key: lowercased, de-pluralised, order kept.

    "Wolverines" and "Wolverine" collapse together so a possessive query
    reaches the character rather than the team book of the same name.
    """
    out = []
    for t in TOKEN.findall(text.lower()):
        t = t.replace("-", "")          # spider-man and spiderman are one name
        if t in ARTICLES or not t:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
            t = t[:-1]
        out.append(t)
    return tuple(out)


def keys_of(text: str) -> set:
    """Every form of a name a person might type.

    The corpus writes Spider-Man; people type "spiderman" and "spider man".
    One token with the hyphen removed matches the first two, and the split
    form matches the third, so a name is indexed under both. Without this
    "what are spidermans variants" could not reach Spider-Man at all - it
    matched an obscure 1940s character called The Spiderman instead.
    """
    return {norm(text), norm(text.replace("-", " "))} - {()}


# Where a name came from, and how much authority that gives it. Higher wins.
#
# The distinction 4.10 could not make. It put codenames in the PRIMARY set,
# correctly - that is what let `who is beast` reach Henry McCoy's
# 105,792-character record instead of an obscure alien headlined the same.
# But primary then held two different things: what a record IS, and one of
# the many things it has been CALLED. `thor odinson` is the first;
# `spider lizard`, sitting on Peter Parker's 126 KB record, is the second,
# and it was beating the record actually named `Lizard`.
PROV_IDENTITY = 2      # the headline, the Page: title, the Full name
PROV_CODENAME = 1      # the Codename: line
PROV_ALIAS = 0         # Also known as, Other aliases


def names_by_provenance(headline: str, record: str):
    """(identity names, codename names, alias names) for one record.

    Three tiers, not two. IDENTITY is what the record IS: the headline, the
    `Page:` title, the `Full name` field - "Thor Odinson" is this for Thor
    Odinson (Earth-616). CODENAME is a name it goes BY, from the `Codename:`
    line - "Beast" is this for Henry McCoy, not what he is but what he is
    called, which is still enough authority to beat an obscure record's own
    headline (4.10). ALIAS is everything else it is also known as. The split
    matters: Beta Ray Bill answers to "Thor" as an ALIAS, and without the
    distinction "who created thor" resolved to him rather than to Thor
    Odinson, whose identity name is "Thor Odinson".
    """
    identity = set(keys_of(REALITY_SUFFIX.sub("", headline)))
    bare = PAREN_SUFFIX.sub("", headline)
    if bare != headline:
        identity.add(norm(bare))
    codename = set()
    alias = set()
    for line in record.split(chr(10))[1:]:
        if line.startswith(STOP_SECTIONS):
            break
        if line.startswith("Page: "):
            # The source page title, treated exactly like the headline: it is
            # what the record is called, so it is primary. `Jean Grey
            # (Earth-616)` is headlined `Phoenix`, and without this her name
            # appears nowhere in her own record.
            # A Fandom title is `Name (disambiguator) (Reality)`, and only the
            # first part names anything. Keeping the disambiguator made
            # `Sasquatch (Beast) (Earth-616)` a record called "Sasquatch
            # Beast", which then answered "beast" on size.
            page = line[6:].strip()
            while True:
                stripped = PAREN_SUFFIX.sub("", page)
                if stripped == page:
                    break
                page = stripped
            identity |= keys_of(page)
        if line.startswith("Full name: "):
            full = line[11:].strip()
            identity.add(norm(full))
            # 'Anthony Edward "Tony" Stark' is how the corpus writes a
            # nickname, and nobody types the whole thing. Without the short
            # form "tony stark" reached Iron Lad, whose Full name happens to
            # be 'Anthony "Tony" Stark' - one token shorter, so it fitted
            # under EXTRA_TOKENS while Iron Man did not.
            identity |= keys_of(full)
            nick = QUOTED.search(full)
            if nick:
                identity |= keys_of(nick.group(1) + " " + full[nick.end():])
        if line.startswith("Codename: "):
            # A codename is what the record IS CALLED - the same status as
            # its headline and its Page: title - so it is primary. It has to
            # be: build() sorts every alias below every primary regardless of
            # size, and `Krahllak (Earth-616)` owns "beast" as a primary, so
            # a codename indexed as an alias leaves `who is beast` answering
            # from an obscure alien with a 6,596-character record while Henry
            # McCoy's 105,792-character one is not even a candidate.
            for c in line[10:].split(";"):
                codename |= keys_of(c.strip())
        for label in ("Also known as", "Other aliases"):
            if line.startswith(label + ": "):
                for a in line[len(label) + 2:].split(";"):
                    a = PAREN_SUFFIX.sub("", a.strip())
                    if 0 < len(a) < 60:
                        alias |= keys_of(a)
    identity = {n for n in identity if n}
    codename = {n for n in codename if n and n not in identity}
    alias = {n for n in alias if n and n not in identity and n not in codename}
    return identity, codename, alias


def names_of(headline: str, record: str):
    """(primary names, alias names) — the two-way split, unchanged.

    Kept because 17 call sites unpack two values and the guarantee they test
    is still true. `names_by_provenance()` is the finer view; primary is
    simply identity and codename together, which is exactly what this
    returned before provenance existed.
    """
    identity, codename, alias = names_by_provenance(headline, record)
    return identity | codename, alias


COMIC_TAIL = re.compile(r"\s+is an? .*$")


def untitled(headline: str) -> str:
    """The title inside a schema-less record's headline.

    Issue records announce themselves in a sentence - "Amazing Spider-Man
    Vol 6 32 is a Marvel comic. It was published on..." - so the title is
    everything before the verb.
    """
    return COMIC_TAIL.sub("", headline.split(". ")[0]).strip()


def field_count(record: str) -> int:
    n = 0
    for line in record.split("\n")[1:]:
        if line.startswith(STOP_SECTIONS):
            break
        if ": " in line:
            n += 1
    return n


def is_main_continuity(record: str) -> bool:
    """Earth-616 is the main comics continuity, and the page people mean.

    Without this, "thor" resolves to the 47 KB Marvel Cinematic Universe
    article, which owns the bare name "Thor" exactly, over the 116 KB Thor
    Odinson (Earth-616) record that answers to "Thor Odinson".
    """
    for line in record.split(chr(10))[1:]:
        if line.startswith(STOP_SECTIONS):
            break
        if line.startswith("Reality: "):
            return line[9:].strip().lower().startswith("earth-616")
    return False


def has_page_title(record: str) -> bool:
    """Whether the record came from a Marvel Database template page.

    Curation writes a `Page:` line for exactly those, so it marks a record
    that can answer a field question. Wikipedia prose has none, and neither
    does a comic issue. That distinction is what keeps the notability oracle
    an oracle: an article is evidence about a NAME, never the answer.
    """
    for line in record.split(chr(10))[1:]:
        if line.startswith(STOP_SECTIONS):
            return False
        if line.startswith("Page: "):
            return True
    return False


def page_title(record: str) -> str:
    """The record's `Page:` value, or "" when it has none.

    The identity `infer/answer_cases.py` keys on: unique per source page,
    where a headline is not - `Beast` names both Henry McCoy and an alien
    called Krahllak. Scans the same span has_page_title() does, so the two
    can never disagree about whether a record has one.
    """
    for line in record.split(chr(10))[1:]:
        if line.startswith(STOP_SECTIONS):
            return ""
        if line.startswith("Page: "):
            return line[len("Page: "):].strip()
    return ""


def kind_of(record: str) -> str:
    """What kind of thing the record describes, or "unknown".

    Written by curation, never inferred here. Inferring it from doc-id ranges
    was considered and rejected: the curated glob order changed twice in one
    day and an inference from it was correct by luck both times.
    """
    for line in record.split(chr(10))[1:]:
        if line.startswith(STOP_SECTIONS):
            break
        if line.startswith("Kind: "):
            return line[6:].strip() or "unknown"
    return "unknown"


def notable_names(titles) -> frozenset:
    """Name keys a Wikipedia article covers, from the article titles.

    The Wikipedia tier is the notability oracle for one reason: it is
    selective where Fandom is exhaustive. 3,276 articles against 103,811
    character records means an article is evidence that people outside the
    wiki care, which is exactly the judgement "who do they mean" needs and
    exactly what page size, reality and match tier cannot express.

    Both forms are kept: Wikipedia disambiguates in the title
    (`Venom (character)`) and people type the bare name.
    """
    out = set()
    for title in titles:
        title = (title or "").strip()
        if not title:
            continue
        out |= keys_of(title)
        bare = PAREN_SUFFIX.sub("", title)
        if bare != title:
            out |= keys_of(bare)
    return frozenset(out - {()})


def size_class(size: int) -> int:
    """Order of magnitude. A record ten times longer is a different kind of
    record, and no provenance argument should overturn that.

    `infer/evaluate.py` already sweeps orderings built on this notion - its
    `decade()` - so the family is measured, not invented here.
    """
    return len(str(max(size, 1)))


def build(index, notable=frozenset()) -> dict:
    """name key -> [(doc_id, size, main, provenance, page, famous)], best first.

    Everything resolve() needs to rank is stored here, so it never touches the
    index. Size is the proxy for "the one people mean": the canonical
    Wolverine page is 137,988 characters and its variants are a few thousand.
    Field COUNT must not lead - Venom's Mary Jane record has 9 fields against
    the symbiote's 7, so ranking by fields answers symbiote questions with
    Mary Jane.
    """
    by_name = defaultdict(list)
    for doc_id in range(len(index.docs)):
        record = index.text(doc_id)
        size, main = len(record), is_main_continuity(record)
        page = has_page_title(record)
        if field_count(record) < MIN_FIELDS:
            # A record with no field schema is an issue or a Wikipedia
            # article, and 38% of the corpus is one. Excluding them meant a
            # question about a comic was answered with a character card:
            # "what is Amazing Spider-Man Vol 6 32" resolved to Spider-Man.
            # They carry their title in the headline, so index that.
            title = untitled(index.headlines[doc_id])
            for name in keys_of(title):
                if name:
                    by_name[name].append((doc_id, size, main, PROV_IDENTITY, page))
            continue
        identity, codename, alias = names_by_provenance(
            index.headlines[doc_id], record)
        # A record with no `Page:` line is not a Fandom character record - it
        # is a Wikipedia article or prose page. Its TITLE is not an identity,
        # however many fields it happens to carry. Without this cap the
        # 48,901-character article "Spider-Man 2099" outranks Miguel O'Hara's
        # own record, because the article is headlined the name while Miguel
        # carries it as a codename - which undoes phase 4.10. This cap only
        # reaches records that PASS `MIN_FIELDS`: the `field_count < MIN_FIELDS`
        # branch above already `continue`d for everything else, on a separate,
        # older path this cap does not touch - which is why 86,959 names (43%)
        # still have a page-less best entry holding `PROV_IDENTITY`.
        cap = PROV_IDENTITY if page else PROV_CODENAME
        for name in identity:
            by_name[name].append((doc_id, size, main, cap, page))
        for name in codename:
            by_name[name].append((doc_id, size, main, PROV_CODENAME, page))
        for name in alias:
            by_name[name].append((doc_id, size, main, PROV_ALIAS, page))
    for name, entries in by_name.items():
        # Main continuity first is right for a name nobody outside the wiki
        # has heard of: among a dozen equally obscure records, the Earth-616
        # one is the one meant. For a name Wikipedia covers it is wrong -
        # there is an Earth-616 Miles Morales, and he is not the one meant.
        # Size leads there instead, because fame and page length agree.
        #
        # Size CLASS above provenance, and this is the line the whole phase
        # turns on. `del entries[8:]` below keeps only the best 8 per name,
        # and 17 main-continuity records are literally headlined "Beast".
        # Henry McCoy carries "Beast" only as a Codename, so sorting
        # provenance above size drops him to 18th and the cap DELETES him -
        # measured, and it undid 4.10 outright. An order of magnitude is not
        # something a provenance argument may overturn: McCoy's 105,808
        # characters beat Krahllak's 6,596 whatever each name came from,
        # while Peter Quill and Katherine Pryde sit in the same class and
        # provenance correctly separates them.
        #
        # `-(e[3] > 0)` comes FIRST, above the size class - primary-or-alias
        # was ABSOLUTE before this phase (4.10's `(-main, -primary, -size)`)
        # and must stay so. A size class above it, not just above provenance,
        # let an alias beat a primary whenever the alias's record was merely
        # bigger: "fred" resolved to Peter Parker, who answers to "Fred" only
        # as an alias printed on a page an order of magnitude larger than
        # Fred (Psychic Fish)'s own record. Measured over the full 201,151
        # names: putting size class below the primary/alias line instead of
        # above it drops the blast radius from 1,227 changed names (0.61%) to
        # 115 (0.06%). Identity-vs-codename (`-e[3]`) still breaks ties
        # WITHIN primary, which is the whole point of this phase.
        #
        # The notable branch's own re-sort just below (`by_size`, also keyed
        # on `-e[3]`) inherited the same change in meaning: `e[3]` used to be
        # a bool there too, so that re-sort now orders identity above
        # codename above alias, not merely primary above alias.
        entries.sort(key=lambda e: (-e[2], -(e[3] > 0), -size_class(e[1]),
                                     -e[3], -e[1]))
        famous = False
        if name in notable:
            # `-e[4]` first keeps the oracle from becoming the answer: a
            # Wikipedia article says the NAME is famous, and carries no
            # schema to answer from.
            by_size = sorted(entries, key=lambda e: (-e[4], -e[3], -e[1], -e[2]))
            if by_size[0][1] >= NOTABLE_SIZE_RATIO * entries[0][1]:
                entries[:] = by_size
                famous = True
        del entries[8:]
        # Notability is a property of the NAME, but it is stored on the entry
        # so the name index stays self-contained: resolve() would otherwise
        # need the oracle passed in at query time, through every caller, for
        # a fact that was already known when the index was built.
        by_name[name] = [e + (famous,) for e in entries]
    return dict(by_name)


def save(names: dict, path=NAMES_PATH) -> None:
    with Path(path).open("wb") as fh:
        pickle.dump(names, fh, protocol=4)


# name key -> [(doc_id, size, main_continuity, is_primary, has_page, famous)].
# The entry tuple grew from 4 fields to 6 when `_page` and `famous` were
# added; a `names.pkl` written by the old layout must be refused here, not
# left to fail as a bare unpack ValueError in the middle of a query (see
# resolve()'s `doc_id, size, main, primary, _page, famous = entries[0]`).
ENTRY_LEN = 6


def load(path=NAMES_PATH):
    p = Path(path)
    if not p.exists():
        return None
    with p.open("rb") as fh:
        names = pickle.load(fh)
    for entries in names.values():
        if entries and len(entries[0]) != ENTRY_LEN:
            raise ValueError(
                f"{path} was written by an older name-index layout "
                f"({len(entries[0])} fields per entry, this build wants "
                f"{ENTRY_LEN}). Rebuild it: py retrieve/build_names.py")
        break
    return names


# Derived tables for the most recently used name index. The dict itself is
# held, not its id(): an id is reused once the object is collected, and a
# cache keyed on one silently answers for the wrong index.
_CACHE = {"names": None, "vocab": None, "tokens": None}


def _derived(names: dict) -> dict:
    if _CACHE["names"] is not names:
        _CACHE.update(names=names, vocab=None, tokens=None)
    return _CACHE


def vocabulary(names: dict) -> frozenset:
    """Every token that appears in any name. Cached per name-index object."""
    cache = _derived(names)
    if cache["vocab"] is None:
        cache["vocab"] = frozenset(t for name in names for t in name)
    return cache["vocab"]


def token_index(names: dict) -> dict:
    """token -> the names containing it. Cached per name-index object.

    84,734 names, 42,749 tokens, 39 ms to build - paid once, against 11 ms
    on every single question for the scan it replaces.
    """
    cache = _derived(names)
    if cache["tokens"] is None:
        by_token = defaultdict(list)
        for name in names:
            for token in set(name):
                by_token[token].append(name)
        cache["tokens"] = dict(by_token)
    return cache["tokens"]


def candidates(names: dict, terms: set) -> list:
    """Every name that could match `terms`, without looking at the rest.

    All three tiers need the name to share at least one token with the query:
    tier 2 is equality, tier 1 puts the name's tokens inside the query, tier 0
    puts the query's tokens inside the name. Names are never empty, so a name
    sharing no token with the query fails all three and would have hit
    `continue` - which makes the union over query tokens a COMPLETE candidate
    set, not a heuristic shortlist.

    "who created moon knight" reaches 238 names instead of 84,734.
    """
    by_token = token_index(names)
    found = {}
    for token in terms:
        for name in by_token.get(token, ()):
            found[name] = None
    return list(found)


# How much bigger a famous record must be before its size overrules main
# continuity. Fame alone is not enough: at any ratio near 1 the winner is an
# alternate continuity, because famous characters have famous alternates and
# film and Ultimate pages are long. 11.5x separates the Earth-1610 Spider-Man
# from the Earth-616 character who is also called Miles Morales; nothing that
# broke was separated by anything like it.
NOTABLE_SIZE_RATIO = 5

FRAGMENT = 1     # tier 1: the name is only a PART of the query


def rank(main: bool, provenance: int, size: int, tier: int, doc_id: int,
         notable: bool = False) -> tuple:
    """How two candidate names are compared. Higher wins.

    Named and swappable because it is the single most consequential line in
    retrieval, and the only honest way to choose it is to measure whole
    orderings against the corpus - see infer/evaluate.py --rank.

    A name that is only a FRAGMENT of the query is demoted, and nothing else
    changes. That is the match that ruins the long tail: "War" sits inside
    "War Fist", so a query naming War Fist was answered from War (First
    Horsemen), and "civil war" from the same place. The opposite direction -
    "thor" sitting inside "thor odinson" - is what the flagships depend on, so
    it keeps its place.

    Measured over ~1,050 corpus round-trips on unambiguous names, and over 40
    flagship queries where several records share a name and the answer is a
    judgement about who people mean. A ranking has to satisfy both:

        main, primary, size, tier               80.9%   39/40   (was)
        main, exact, primary, size              96.4%   31/40   (was)
        main, primary, size class, exact        86.9%   39/40   (was)
        main, not-fragment, primary, size       96.5%   39/40   (was)
        main-or-notable, not-fragment,
          primary, size, tier                   97.5%   39/40   <- this

    A --rank sweep over the 250-record corpus sample cannot tell the
    notability term apart from having no oracle at all: both the
    main-or-notable key above and the plain `main` key it replaced score
    identically on that sweep, 95.7% and 39/40 flagships each. The term's
    only measurable effect anywhere in the sweep is a single query, "who is
    miles morales" - which is exactly the case NOTABLE_SIZE_RATIO's docstring
    describes, so it stays; it just is not visible in the aggregate numbers.

    `provenance` replaced a primary/alias boolean on 2026-09-05, and this
    ordering deliberately did NOT change: `provenance > 0` is exactly the
    old `primary`. Provenance does its work in build()'s per-name sort and
    in resolve()'s tier-0 gate, not here. Two earlier attempts put it - and
    then a size class - into this tuple; the first undid 4.10, and the
    second let a tier-0 fragment ("Bog-Beast") beat an exact tier-2 match.
    """
    # `main or notable`, not a separate slot: the key has to stay positionally
    # uniform, or a notable candidate's tuple compares its primary-ness
    # against another candidate's continuity and the ordering is nonsense.
    # Fame satisfies the continuity test instead - an Earth-1610 Spider-Man
    # people have heard of stops losing to an Earth-616 record nobody has,
    # and everything below decides exactly as it did before.
    return (tier != FRAGMENT, main or notable, provenance > 0, size, tier,
            -doc_id)


def query_key(text: str) -> tuple:
    """The entity terms of a query - empty when it names nobody.

    "more details about his powers" is not a question about a character called
    Power; it is a question about whoever was being discussed. Emptiness is
    the signal, so it gets a name of its own rather than being buried inside
    resolve().
    """
    return tuple(t for t in norm(text) if t not in QUERY_NOISE)


def resolve(names: dict, query: str, known_words=None):
    """Best doc_id for the character named in `query`, or None.

    Three ways a query can name a record, ranked as tiers because none alone
    is enough:

      2 exact         "wolverine" IS the record's name
      1 name in query "who created moon knight" contains "moon knight"
      0 query in name "thor" is part of "thor odinson"

    Tier 0 exists because the corpus uses two naming conventions: some
    characters are headlined by alias, others by real name with the alias in a
    field. Without it "thor" cannot reach Thor Odinson (Earth-616) at all.

    Tier 0 is also dangerous, since a one-word query matches every name
    containing that word - "venom" reached Captain Marvel through an alias
    before EXTRA_TOKENS capped how much longer the name may be.

    Main continuity outranks everything: the Marvel Cinematic Universe article
    owns the bare name "Thor" exactly, and is not the character people mean.
    """
    if not names:
        return None
    key = query_key(query) or norm(query)
    if not key:
        return None

    terms = set(key)
    vocab = vocabulary(names)
    best = None
    for name in candidates(names, terms):
        entries = names[name]
        name_set = set(name)
        doc_id, size, main, provenance, _page, famous = entries[0]
        if name == key:
            tier = 2
        elif name_set <= terms:
            tier = 1
        elif (terms <= name_set and len(name) - len(key) <= EXTRA_TOKENS
              and provenance == PROV_IDENTITY):
            # "the query is inside the name" may only outrank an exact match
            # when the longer name is what the record IS. `thor` reaching
            # `thor odinson` is that; `lizard` reaching `spider lizard`, one
            # of Peter Parker's codenames on a 126 KB record, is not - and it
            # was beating Curtis Connors, whose actual name is Lizard.
            tier = 0
        else:
            continue
        # A partial match on an unknown entity is worse than no answer:
        # "who created Zyxthaloraxian the Devourer" matched the real entity
        # "Devourer" and answered confidently.
        #
        # `known_words` should be the CORPUS vocabulary, not the name
        # vocabulary. Judged against names alone, "give me a rundown on storm"
        # was refused because no character is called "Rundown". A word the
        # corpus has never seen at all - "zyxthaloraxian" - is the real signal.
        known = vocab if known_words is None else known_words
        if tier == 1 and any(t not in known for t in terms - name_set):
            continue
        # Order: main continuity, then provenance, then SIZE, then match
        # tier. Size before tier because a 4 KB record headlined exactly
        # "Thor" should not beat the 116 KB "Thor Odinson (Earth-616)" just
        # for owning the shorter name. Provenance before both because Beta
        # Ray Bill is also called Thor, as an alias.
        cand = rank(main, provenance, size, tier, doc_id, famous)
        if best is None or cand > best[0]:
            best = (cand, doc_id)
    return best[1] if best else None


def rivals(names: dict, query: str, known_words=None) -> int:
    """How many records contend for the name this query resolves to.

    The name index keeps at most 8 entries per name, so this is "how
    contested is this name", capped - not a corpus-wide count. That is what a
    confidence signal wants: the difference between 1 and 8 is the whole
    question, and the difference between 40 and 900 is not.
    """
    if not names:
        return 0
    key = query_key(query) or norm(query)
    if not key:
        return 0
    terms = set(key)
    # Same unknown-word guard as resolve() (resolve.py:513-515), and the same
    # `known_words` escape hatch: it should be the CORPUS vocabulary in
    # production (index.postings), not the name vocabulary - judged against
    # names alone, a real corpus word that names nobody wrongly evicts every
    # tier-1 candidate. Defaults to the name vocabulary, as before, when the
    # caller has no corpus to hand - true of every caller today, since
    # nothing in production calls rivals(); infer/evaluate.py's --picker
    # sweep is the one caller that needs the corpus vocabulary, to measure
    # the same guard confidence() and resolve() use.
    known = vocabulary(names) if known_words is None else known_words
    best, best_name = None, None
    for name in candidates(names, terms):
        name_set = set(name)
        if name == key:
            tier = 2
        elif name_set <= terms:
            tier = 1
        elif terms <= name_set and len(name) - len(key) <= EXTRA_TOKENS:
            tier = 0
        else:
            continue
        if tier == 1 and any(t not in known for t in terms - name_set):
            continue
        doc_id, size, main, primary, _page, famous = names[name][0]
        cand = rank(main, primary, size, tier, doc_id, famous)
        if best is None or cand > best:
            best, best_name = cand, name
    return len(names[best_name]) if best_name is not None else 0


def confidence(names: dict, query: str, known_words=None) -> float:
    """The size ratio of the top-RANKED candidate record to the runner-up.

    The spec's other candidate signal, and the one the sweep kept:
    `rivals()` saturates at build()'s cap of 8 and its uncapped form still
    cannot separate "answer" from "offer" - emma frost has 115 rivals,
    galactus 211, taskmaster 67, none of them ambiguous to a reader, while
    spider-man's 907 is (measured 2026-09-02). This is "closest to the
    actual decision" instead: how much bigger the top record is than its
    nearest competitor.

    NOT bounded below by 1.0. The "top" and "runner-up" are chosen by
    `rank()` - main continuity, primary-ness and tier before size - so the
    ranked winner can be SMALLER than the runner-up, and the ratio then reads
    below 1. Measured 2026-09-02: "who is the ultimate spider-man" reads
    0.019, "tell me about wolverine claws" reads 0.033. Large still means the
    top record dominates in size; a small number - whether it is 0.03 or
    3.0 - means the two are close, in whichever direction, and a reader would
    want a choice. Behaviour is unaffected either way: both readings are far
    under CONFIDENCE_TO_ASK and both correctly offer.

    Reads at most the 8 entries per name that `build()` retained, same as
    `rivals()` - so on a name with more than 8 real contenders the "runner-up"
    found here is the best of the RETAINED 8, not of every rival. Unlike
    `rivals()`, this cannot be seen by inspecting the number returned: it can
    only ever make confidence() read HIGHER than the truth (a genuine close
    rival sits outside the 8 kept), which can only cause under-asking, never
    a spurious offer. No behaviour change has been measured from it; noted so
    the saturation is not mistaken for `rivals()`'s flaw alone.

    One candidate is the MAXIMUM confidence, not the minimum, so it and the
    no-match case both return `float("inf")` rather than dividing by nothing.

    Same EXTRA_TOKENS rule and tier-1 unknown-word guard as resolve() and
    rivals() above - but NOT the same tiering. resolve()'s tier 0 additionally
    requires `provenance == PROV_IDENTITY` (4.11); neither rivals() nor this
    function gates on it, deliberately deferred - see docs/PHASE-4-ENDGAME.md
    item 1, whose measurement (11 of 1,274 sampled multi-token names rank a
    different record here than resolve() returns, `lizard` among them) is
    that work's first input, not a footnote. Where resolve() and rivals()
    only ever need the single best entry per candidate name (entries are
    stored best-first), this needs the runner-up too, so it scores every
    entry under every matching name, not just entries[0].

    That is exactly why the top two must be deduped by doc id before being
    compared: a record is reachable under several names - its headline and
    its `Page:` title, for instance - so two different matching names can
    each resolve, on their own, to the SAME record. Without dedup the top
    two can both be that one record, which compares it to itself, reads as
    a ratio of 1.0, and looks maximally ambiguous. Measured on emma frost:
    1.0 before dedup, 30.0 after - keep the best rank per doc id, then
    compare the top two DISTINCT records.
    """
    if not names:
        return float("inf")
    key = query_key(query) or norm(query)
    if not key:
        return float("inf")
    terms = set(key)
    # Same unknown-word guard as resolve() and rivals(), and the same
    # `known_words` escape hatch resolve() has: it should be the CORPUS
    # vocabulary in production, not the name vocabulary. infer/engine.py's
    # wants_choice() passes `index.postings` here, same as resolve() gets
    # from resolved_doc(). Judged against names alone, a real corpus word
    # that names nobody wrongly evicted every tier-1 candidate for the
    # contested name, which can only push the ratio UP - silently
    # suppressing the picker. Measured 2026-09-02: "who is venom symbiotic" -
    # "symbiotic" is in no name but is in the corpus - read `inf` against the
    # name vocabulary and 1.54 (same as bare "venom") against the corpus one.
    known = vocabulary(names) if known_words is None else known_words
    best_by_doc = {}
    for name in candidates(names, terms):
        name_set = set(name)
        if name == key:
            tier = 2
        elif name_set <= terms:
            tier = 1
        elif terms <= name_set and len(name) - len(key) <= EXTRA_TOKENS:
            tier = 0
        else:
            continue
        if tier == 1 and any(t not in known for t in terms - name_set):
            continue
        for doc_id, size, main, primary, _page, famous in names[name]:
            # Prose is NOT excluded here, and that was tried and reverted on
            # 2026-09-02. Skipping page-less records looks principled - an
            # article about someone is not a rival to them - but it removes
            # the LARGEST candidate from the comparison, leaving two smaller
            # Fandom records as the top two and LOWERING the ratio. It made
            # `who is spider-man 2099` offer a menu, which is the one case
            # the spec names as the floor on directness.
            cand = rank(main, primary, size, tier, doc_id, famous)
            prev = best_by_doc.get(doc_id)
            if prev is None or cand > prev[0]:
                best_by_doc[doc_id] = (cand, size)
    if len(best_by_doc) < 2:
        return float("inf")
    (_, best_size), (_, second_size) = sorted(
        best_by_doc.values(), key=lambda cand_size: cand_size[0], reverse=True
    )[:2]
    return best_size / second_size if second_size else float("inf")
