"""Turn a typed phrase into a row of the menu that was just shown.

`tell me about the ultimate one` is the measured failure that motivated the
whole picker, and its referent is a ROW ON SCREEN - not an entry in
conversation history. This module is the whole of that decision.

Kept free of `retrieve/` on purpose: it sees a line and a list of tuples, so
its tests need no index and can cover the edge cases exhaustively.

Nothing here scores or ranks. A matcher names exactly one row or declines,
because a silent tie-break is a guess, and refusing to guess is the rule
`resolve.py` already follows when a query names nobody.
"""
import re

# The reality written into a headline: `Spider-Man (Earth-1610)`. Kept
# index-free (no import from retrieve/resolve.py) rather than reusing its
# REALITY_SUFFIX - and NOT a copy of it either: REALITY_SUFFIX has no capture
# group, so the two cannot be identical. Deliberately STRICTER: it requires
# the literal hyphen after Earth/Reality, and the capture group ([\w]+)
# cannot span a second hyphen. Measured over all 202,101 headlines, the two
# disagree on exactly 10: two bare "(Earth)" suffixes with no hyphen at all
# (Oceania, Temple of Apocalypse) and eight multi-hyphen realities - Scarlet
# Witch (Earth-256656-Aleph), Kang the Conqueror (Earth-Mesozoic-24), and six
# "(Earth-TRN924) (Earth-Earth-TRN924)" doubled suffixes. reality_of() falls
# back to "616" for all ten.
REALITY = re.compile(r"\((?:Earth|Reality)-([\w]+)\)\s*$", re.I)

WORD = re.compile(r"[a-z0-9][a-z0-9-]*")

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
}

# Words that carry no identity of their own. Matchers 2 and 3 name a row by
# ONE token - an ordinal, a nickname, a literal reality - and every other
# word in the line must come from this set, or the line is a question with a
# row-shaped word sitting in it: "who is the ultimate hulk" must not resolve
# to the Ultimate row just because "ultimate" is in it, with "hulk" left
# over unexamined. FINDING 2026-09-04: with a Spider-Man menu open,
# "who is the ultimate hulk" and "what is the main difference" were both
# silently answered from the wrong row - worse than the swallow this module
# fixed, because a wrong answer looks like a right one and a swallow does
# not. Matcher 1 (a bare integer) is untouched: a line that is nothing but
# digits has no leftover to examine.
SCAFFOLDING = frozenset({
    "the", "a", "an", "one", "ones", "that", "this", "these", "those", "it",
    "tell", "me", "my", "about", "show", "give", "please", "what", "which",
    "who", "is", "are", "was", "were", "be", "of", "for", "from", "with",
    "and", "or", "to", "in", "on", "at", "pick", "choose", "select", "take",
    "see", "look", "more", "detail", "details", "info", "information",
    "version", "versions", "variant", "variants", "record", "entry", "row",
    "option",
})


def _names_nothing_extra(words, matched: set) -> bool:
    """True when every word except the ones in `matched` is scaffolding.

    A hard yes/no, not a score: one content word left over is enough to
    decline. Guessing which leftover word "matters more" is exactly the
    kind of ranking this module refuses to do.
    """
    return all(w in SCAFFOLDING for i, w in enumerate(words) if i not in matched)


# A universe by the name people call it, mapped to its bare reality number.
#
# Each nickname maps to EXACTLY ONE reality on purpose. The live spider-man
# menu carries both Earth-1610 and Earth-1610B, so a nickname matching both
# would match two rows, decline, and leave `the ultimate one` as broken as it
# is today. Anything more specific is typed literally - `the earth-1610b one`
# goes through the literal branch below.
#
# Hand-written because Category:Realities was never crawled: no curated file
# carries a `Page: Earth-1610` record. `infer/test_reference_corpus.py` pins
# every value to a reality that actually appears in the index, so this table
# is falsifiable rather than a constant nobody can challenge.
REALITY_NICKNAMES = {
    "ultimate": "1610",
    "mcu": "199999",
    "movie": "199999",
    "main": "616",
    "mainstream": "616",
    "prime": "616",
    "classic": "616",
    "noir": "90214",
    "zombie": "2149",
    "zombies": "2149",
    "lego": "13122",
    "2099": "928",
}


def reality_of(headline: str) -> str:
    """The bare reality number a headline carries."""
    m = REALITY.search(headline)
    if m:
        return m.group(1).lower()
    # An unmarked headline is Earth-616. The corpus writes the main
    # continuity without a suffix and marks every other reality - a curation
    # decision recorded in CLAUDE.md, not an assumption invented here.
    return "616"


def pick_row(line: str, rows):
    """The index of the one row `line` names, or None.

    The FIRST matcher that matches any row decides; matchers are never
    combined and a later one never rescues an earlier one. Within a matcher,
    exactly one row wins or the answer is None.
    """
    if not rows:
        return None
    words = WORD.findall(line.lower())
    if not words:
        return None

    # 1. A bare number, and only a bare number, is a row number. This is also
    # what keeps `2099` a question rather than a reality nickname: a line
    # that is nothing but digits never reaches matcher 3.
    if len(words) == 1 and words[0].isdigit():
        n = int(words[0])
        return n - 1 if 1 <= n <= len(rows) else None

    # 2. An ordinal names a row by position - but only when every other word
    # is scaffolding. "give me the second book" must decline, not pick row
    # 2 with "book" left unexamined; once an ordinal word is found this
    # matcher still commits, same as before - it just commits to None
    # instead of a guess when something is left over.
    for i, w in enumerate(words):
        if w in ORDINALS:
            if not _names_nothing_extra(words, {i}):
                return None
            n = ORDINALS[w]
            return n - 1 if 1 <= n <= len(rows) else None

    # 3. A reality, by nickname or written out - same guard. "what is the
    # main difference" must decline rather than silently pick the Earth-616
    # row: "main" names a reality, "difference" names nothing.
    wanted = set()
    matched = set()
    for i, w in enumerate(words):
        m = re.fullmatch(r"(?:earth|reality)-?([a-z0-9]+)", w)
        if m:
            wanted.add(m.group(1))
            matched.add(i)
        elif w in REALITY_NICKNAMES:
            wanted.add(REALITY_NICKNAMES[w])
            matched.add(i)
    if not wanted:
        return None
    if not _names_nothing_extra(words, matched):
        return None
    hits = [i for i, row in enumerate(rows) if reality_of(row[1]) in wanted]
    return hits[0] if len(hits) == 1 else None
