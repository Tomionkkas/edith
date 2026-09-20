"""Select a verbatim passage from a record's prose. Phase 4.5b.

**Nothing here writes a sentence.** It chooses whole sentences already in the
record and hands them back unchanged. That is the phase's safety property:

    A verbatim quote can be the WRONG passage; it can never be a FALSE one.

It is also the only thing that works. A spike on 2026-09-20 passed the model
957 characters of Civil War narrative through `context.build_context(...,
histories=...)` - the same argument stage 3 trained on - and it still answered
"Civil War is from Earth-616." A 250M model does not use prose it is handed,
so a deterministic selector is not the cautious option, it is the available
one.

Scoped to ONE record. Retrieval already answers "which record" at 97.5%; what
is missing is "which sentence", over text already in hand. No second index, no
extra download, no corpus-wide passage search.

Nothing here imports the index or the resolver - `query_key` and `norm` are
passed in, so this module is testable against a string.
"""
from __future__ import annotations

import re

# Where one unit of prose ends: a full stop, OR the "; " that curation used to
# join a list-valued narrative field back together. The join is why a single
# "sentence" can otherwise run
#
#     ...a Civil War ensued.; Background and Casus Belli; Mutant Registration
#     Act, Secret War, Hulk; The Superhuman Registration Act had been...
#
# and swallow two section headings whole. Splitting on it recovers the lines
# curation collapsed. Measured in ROADMAP 4.14 and deliberately left in the
# corpus there - fixing it at the source means re-curating characters.txt and
# re-uploading 296 MB, where fixing it here costs one alternation.
SEGMENT = re.compile(r"(?<=[.!?])\s+|;\s+")

# A segment shorter than this is a fragment, a heading, or debris like
# "Brief Summary" (13), "Background and Casus Belli" (26) or "===Early
# Life===" (16) - never an answer on its own. Splitting on "; " is what turns
# those from mid-sentence debris into their own short segments, and this is
# what then drops them. The two rules only work together.
MIN_SENTENCE = 40

# How many sentences a passage may carry. Three is the 4.5b figure; measured
# against the 2,400-character context budget it is ~16% of it.
MAX_SENTENCES = 3

# A sentence must score at least this share of the best one to be quoted
# beside it. 4.5b asks for strict selection - "below threshold, no passage" -
# and scoring without a floor is not that: on "how did nitro kill the new
# warriors" the one sentence about Nitro scored highest, then two sentences
# matching only `kill` filled the remaining slots and, re-sorted into document
# order, were printed AHEAD of it. A passage of three sentences where two are
# padding reads as a miss even when the answer is in there.
RELATIVE_FLOOR = 0.5

NARRATIVE = ("History:", "Synopsis:")

# Wiki boilerplate: a sentence about the ARTICLE rather than its subject.
# Quoting it answers a question with a cross-reference. Only 44 of 98,214
# character records open this way (0.04%) - but they are Tony Stark, Ben Grimm
# and Peter Parker, which is to say the ones people ask about. One fixed
# template, not a list of special-cased records.
BOILERPLATE = re.compile(
    r"abridged version|expanded history|for a complete history", re.I)


def prose(record: str) -> str:
    """The narrative section of a record, or "" when it has none."""
    for label in NARRATIVE:
        _, sep, body = record.partition("\n" + label)
        if sep:
            return body.strip()
    return ""


def sentences(text: str) -> list:
    """Whole sentences, fragments and headings dropped."""
    return [s.strip() for s in SEGMENT.split(text)
            if len(s.strip()) >= MIN_SENTENCE
            and not BOILERPLATE.search(s)]


def discriminating(key: tuple, record: str, norm) -> frozenset:
    """Question terms that are not the record's own name.

    `what happened in the civil war` keys to ("civil", "war") once the
    question scaffolding is stripped - and those are the name of the record
    being read, present in most of its sentences. Scoring on them ranks by
    sentence length, not by relevance, so they are removed. What is left is
    what the question is actually asking about.
    """
    headline = record.split("\n", 1)[0]
    own = set(norm(headline))
    page = re.search(r"^Page:\s*(.+)$", record, re.M)
    if page:
        own |= set(norm(page.group(1)))
    return frozenset(t for t in key if t not in own)


def select(record: str, question: str, query_key, norm,
           max_sentences: int = MAX_SENTENCES):
    """Up to `max_sentences` whole sentences of `record`, or None.

    Two modes, and which one applies is decided by the question, not tuned:

    **Specific ask** - terms remain after the record's own name is removed
    ("how did NITRO DIE"). Sentences are scored by how many of those terms
    they carry, and a passage is returned only if something matched. Nothing
    matching means the record does not address the question, and the opening
    would be a non-answer dressed as one.

    **Bare narrative ask** - nothing remains ("what happened in the civil
    war"). The opening sentences are returned, because a narrative's opening
    IS its summary. This is the case 4.5b's "require real overlap" rule could
    not express: there is no overlap to require.
    """
    body = prose(record)
    if not body:
        return None
    found = sentences(body)
    if not found:
        return None

    terms = discriminating(query_key(question) or (), record, norm)
    if not terms:
        return " ".join(found[:max_sentences])

    # Rarity, measured WITHIN this record. 4.5b asks for overlap with the
    # question's RARE terms, and an unweighted count is not that: asked "how
    # did nitro kill the new warriors", `kill` appears all over a 42,989-
    # character record and `nitro` almost nowhere, so counting hits equally
    # returned "Chord attempted to kill himself." A term carried by most of a
    # record's sentences cannot discriminate between them, whatever the wider
    # corpus thinks of it - so this needs no index and no df table.
    tokens = [set(norm(s)) for s in found]
    weight = {}
    for term in terms:
        seen = sum(1 for tok in tokens if term in tok)
        weight[term] = 1.0 / seen if seen else 0.0

    scored = []
    for i, sentence in enumerate(found):
        score = sum(weight[term] for term in terms & tokens[i])
        if score:
            # `i` keeps the record's own order among equal scores: earlier
            # prose is nearer the summary, and a stable key stops the answer
            # changing between runs.
            scored.append((-score, i, sentence))
    if not scored:
        return None
    scored.sort()
    floor = -scored[0][0] * RELATIVE_FLOOR
    keep = [s for s in scored[:max_sentences] if -s[0] >= floor]
    return " ".join(s[2] for s in sorted(keep, key=lambda s: s[1]))
