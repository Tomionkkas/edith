"""Typo recovery: propose corrected query keys. Phase 4.13.

**This module never chooses a record and never chooses a word.** It proposes
candidate keys; `resolve.rank()` disposes. That split is the whole phase.

`difflib` ranks by string similarity, which has no idea who is famous:

    spidermmn -> ['spidermen', 'spiderman', 'spiderm']      wrong first
    magnteo   -> ['mantego', 'magnuto', 'magneto']          wrong THIRD
    blck      -> ['block', 'bleck', 'black']                wrong first

Taking the top match answers Magneto questions with Mantego. Handing all
three to the ordering that already knows record size, main continuity,
provenance and notability gets Magneto, and it is not close.

Nothing here imports `resolve`: it would be a cycle, and a pure module is
testable without a 521 MB index. The caller passes the vocabulary and the
noise list in, which is how the rest of this codebase wires modules together.

Measured baseline this exists to move: 280 single-edit typos over the 40
flagships, 12 survived (4.3%), 249 resolved to None (88.9%). See
docs/superpowers/specs/2026-09-18-typo-recovery-design.md.
"""
from __future__ import annotations

import difflib
from itertools import combinations, product

# How close a token must be to count as a misspelling of it. Both swept over
# the 280-typo set; the tables are in ROADMAP 4.13.
NOISE_CUTOFF = 0.74     # against the ~120 question words
NAME_CUTOFF = 0.80      # against the 53,912-token name vocabulary

# Bounds on how much of a query may be wrong before it stops being a typo.
# Three misspelled tokens is a sentence, not a slip, and the candidate set is
# the PRODUCT of each token's matches - uncapped it stops being bounded.
MAX_CORRECTED = 2
MAX_PER_TOKEN = 8


def drop_noise(key: tuple, noise, vocab=frozenset()) -> tuple:
    """Remove tokens that are misspelled question words.

    `query_key` strips QUERY_NOISE by exact match, so one typo in `created`
    leaves ("crrreated", "thor"), which is a subset of no name and is then
    refused outright by resolve()'s unknown-word guard. 23% of typo damage
    is this, with the character typed perfectly.

    Only ever DELETES, so it cannot name the wrong character - the failure
    mode it does have is deleting a real name, which `vocab` prevents: a
    token the name index knows is a name, whatever else it resembles.

    Returns the key unchanged rather than empty. resolve() reads an empty key
    as "names nobody" and refuses, which would turn a typo into a refusal -
    the exact outcome this phase exists to remove.
    """
    kept = tuple(t for t in key
                 if t in vocab
                 or not difflib.get_close_matches(t, noise, n=1,
                                                  cutoff=NOISE_CUTOFF))
    return kept or key


def corrections(vocab, key: tuple):
    """Yield candidate keys with unknown tokens replaced by close names.

    `vocab` must be an ordered sequence, not a set: `difflib` breaks ties by
    input order, and a set's iteration order varies with PYTHONHASHSEED. A
    harness whose score moves between runs measures nothing.

    Yields nothing when every token is known - a correctly spelled query must
    never enter this code - and nothing when more than MAX_CORRECTED tokens
    are unknown.
    """
    unknown = [i for i, t in enumerate(key) if t not in vocab]
    if not unknown or len(unknown) > MAX_CORRECTED:
        return
    matches = {i: difflib.get_close_matches(key[i], vocab, n=MAX_PER_TOKEN,
                                            cutoff=NAME_CUTOFF)
               for i in unknown}
    if not all(matches.values()):
        return
    # Fewest substitutions first: a correction that changes one token is a
    # likelier reading than one that changes two, and resolve() keeps the
    # best-RANKED candidate, not the first, so order is a tie-break only.
    for size in range(1, len(unknown) + 1):
        for positions in combinations(unknown, size):
            for picks in product(*(matches[i] for i in positions)):
                out = list(key)
                for i, word in zip(positions, picks):
                    out[i] = word
                yield tuple(out)


def respell(question: str, key: tuple, corrected, norm, noise=()) -> str:
    """The question with its misspelled entity words fixed.

    Retrieval being right is not enough: `build_prompt` sends the QUESTION to
    the model, so an uncorrected one produced "I couldn't find Deadpol"
    printed directly beneath "reading that as Deadpool". Found by piping a
    session, not by a unit test - the answer text is downstream of everything
    the resolver's own tests can see.

    Matched by normalised WORD rather than by position: `drop_noise` can
    shorten the key, after which index i of one tuple is not index i of the
    other. Only words that actually changed are touched; rewriting the rest
    would put the model's words into the user's question.
    """
    if not corrected:
        return question
    if len(key) == len(corrected):
        fixed = {was: now for was, now in zip(key, corrected) if was != now}
    else:
        # `drop_noise` shortened the key, so index i of one tuple is not
        # index i of the other and zip would pair `crrreated` with `magneto`.
        # Align on what actually differs instead.
        spare = [k for k in key if k not in corrected]
        news = [c for c in corrected if c not in key]
        # Align each correction to the token it most resembles, NOT by
        # position. Position assumes the dropped noise word came first, which
        # "who created thor" makes look safe and "thor varriants" breaks:
        # the positional guess maps `varriants` -> `thor`.
        fixed = {}
        for now in news:
            was = max(spare, key=lambda s: difflib.SequenceMatcher(
                None, s, now).ratio(), default=None)
            if was is not None:
                fixed[was] = now
                spare.remove(was)
        # What is left was DROPPED as misspelled scaffolding. The key does
        # not want it; the QUESTION does, spelled right - deleting it leaves
        # the model reading "who thor", which is not an improvement on "who
        # crrreated thor". drop_noise already applied the vocabulary guard,
        # so these tokens are known not to be names.
        for token in spare:
            match = difflib.get_close_matches(token, noise, n=1,
                                              cutoff=NOISE_CUTOFF)
            if match:
                fixed[token] = match[0]
    if not fixed:
        return question
    out = []
    for word in question.split():
        tokens = norm(word)
        out.append(fixed[tokens[0]] if len(tokens) == 1 and tokens[0] in fixed
                   else word)
    return " ".join(out)

