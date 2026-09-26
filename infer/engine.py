"""Question -> retrieval -> prompt -> answer.

The prompt is built by the SAME functions that built the training data
(`search.format_context` and `sft_data.format_example`), not reimplemented.
A prompt that differs from training by even a separator makes a good model
look broken, and the difference is invisible in any metric.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _normalise(text: str) -> str:
    """Lowercase, collapse spaces, drop trailing punctuation.

    Training saw "who are you?"; a user typing "who are you" got a different
    string and fell through to the "not in my sources" template. 18,000
    chit-chat examples across 24 exact prompts is ~750 repeats each, which
    memorises strings rather than teaching the category.
    """
    return " ".join(text.lower().split()).strip(" ?!.,")


# Short openers that are conversation, not lookups. Retrieval matches "hi" to
# the character Hi-Lite with score 60.1 - confidently, and wrongly - so a
# score threshold cannot separate these.
SMALL_TALK = frozenset("""
hi hello hey yo hiya sup heya greetings
thanks thanx cheers ta
bye goodbye cya later
ok okay cool nice sure yes no
help
""".split())


def is_chitchat(question: str, sft) -> bool:
    """True when the question should be answered with an EMPTY context.

    Chit-chat was trained with no context block. Retrieval fires on greetings
    anyway, so "hi" arrived with a Marvel record attached - a pairing the
    model never saw in training, and it degenerated into a repetition loop.

    The reference list is sft_data.CHITCHAT itself, so the router cannot drift
    from what the model was actually taught.
    """
    q = _normalise(question)
    if not q:
        return True
    if q in {_normalise(u) for u, _ in sft.CHITCHAT}:
        return True
    words = q.split()
    return len(words) <= 3 and all(w in SMALL_TALK for w in words)


def tidy(answer: str, sft) -> str:
    """Repair what the training data taught, for the checkpoint we have.

    Stage 3's dataset carried 32,345 doubled full stops ("...are unknown..")
    and 2,621 "They works", and the model reproduces both faithfully.
    sft_data.emit() stops the next dataset carrying them; until stage 3 is
    retrained this repairs them on the way out, with the same two functions
    that clean the data - not a second implementation that could disagree.
    """
    return sft.one_terminator(sft.agree(answer))


def top_record(question: str, index, disambiguate, k: int = 3):
    """The record a factual answer should be read from, or None."""
    cands = disambiguate.candidates(index, question, k=k)
    return index.text(cands[0][0]) if cands else None


def display_name(text: str, resolve) -> str:
    """The entity words of a query, for a refusal message.

    Without this the refusal echoed the whole question: "I don't have who is
    Blorptron the Unmaker in my sources."
    """
    if resolve is None:
        return text.strip()
    kept = [w for w in text.split()
            if resolve.norm(w) and resolve.norm(w)[0] not in resolve.QUERY_NOISE]
    return " ".join(kept) or text.strip()


# The first sentence is load-bearing: run_cases, answer_cases and
# test_engine all assert on it. Only the second sentence is new - an error
# should say what to do next, not only what failed.
REFUSAL = ("I don't have {name} in my sources. "
           "Try a codename, a real name, or a reality — "
           "spider-man 2099, ororo munroe, earth-1610.")


class Plan:
    """What answering this question needs.

    `text` is an answer that is already final - rendered from a record, or a
    refusal. `prompt` is what the model should continue. `doc_id` is the
    record the answer stands on: the terminal counts it as the one source
    caught, reads `rows` out of it, and carries it into the next question as
    the entity a follow-up belongs to. None means nothing was retrieved and
    nothing may be claimed. (It used to be what the sources box drew; 4.12
    replaced that box with the field rows and this sentence outlived it.)
    """

    __slots__ = ("text", "prompt", "doc_id", "chitchat", "choices", "rows",
                 "corrected", "quoted", "quoted_leads")

    def __init__(self, text=None, prompt=None, doc_id=None,
                 chitchat=False, choices=None, rows=None, corrected=None,
                 quoted=None, quoted_leads=False):
        self.text, self.prompt, self.doc_id = text, prompt, doc_id
        self.chitchat = chitchat
        # The corrected query key when 4.13's typo fallback answered this,
        # else None. The terminal turns it into one line above the answer;
        # the KEY is kept rather than a rendered string so the caller decides
        # how to say it, and says it with the record's headline.
        self.corrected = corrected
        # A verbatim passage from `doc_id`'s prose, or None (4.5b). Appended
        # by the caller AFTER the answer, never substituted for it, so a
        # wrong passage is noise rather than error. It is quoted, not
        # generated - the model never sees it and never rewrites it.
        self.quoted = quoted
        # True when `quoted` should be shown INSTEAD of the answer rather
        # than after it - see passage_leads().
        self.quoted_leads = quoted_leads
        # Records to offer instead of answering. When this is set, `text` and
        # `prompt` are both None: a plan that offers a choice does not also
        # answer. The terminal decides how to show it; deciding WHETHER is
        # here, because confidence is a retrieval question.
        self.choices = choices
        # The record's fields as ordered (label, value) pairs. Present only
        # when the question wants the WHOLE entity. A field question ("who
        # created X") gets one sentence and no rows: dumping a profile at
        # someone who asked for one fact is a different answer to the one
        # they asked for.
        self.rows = rows


def resolved_doc(question: str, index, facts, resolve, previous=None,
                 settled=None):
    """The record a question resolves to, or None. ~40 callers read this.

    A thin wrapper over `resolved_doc_spelled` so the two cannot drift.
    """
    return resolved_doc_spelled(question, index, facts, resolve, previous,
                                settled)[0]


def resolved_doc_spelled(question: str, index, facts, resolve, previous=None,
                         settled=None):
    """`(doc_id, corrected_key)` - the record, and any typo it had to fix.

    `corrected_key` is None unless 4.13's fallback was used. Only the direct
    resolution can carry one: a follow-up inherits `previous` and a settled
    pick is the user's own choice, and announcing "reading that as" over
    either would be describing a correction that did not happen.

    None from a LOADED name index means the corpus does not have this
    character - a refusal, not a reason to fall back to document scoring.

    `previous` is the record the last question resolved to. A question that
    names nobody once its intent words are removed - "more details about his
    powers", "and his real name?" - is a follow-up, and belongs to whoever was
    already being discussed. Without this the leftover word "powers" resolved
    to a character called The Power. It is deliberately NOT a conversation
    memory: the model saw only single-turn examples in training, so what
    carries over is the retrieved record, not the dialogue.
    """
    if resolve is None or facts is None or not (names := _names(resolve)):
        return None, None
    probe = facts.entity_text(question)
    key = resolve.query_key(probe)
    if not key:
        # Nothing here names anyone. resolve() would fall back to the
        # unstripped query and match on the scaffolding itself - "more
        # details about his powers" found a character called More. Carry the
        # previous record if there is one, and otherwise admit we do not know
        # who is meant. The cost is that "who is him" no longer reaches Adam
        # Warlock, whose alias is literally "Him"; guessing was worse.
        return previous, None
    # A name the user explicitly picked answers from that record for the
    # rest of the session. Checked HERE, not only in plan(), because
    # try_facts() and build_prompt() each resolve again to build the text -
    # overriding plan()'s local doc_id alone changed the trace and left the
    # answer coming from the record the user did not pick.
    if settled and key in settled:
        return settled[key], None
    return resolve.resolve_spelled(names, probe, index.postings)


# How close the top two records must be before EDITH asks instead of
# answering. `resolve.confidence()` is the size ratio of the top-ranked
# record to the runner-up - NOT bounded below by 1.0, since rank chooses the
# "top" one, not size (see resolve.confidence()'s docstring). A large number
# means the top record dominates in size; a small one, in either direction,
# means the top two are close and a reader would want a choice.
#
# MEASURED, not chosen, and not restated here - restating invites drift
# between this comment and the evidence. The table that chose 3.0, and the
# means to reproduce or challenge it, is one command:
#
#     PYTHONHASHSEED=0 py infer/evaluate.py --n 250 --picker
#
# Its representative-set arm sweeps candidate thresholds against the sets
# that represent real questions - the 40 flagships, and the answer-cases in
# infer/answer_cases.py that must and must not offer - rather than the
# corpus-wide rate, which is misleading alone: probes are drawn uniformly
# from 201,815 records, most of them obscure variants where two 400-character
# stubs give a low RATIO without any ambiguity a reader would care about. The
# ACTUAL output of that command, and the reasoning for 3.0 over its
# neighbours, is transcribed in ROADMAP.md's Phase 4.7 entry - copy it there
# again after any change here, rather than editing this comment.
CONFIDENCE_TO_ASK = 3.0


def wants_choice(question, index, facts, resolve, doc_id, settled=None) -> bool:
    """Whether to offer a choice rather than answer.

    A broad ask - "show me variants of X" - always offers: that request IS a
    request to be offered the choice. A specific ask offers only when the top
    two records are close enough that answering would be a guess.

    The confidence is measured on `facts.entity_text(question)`, NOT the raw
    question, because that is the string `resolved_doc()` actually resolves.
    Measuring the raw one scores something the reader will never be answered
    from: "what powers does moon knight have" reads 5.9 raw and 38.7 stripped,
    so it would interrupt an ordinary field question with a menu.
    """
    if doc_id is None or resolve is None:
        return False
    if facts.wants_variants(question):
        return True
    # Computed once and reused below - resolved_doc() de-duplicated the same
    # call for the same reason.
    probe = facts.entity_text(question)
    # BELOW the broad ask, deliberately. A pick already answered this
    # question for this name in this session, so an ordinary ask should not
    # ask it again - but `show me the variants of X` is the user explicitly
    # asking to see the list, and settling must not override that.
    # Keyed on the name and not global: one pick must not silence every
    # later ambiguity.
    if settled and resolve.query_key(probe) in settled:
        return False
    # A question that names nobody is a follow-up - "what are his powers",
    # and the "who is this" the terminal asks itself after a pick. The record
    # is already chosen and `resolved_doc` carried it here; there is nothing
    # to disambiguate. Without this guard confidence() scored the leftover
    # scaffolding, came back low, and re-opened the picker on the turn
    # immediately after you had used it - found by reading a session.
    if not resolve.query_key(probe):
        return False
    # index.postings is the corpus vocabulary, same as resolved_doc() passes
    # resolve.resolve() (see resolve.resolve's docstring at resolve.py:509).
    # FINDING 2026-09-02: confidence()'s tier-1 unknown-word guard used to
    # read the NAME vocabulary instead, on its own - `index` was already a
    # parameter here and simply went unread. A query word that names nobody
    # but IS a real corpus word then wrongly evicted every tier-1 candidate,
    # which can only push the ratio UP and silently suppress the picker.
    return (resolve.confidence(_names(resolve), probe, index.postings)
            < CONFIDENCE_TO_ASK)


def plan(question: str, index, sft, search, disambiguate, facts,
         resolve=None, k: int = 3, previous=None, settled=None,
         shown=None) -> Plan:
    """The whole decision, in the form a caller can render.

    Wraps try_facts and build_prompt rather than reimplementing them, so the
    terminal cannot drift from what the tests cover.
    """
    # "hi" is not a lookup. It resolved to Hi-Vo (Earth-616) and the terminal
    # drew a full character card beside a greeting - the clearest case of the
    # thing this system must not do, and no unit test saw it because they all
    # called try_facts directly rather than talking to it.
    small_talk = is_chitchat(question, sft)
    doc_id, corrected = ((None, None) if small_talk
                         else resolved_doc_spelled(question, index, facts,
                                                   resolve, previous, settled))
    # Retrieval being right is not enough. `build_prompt` sends the QUESTION
    # to the model, so leaving the typo in it printed "I couldn't find
    # Deadpol" directly beneath "reading that as Deadpool". Rewrite once,
    # here, so try_facts and build_prompt both see the corrected wording -
    # fixing it in either alone would leave the other saying the typo.
    if corrected is not None:
        question = resolve.spell.respell(
            question, resolve.query_key(facts.entity_text(question)),
            corrected, resolve.norm, resolve.QUERY_NOISE)
    # Before try_facts, not after. facts.variants() walks all 201,815
    # headlines - 636 ms measured - and try_facts walks them again to build a
    # sentence this branch then throws away. Asking first pays that once.
    if wants_choice(question, index, facts, resolve, doc_id, settled):
        rows, total = facts.variants(index, resolve, doc_id)
        # Gate on len(rows), the DEDUPED list offer() actually draws the menu
        # from - not `total`, the raw count before facts.variants() collapses
        # same-headlined records into one row. FINDING 2026-09-02 (Minor): a
        # name whose every contending record shares one exact headline
        # collapses to a single row while total stayed > 1, so this used to
        # offer anyway: "1 records could be this. Which?" - a menu of one.
        if len(rows) > 1:
            return Plan(doc_id=doc_id, chitchat=small_talk, choices=rows,
                        corrected=corrected)
    text = try_facts(question, index, sft, search, disambiguate, facts,
                     resolve, k=k, previous=previous, settled=settled)

    # Rows are needed by the passage trigger as well as by the answer, so
    # compute them once here rather than inside the branch that renders them.
    field_rows = None
    if (text is not None and doc_id is not None
            and facts.wants_whole_entity(question)):
        field_rows = facts.profile_rows(index.text(doc_id)) or None

    # Written by curation, read here - never inferred. `Kind:` is the one
    # field three modules read separately ON PURPOSE (CLAUDE.md); this is a
    # fourth READER, not a fourth copy of the rule.
    # getattr, because half the engine tests pass a resolve stub with only
    # the two functions they exercise - and a missing Kind must behave as
    # pre-Kind code did, which is the rule CLAUDE.md states for all three of
    # its readers.
    kind_of = getattr(resolve, "kind_of", None)
    kind = (kind_of(index.text(doc_id))
            if kind_of and doc_id is not None else None)

    quoted = None
    leads = False
    if doc_id is not None and wants_passage(question, text, field_rows, kind):
        psg = _load("passages", "infer/passages.py")
        # "go on" reads the NEXT sentences, not the opening again (4.17).
        skip = passage_skip(question, doc_id, shown,
                            lambda q: resolve.query_key(facts.entity_text(q)))
        quoted = psg.select(index.text(doc_id), question,
                            resolve.query_key, resolve.norm, skip=skip)
        if shown is not None and quoted:
            shown[doc_id] = skip + len(psg.sentences(quoted))
        leads = quoted is not None and passage_leads(question, text,
                                                     field_rows, kind)

    if text is not None:
        # `field_rows` is computed above, with the SAME predicate try_facts
        # uses to reach its profile branch - facts.wants_whole_entity(), one
        # expression in one module, not a copy of one. Two copies drifted
        # twice: gating on wants_profile() alone attached rows to field
        # questions (the two predicates overlap heavily), and adding
        # detect_intent() still left variants questions getting rows beside
        # the list try_facts had just rendered.
        return Plan(text=text, doc_id=doc_id, chitchat=small_talk,
                    rows=field_rows, corrected=corrected, quoted=quoted,
                    quoted_leads=leads)
    return Plan(prompt=build_prompt(question, index, sft, search, disambiguate,
                                    facts, resolve, k=k, previous=previous,
                                    settled=settled),
                doc_id=doc_id, chitchat=small_talk, corrected=corrected,
                quoted=quoted, quoted_leads=leads)


def try_facts(question: str, index, sft, search, disambiguate, facts,
              resolve=None, k: int = 3, previous=None, settled=None):
    """A rendered answer for a field question, or None to use the model.

    Code answers what a field answers. Asked who created Moon Knight, the
    record already holds "Created by: Doug Moench; Don Perlin" - restating
    that through a 254M model can only add error, and did: identical correct
    context produced a correct answer on one seed and an invented Earth-TRN1518
    on another.
    """
    if is_chitchat(question, sft):
        return None
    probe = facts.entity_text(question)

    # Resolve the CHARACTER first, from a name index, before any document
    # scoring. BM25 answers "which documents contain these words", which is a
    # different question: "what is storm skilled in" reached Of-Storm and
    # "what powers does venom have" reached Mary Jane's Venom.
    doc_id = None
    if resolve is not None and _names(resolve):
        # index.postings is the corpus vocabulary; see resolve.resolve.
        doc_id = resolved_doc(question, index, facts, resolve, previous,
                              settled)
        if doc_id is None:
            # The resolver knows every name in the corpus. If it cannot place
            # this one, we do not have the character - and saying so beats
            # both a BM25 partial match ("Zyxthaloraxian the Devourer" found
            # the real entity "Devourer") and letting the model generate, which
            # invented Blorptron the Unmaker complete with creators and a
            # first appearance. This refuses regardless of question shape,
            # because "tell me about X" hallucinates just as readily.
            return REFUSAL.format(name=display_name(probe, resolve))

    # A list question first: it names a character but does not ask about one.
    #
    # plan() no longer reaches this: wants_choice() fires first there and
    # returns a Plan carrying `choices` instead of calling try_facts's result
    # at all. This branch stays live because engine.main()'s CLI still calls
    # try_facts() directly - so `edith` offers a picker for "show me the
    # variants of spider-man" while `py infer/engine.py --ask "..."` prints
    # this sentence, and that divergence is deliberate, not an oversight.
    if doc_id is not None and facts.wants_variants(question):
        listed = facts.render_variants(index, resolve, doc_id)
        if listed:
            return listed

    if facts.detect_intent(question) is None:
        # "who is X" / "tell me about X" names no single field, but the record
        # answers it in full. The model used to write these and its prose could
        # contradict the sources box printed directly above it - asked who the
        # Hood is, it credited Paul Jenkins and Humberto Ramos while the record
        # said Brian K. Vaughan and Kyle Hotz. Composed from fields it cannot.
        # wants_whole_entity() rather than wants_profile(): the same one
        # expression plan() gates its rows on, so the two cannot disagree
        # about which questions this branch answers. It re-states the
        # variants condition the branch above already consumed, which costs
        # one regex and buys the binding - and it moves one case: a variants
        # question about a name with no siblings (render_variants returned
        # "") used to be answered with a profile of that record, and is now
        # left to the model, which still receives the record as context.
        if doc_id is not None and facts.wants_whole_entity(question):
            summary = facts.profile(index.text(doc_id),
                                    detail=bool(facts.DETAIL_RE.search(question)))
            if summary:
                return summary
        return None                      # genuinely open: the model

    if doc_id is not None:
        return facts.answer(question, index.text(doc_id))

    record = top_record(search.entity_query(index, probe), index, disambiguate, k=k)
    if record is None:
        return None
    return facts.answer(question, record)


_NAMES = None


def _names(resolve):
    """The name index, loaded once. Built by retrieve/build_names.py."""
    global _NAMES
    if _NAMES is None:
        _NAMES = resolve.load() or {}
    return _NAMES


# ---------------------------------------------------------- passages (4.5b)

# A field question has one right answer and a paragraph after it is worse.
# Checked FIRST, so "how was spider-man created" is still a credit line.
NEVER_QUOTE = re.compile(
    r"who (created|made|wrote|drew)|first appear|when did .* debut", re.I)

# The questions no field can answer.
NARRATIVE_Q = re.compile(
    r"what happened|what led to|how did|how was|how does|why did|why is|"
    r"why was|tell me the story|what was the story", re.I)

# Below this a field answer is thin enough that a passage cannot make it
# worse. 4.5b's figure.
THIN_TEXT = 80

# A record whose fields cannot answer "tell me about it". An event has no
# eye colour: its fields are reality, creators and a first appearance, all
# bibliography, so for these kinds the STORY is the profile. Written by
# curation and read with resolve.kind_of() - never inferred here.
NARRATIVE_KINDS = ("event", "story arc")

# A whole-entity question that yielded this many fields or fewer is thin
# whatever its sentence weighs. The trigger 4.5b lacked: `tell me about the
# civil war` renders 91 characters - over THIN_TEXT - out of ONE row, and
# says nothing. Length is the wrong proxy for a profile.
THIN_ROWS = 1


def passage_leads(question: str, text=None, rows=None, kind=None) -> bool:
    """Whether the passage should come BEFORE the answer rather than after.

    4.5b's append-never-substitute rule exists so a passage cannot make a
    CORRECT answer worse. On a narrative ask the field answer is not
    correct-but-incomplete - it answers a different question - and leading
    with it buries the one that was asked. Measured on a live run of 260
    questions, 2026-09-20: 69% of quoted turns opened with a bibliographic
    credit.

        what happened in American Revolutionary War
          "American Revolutionary War was created by Ken Bald."
          > The American Revolutionary War ... was the armed struggle in
            which the thirteen North American colonies rejected ...

    REVISED 2026-09-26, from the web app where the shape is unmissable. 4.5b
    held that a thin profile answer "is still an answer to what was asked"
    and kept its place. For an EVENT that answer reads:

        Civil War (Event) is from Earth-616. Civil War (Event) first
        appeared in Civil War Vol 1 1.

    which is the same bibliographic credit in a different hat. ONE row is
    not a thin answer to `tell me about the civil war`, it is the absence of
    one - and `THIN_ROWS` already names that exact question as the case it
    was added for. Two rows or more is a profile and still leads.

    The safety property is untouched either way: the quote is verbatim and
    the model never sees it.
    """
    if NEVER_QUOTE.search(question):
        return False
    if NARRATIVE_Q.search(question):
        return True
    if kind in NARRATIVE_KINDS and rows:
        return True
    return rows is not None and len(rows) <= THIN_ROWS


def passage_skip(question, doc_id, shown, query_key) -> int:
    """How many sentences of this record the reader has already been given.

    `shown` is to passages what `settled` is to the picker: a dict the CALLER
    owns, keyed by doc id, that the engine reads and writes. Both renderers
    keep one; neither needs any other state.

    Only a question that names NOBODY continues - "go on", "more please",
    "explain in more detail" all key to () since 4.17, while "what are his
    powers" keys to ("power",) and is a fresh question about the same record.
    A new subject, or a different record, starts from the opening.
    """
    if not shown or doc_id is None:
        return 0
    if query_key(question):
        return 0
    return shown.get(doc_id, 0)


def wants_passage(question: str, text, rows, kind=None) -> bool:
    """Whether to append a quoted passage to this answer.

    Taste, not safety: the quote is appended, so the worst case is an
    irrelevant paragraph after a correct answer. But a quote after every
    answer teaches the reader to skip it, which costs the feature its point.
    """
    if NEVER_QUOTE.search(question):
        return False
    if NARRATIVE_Q.search(question):
        return True
    if text is None:
        # The model is answering an ordinary question. Not a reason to quote.
        return False
    # An event's profile is a bibliography however many rows it runs to, so
    # row count cannot decide it. Measured 2026-09-26: `tell me about secret
    # wars` renders three rows, reads as a credit, and quoted nothing.
    if kind in NARRATIVE_KINDS and rows:
        return True
    if len(text) < THIN_TEXT:
        return True
    return rows is not None and len(rows) <= THIN_ROWS


def build_prompt(question: str, index, sft, search, disambiguate,
                 facts=None, resolve=None, k: int = 3, previous=None,
                 settled=None) -> str:
    """Exactly the string stage 3 was trained to continue.

    Built with split_prompt so it ends where the loss mask started - after
    "Assistant:" and before the space - rather than by hand-assembling a
    lookalike.
    """
    # Chit-chat carries no context: that is how it was trained, and sending
    # a retrieved record with a greeting is what caused the repetition loop.
    if is_chitchat(question, sft):
        return sft.split_prompt(sft.format_example("", question, "x"))[0]

    # Adaptive k. Three records exist so a wrong top-1 is survivable, but they
    # also give the model three chances to quote the wrong one: asked who
    # created Moon Knight, it took the 2023 creative team from record 2 while
    # Doug Moench and Don Perlin sat in record 1. When one candidate clearly
    # dominates there is no reason to offer decoys, so send only that record;
    # keep all three when the scores are genuinely close and the model should
    # choose.
    # The same two-stage resolution the fact path uses. Without it here, an
    # open-ended question fell back to BM25 and "tell me about daredevil in
    # detail" answered from an Earth-TRN1444 record.
    resolved = None
    unknown = False
    if resolve is not None and facts is not None and (names := _names(resolve)):
        resolved = resolved_doc(question, index, facts, resolve, previous,
                                settled)
        unknown = resolved is None

    if unknown:
        # An unplaceable name gets no context at all, so the model answers
        # from its "not in my sources" training rather than grounding on
        # whatever BM25 happened to surface.
        context = ""
    elif resolved is not None:
        ctx = _load("context", "retrieve/context.py")
        context = ctx.build_context([index.text(resolved)])
    elif not (cands := disambiguate.candidates(index, question, k=k)):
        context = ""
    else:
        # Build from the CHOSEN candidates, never from a fresh k=1 search:
        # candidates() promotes the flagship (Steve Rogers over Sam Wilson),
        # and re-searching would silently hand back the unpromoted top hit.
        chosen = cands[:1] if not disambiguate.is_ambiguous(cands) else cands[:k]
        ctx = _load("context", "retrieve/context.py")
        context = ctx.build_context([index.text(c[0]) for c in chosen])
    example = sft.format_example(context, question, "placeholder")
    prompt, _ = sft.split_prompt(example)
    return prompt


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k", type=int, default=3)
    # Answers run to 164 tokens before the model emits its own stop, so a
    # 140-token cap was cutting the longest ones mid-sentence.
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--no-facts", action="store_true",
                    help="skip the rendered path; make the model answer everything")
    ap.add_argument("questions", nargs="+")
    args = ap.parse_args()

    import torch
    import sentencepiece as spm
    search = _load("search", "retrieve/search.py")
    disambiguate = _load("disambiguate", "retrieve/disambiguate.py")
    facts = _load("facts", "infer/facts.py")
    resolve = _load("resolve", "retrieve/resolve.py")
    sft = _load("sft_data", "train/sft_data.py")
    sample = _load("sample", "train/sample.py")

    sp = spm.SentencePieceProcessor(
        model_file=str(ROOT / "tokenizer" / "marvel_bpe_50257.model"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, block, step, _ = sample.load_model(Path(args.ckpt), device)
    index = search.Index.load()
    print(f"{Path(args.ckpt).name}: step {step:,} | index {len(index):,} records\n")

    for q in args.questions:
        print("=" * 70)
        print(f"Q: {q}")

        rendered = None if args.no_facts else try_facts(
            q, index, sft, search, disambiguate, facts, resolve, k=args.k)
        if rendered is not None:
            print(f"A: {rendered}")
            print("   [from the record, not generated]")
            continue

        prompt = build_prompt(q, index, sft, search, disambiguate,
                              facts, resolve, k=args.k)
        if args.show_prompt:
            print("-" * 70)
            print(prompt[:600])
        answer = sample.generate(model, sp, device, prompt, block,
                                 max_new=args.max_new, temp=args.temp,
                                 top_k=50, top_p=0.95, repetition_penalty=1.1)
        print(f"A: {tidy(answer.strip(), sft)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
