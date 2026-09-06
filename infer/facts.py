"""Answer field questions from the record, in code, without the model.

Asked who created Moon Knight, the retrieved record already contains

    Created by: Doug Moench; Don Perlin

The answer is present and exact. Passing it through a 254M model to be
restated can only introduce error, and measurably does: the same question on
the same correct context produced "T'Challa, enhanced by the Heart-Shaped
Herb" on one seed and an invented "Earth-TRN1518" on another. There is nothing
creative about restating a field.

So code answers what a field answers, and the model is kept for what it is
actually good at - open-ended synthesis and conversation. Anything this module
does not recognise returns None and falls through to the model.

This does NOT make answers true. A wrong or missing record renders a wrong
fact with total confidence, exactly as the model did; Victor von Doom
(Earth-616) is absent from the corpus and no amount of determinism fixes that.
It removes the model as a source of error, not the corpus.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Question shapes, most specific first: "who created X" must beat "who is X".
# Each entry is (name, record field, pattern, template). {e} is the entity,
# {v} the humanised field value.
INTENTS = (
    ("creator", "Created by",
     r"\bwho\s+(created|made|wrote|drew|designed)\b|\bcreators?\s+of\b|"
     r"\bwho\s+(is|are)\s+the\s+creators?\b",
     "{e} was created by {v}."),
    ("first_appearance", "First appearance",
     # \w* because entity_text DELETES whatever matched: "his first
     # appearance" against a bare "first appear" left the fragment "ance"
     # behind, and the resolver then went hunting for a character called Ance.
     r"\bfirst\s+appear\w*|\bdebut\w*|\bwhich\s+(comic|issue|book)\b|"
     r"\bwhere\s+(do|should|can|would)\s+i\s+(start|begin)|\bstart\s+(with|reading)\b",
     "{e} first appeared in {v}."),
    ("real_name", "Full name",
     r"\breal\s+name\b|\bsecret\s+identity\b|\btrue\s+identity\b|"
     r"\bwho\s+is\s+.+\s+really\b|\bunder\s+the\s+(mask|hood|helmet)\b",
     "{e}'s real name is {v}."),
    ("powers", "Powers",
     # A bare \bpowers?\b matched "power" inside "Power Man", so "who is power
     # man" answered with Iron Man's power list. Require question context.
     r"\bwhat\s+powers?\b|\bpowers?\s+(?:does|do|has|have|of)\b|"
     # "explain in detail his powers" matched none of the above, so the
     # question kept the word "powers" as an entity term and resolved on it -
     # which is how a 1,986-char Golden Age stub answered for Parker Robbins.
     r"\b(?:his|her|its|their)\s+powers?\b|"
     r"\bwhat\s+can\s+.+\s+do\b|\bhow\s+strong\b",
     "{e}'s powers include {v}."),
    ("abilities", "Abilities",
     r"\babilit\w*|\bskill\w*|\btrained\s+in\b|\bgood\s+at\b",
     "{e} is skilled in {v}."),
    ("reality", "Reality",
     r"\bwhich\s+(earth|reality|continuity|universe)\b|"
     r"\bwhat\s+(earth|reality|continuity|universe)\b|\bwhere\s+is\s+.+\s+from\b",
     "{e} is from {v}."),
    ("species", "Species / origin",
     r"\bspecies\b|\bwhat\s+kind\s+of\b|\borigin\b",
     "{e} is {v}."),
    ("affiliation", "Affiliation",
     r"\bwhat\s+teams?\b|\bwhich\s+teams?\b|\baffiliat|"
     r"\bwho\s+does\s+.+\s+work\s+(with|for)\b",
     "{e} is affiliated with {v}."),
    ("occupation", "Occupation",
     r"\boccupation\b|\bfor\s+a\s+living\b|\bwhat.s\s+.+\s+job\b",
     "{e} works as {v}."),
    ("base", "Base of operations",
     r"\bwhere\s+(does|do)\s+.+\s+(live|operate|based)\b|\bbase\s+of\s+operations\b",
     "{e} is based in {v}."),
)

_SD = None


def _sd():
    """sft_data, for the phrasing helpers the training data was built with.

    Reusing humanize() and clean_value() rather than reimplementing them keeps
    a rendered answer indistinguishable from a generated one, which matters
    when both appear in the same terminal.
    """
    global _SD
    if _SD is None:
        spec = importlib.util.spec_from_file_location(
            "sft_data", ROOT / "train" / "sft_data.py")
        _SD = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("sft_data", _SD)
        spec.loader.exec_module(_SD)
    return _SD


def detect_intent(question: str):
    """(name, field, template) for the first shape that matches, else None."""
    q = question.lower()
    for name, field, pattern, template in INTENTS:
        if re.search(pattern, q):
            return name, field, template
    return None


def entity_text(question: str) -> str:
    """The question with its intent words removed, for retrieval.

    BM25 scores every term, so wording moves the ranking even when the entity
    is obvious: "what powers does storm have" found Ororo Munroe while "what
    is storm skilled in" found Of-Storm, and "what is storm good at" found
    Prince of Good. Document frequency alone cannot fix that, because "skilled"
    and "good" are real, reasonably rare terms. But once an intent is matched
    we know precisely which words formed the question, so they can go.
    """
    hit = None
    q = question
    for name, field, pattern, template in INTENTS:
        if re.search(pattern, q.lower()):
            hit = pattern
            break
    if hit is None:
        return question
    cleaned = re.sub(hit, " ", q, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    return cleaned or question


# "who is X", "tell me about X" - a shape the record can answer in full, with
# no field to single out. The model used to write these, and its prose could
# contradict the box printed directly above it: asked who the Hood is, it
# credited Paul Jenkins and Humberto Ramos while the record said Brian K.
# Vaughan and Kyle Hotz. Composing from fields cannot do that.
PROFILE_RE = re.compile(
    r"\bwho\s+(?:is|are|was|were)\b|\bwho.s\b|\bwhat\s+(?:is|are)\b|"
    r"\btell\s+me\s+about\b|\bdescribe\b|\bexplain\b|\brundown\b|"
    r"\bmore\s+about\b|\binfo(?:rmation)?\s+(?:on|about)\b", re.I)

DETAIL_RE = re.compile(r"\bin\s+detail\b|\bdetailed\b|\beverything\b|"
                       r"\bfull\s+(?:profile|story|details?)\b", re.I)


def wants_profile(question: str) -> bool:
    """True when the question asks who someone IS, rather than for a field."""
    return bool(PROFILE_RE.search(question))


# The fields that answer "who is X", in the order a reader wants them, with
# the labels a reader expects rather than the corpus's own. This is the same
# judgement render.BOX_FIELDS made for the sources box; it lives here now
# because the rows are the ANSWER, not chrome around one.
PROFILE_FIELDS = (
    ("Powers", "Powers"),
    ("Created by", "Created by"),
    ("First appearance", "First appearance"),
    ("Occupation", "Occupation"),
    ("Affiliation", "Teams"),
)
PROFILE_ROWS = 4


def _strip_citations(items: list) -> list:
    """Items with a bare comic citation dropped, unless that leaves nothing.

    unwrap_templates() can leave a citation standing as its own item -
    "Unbeatable Squirrel Girl Vol 2 40" opens Squirrel Girl's Powers. Both
    the field answer and the profile rows need this, and they are one module
    apart, so it lives here once rather than twice. (train/sft_data.py keeps
    its own copy: that one crosses an import boundary this cannot - see
    CLAUDE.md's note on the deliberate `kind_of()` triplication for the
    general rule.)
    """
    return [it for it in items if not _sd().CITATION.match(it)] or items


def profile_rows(record: str, limit: int = PROFILE_ROWS) -> list[tuple[str, str]]:
    """The record as ordered (label, value) pairs, or [] if it has none.

    An empty list is meaningful: a record with no usable field must fall
    back to prose rather than print an empty frame, which would claim more
    structure than we have.

    Every value goes through the SAME pipeline answer() applies, in the same
    order, because as of 4.12 these rows ARE the answer for a whole-entity
    question and the two functions must not disagree about the same field.
    That is one pipeline shared by two callers, not a third one: the steps
    are clean_value(), then _strip_citations() for the two narrative fields,
    then _first_sentence() on an item that is itself a paragraph. The rows
    are labelled rather than prose, so humanize() is what answer() adds and
    this does not.

    FINDING 2026-09-05 (whole-branch review): promoting the field block to
    the default answer promoted the less hygienic of the two functions.
    profile_rows() split the RAW value and filtered only Powers/Abilities
    citations, so the `Formerly:;` sub-labels clean_value() exists to remove
    - the exact artifact the phase's spec names - still rendered on Namor,
    the spec's own worked example, and an over-long Powers entry was cut
    mid-sentence by the renderer's line budget instead of at its own full
    stop. Measured over curated/characters.txt: clean_value() changes 10.9%
    of Occupation values, 16.2% of Affiliation, 6.1% of Powers.
    """
    rows = []
    for key, label in PROFILE_FIELDS:
        raw = field_value(record, key)
        if not raw:
            continue
        items = [p.strip() for p in _sd().clean_value(raw).split(";")
                 if p.strip()]
        if key in ("Powers", "Abilities"):
            # 4.12 REBASELINE regression: unwrap_templates() can leave a bare
            # comic citation standing as its own item ("Unbeatable Squirrel
            # Girl Vol 2 40" opens Squirrel Girl's Powers), and it reached the
            # screen verbatim once these rows became the answer - caught by
            # run_cases' corpus-hygiene case. Shared with answer() via
            # _strip_citations() rather than a second copy in this module.
            items = _strip_citations(items)
        value = "; ".join(item if len(item) <= ANSWER_CHARS
                          else _first_sentence(item) for item in items)
        if value:
            rows.append((label, value))
        if len(rows) == limit:
            break
    return rows


def profile(record: str, detail: bool = False):
    """A summary composed from the record's fields, or None.

    Built by the SAME function that writes the stage-3 dataset's open-ended
    answers, so what a user reads and what the model was trained on cannot
    drift apart - and every clause traces to a field.
    """
    pair = _sd().open_ended_pair(record, detail=detail)
    if not pair:
        # No field schema at all - a Wikipedia article like X-Men or Fantastic
        # Four, which is 38% of the corpus. There is nothing to compose from,
        # but the article's own opening says what the thing is, and quoting it
        # is grounded where letting the model improvise is not.
        return prose_summary(record, 520 if detail else 300)
    return _sd().one_terminator(_sd().agree(pair[1]))


def prose_summary(record: str, max_chars: int = 300):
    """The opening claim of a schema-less record, or None.

    The first lines are a subtitle and a disambiguation hint ("Marvel Comics
    superhero team of mutants", "the superhero team"); the article proper
    starts at the first line that is actually a sentence.
    """
    body = ""
    for line in record.split(chr(10))[1:]:

        line = line.strip()
        if len(line) > 60 and line[:1].isupper() and ". " in line + " ":
            body = line
            break
    if not body:
        return None
    out = []
    for piece in body.split(". "):
        if sum(len(x) + 2 for x in out) + len(piece) > max_chars and out:
            break
        out.append(piece.strip())
    text = ". ".join(out).rstrip(".") + "."
    return _sd().one_terminator(text)


# "what are Spider-Man's variants" is not a request for a character; it is a
# request for a LIST, and answering it with one character card is what made
# the terminal look like it turns every word into a character. 18 records are
# headlined exactly "Spider-Man" and the corpus knows all of them.
VARIANTS_RE = re.compile(
    r"\bvariants?\b|\bversions?\b|\bincarnations?\b|\bcounterparts?\b|"
    r"\bhow\s+many\s+\w+\s+(?:are|were|exist)\b|\bwhich\s+earths?\b|"
    r"\bother\s+\w*\s*(?:spider|version|self|selves)\b", re.I)


def wants_variants(question: str) -> bool:
    return bool(VARIANTS_RE.search(question))


def wants_whole_entity(question: str) -> bool:
    """True when the question asks for the WHOLE entity, not one part of it.

    THE single condition engine.plan() and engine.try_facts() must agree on.
    It lives here, once, because the two of them stated it separately and
    drifted twice: try_facts() reaches its profile branch through a sequence
    of early returns, and plan()'s copy of that sequence first omitted
    detect_intent() - so a field question got a profile dump instead of the
    sentence it asked for - and then omitted wants_variants(), so "what are
    the other versions of beast" had its list rendered and thrown away.
    terminal.ask() prefers rows to text, so a disagreement in either
    direction discards the answer that was actually asked for.

    The three conditions are try_facts()'s own, in its order: a field intent
    is answered by that field, a variants request is answered by the list,
    and only what is left over - a question asking who someone IS - wants the
    record entire. Chit-chat is excluded upstream of both callers, by
    engine.is_chitchat(), which needs the training data this module does not
    import.
    """
    return (detect_intent(question) is None
            and not wants_variants(question)
            and wants_profile(question))


def variants(index, resolve, doc_id: int, limit: int = 10):
    """Every record sharing this record's name, best first.

    Read from the headlines rather than the name index, because the name index
    keeps only the best eight per name and the count is the point of the
    question.
    """
    want = resolve.norm(resolve.REALITY_SUFFIX.sub(
        "", index.headlines[doc_id]).strip())
    if not want:
        return [], 0
    found, seen = [], set()
    for other in range(len(index.docs)):
        head = index.headlines[other]
        if resolve.norm(resolve.REALITY_SUFFIX.sub("", head).strip()) == want:
            found.append((len(index.text(other)), head, other))
    found.sort(reverse=True)
    # One line per DISTINCT headline: 41 records are headlined
    # "Spider-Man (Unknown)" and listing it 41 times helps nobody.
    unique = []
    for size, head, other in found:
        if head not in seen:
            seen.add(head)
            unique.append((size, head, other))
    return unique[:limit], len(found)


def render_variants(index, resolve, doc_id: int, limit: int = 10) -> str:
    rows, total = variants(index, resolve, doc_id, limit)
    if total <= 1:
        return ""
    name = resolve.REALITY_SUFFIX.sub("", index.headlines[doc_id]).strip()
    shown = ", ".join(head for _size, head, _d in rows)
    lead = (f"{total} records in my sources are called {name}"
            if total > len(rows) else
            f"{name} appears in {total} records")
    tail = f", showing the {len(rows)} largest" if total > len(rows) else ""
    return f"{lead}{tail}: {shown}."


def field_value(record: str, label: str) -> str:
    """The field's value, or "" - stopping before the narrative sections."""
    prefix = label + ": "
    for line in record.split("\n"):
        if line.startswith(("History:", "Synopsis")):
            break
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


ANSWER_CHARS = 320       # a rendered answer should read, not scroll


def answer(question: str, record: str, max_items: int = 6,
           max_chars: int = ANSWER_CHARS):
    """A rendered answer, or None to let the model handle it.

    Returns a refusal rather than None when the question IS a field question
    and the record lacks that field: falling through to the model there is how
    an invented answer gets produced.
    """
    hit = detect_intent(question)
    if not hit:
        return None
    name, label, template = hit

    sd = _sd()
    entity = sd.bare_name(sd.entity_name(record)) or record.split("\n")[0].strip()
    raw = field_value(record, label)
    if not raw:
        return _missing(question, record, entity, label)

    # Cap by characters as well as items. Storm's Abilities field is six
    # multi-sentence paragraphs joined by semicolons, so an item cap alone
    # rendered a wall of text: a field answer should read, not scroll.
    parts, used = [], 0
    items = [p.strip() for p in sd.clean_value(raw).split(";") if p.strip()]
    if label in ("Powers", "Abilities"):
        # Shared with profile_rows() via _strip_citations() (4.12 fix
        # round 2) - this filter used to be duplicated across the two
        # functions in this same module. It was added here only at first,
        # so "who is spider-man" was clean while "what are his powers"
        # still opened "Marvel Super Heroes Secret Wars Vol 1 3".
        items = _strip_citations(items)
    for part in items:
        if not part:
            continue
        if parts and (used + len(part) > max_chars or len(parts) >= max_items):
            break
        parts.append(part if len(part) <= max_chars else _first_sentence(part))
        used += len(parts[-1])
    if not parts:
        return _missing(question, record, entity, label)
    # one_terminator because the template appends a full stop to a value
    # that usually ends in one: "...ability are unknown.." .
    return sd.one_terminator(
        template.format(e=entity, v=sd.humanize("; ".join(parts))))


def _missing(question: str, record: str, entity: str, label: str) -> str:
    """The field is not in the record. Say so - and answer the rest.

    "who is the Hood, explain in detail his powers" is two questions, and the
    record answers one of them. Replying only "I don't have Lord Hood's powers
    in my sources" throws away everything it does have.
    """
    note = f"I don't have {entity}'s {_human_label(label)} in my sources."
    if wants_profile(question):
        summary = profile(record, detail=bool(DETAIL_RE.search(question)))
        if summary:
            return f"{summary} {note}"
    return note


def _first_sentence(text: str) -> str:
    """One field entry can itself be a paragraph; keep its opening claim."""
    cut = text.find(". ")
    return text[:cut] if 0 < cut < ANSWER_CHARS else text[:ANSWER_CHARS].rstrip()


def _human_label(label: str) -> str:
    return {
        "Created by": "creators",
        "First appearance": "first appearance",
        "Full name": "real name",
        "Species / origin": "species",
        "Base of operations": "base of operations",
    }.get(label, label.lower())
