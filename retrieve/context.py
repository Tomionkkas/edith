"""The `Context:` block - one implementation, used by training AND inference.

Two things this file fixes.

**Drift.** There used to be two builders. `sft_data.context_for` made the
block for training (one record, 700 chars, stopping before `History:`) while
`search.format_context` made it for inference (up to three FULL records,
`History:` included). On "who created moon knight" that was 11,200 tokens
against a 1,024-token window, so the prompt would have been truncated to an
arbitrary mid-record slice.

**Selection.** Taking lines in document order until a byte budget runs out
drops the fields that matter. `Powers` sits around character 1,500-2,800 in a
major character's record, so a 700-char budget lost it for 8 of 8 characters
tested, while `Also known as:` - dozens of aliases - consumed the budget on
its own. Spider-Man's whole context was three lines: his title, his real name
and his alias. Fields are now chosen by importance, and long values are cut at
a clean boundary rather than mid-phrase.
"""
from __future__ import annotations

# Narrative sections: long, and the fielded lines above carry the facts.
STOP_SECTIONS = ("History:", "Synopsis")
NL = chr(10)

# What a Marvel question actually needs, most important first. Powers ranks
# high because it is the part worth expanding on; the identity trivia below it
# is what a template can render without the model's help.
FIELD_PRIORITY = (
    "Full name", "Powers", "Abilities", "Also known as",
    "First appearance", "Created by", "Species / origin", "Reality",
    "Affiliation", "Occupation", "Base of operations",
    "Identity status", "Citizenship", "Gender", "Marital status",
)

PER_RECORD_CHARS = 900      # one record's field block
TOTAL_CHARS = 2400          # ~520 tokens, leaving room for question and answer
VALUE_CHARS = 260           # one field's value; alias lists run to thousands


def _split_fields(record: str):
    """(title, [(label, value)]) for the fielded head, stopping at narrative."""
    lines = record.split("\n")
    # The first line is the character's name. If it is itself a narrative
    # heading then the record has no fielded head, and promoting it to a title
    # would smuggle in the very section the trimmer exists to exclude.
    if not lines or lines[0].startswith(STOP_SECTIONS):
        return "", []
    title = lines[0].strip()
    fields = []
    for line in lines[1:]:
        if line.startswith(STOP_SECTIONS):
            break
        if ": " in line:
            label, value = line.split(": ", 1)
            if label and value.strip():
                fields.append((label.strip(), value.strip()))
    return title, fields


def _clip(value: str, limit: int = VALUE_CHARS) -> str:
    """Trim a long value at a clean boundary, never mid-phrase.

    Cutting "Created by: Brian K. Vaughan; Kyle" teaches the model that
    half-names are normal. Latest boundary wins: sentence end, then list
    separator, then a word break.
    """
    if len(value) <= limit:
        return value
    head = value[:limit]
    best = max(head.rfind("; "), head.rfind(". ") + 1, head.rfind(", "))
    if best > limit // 3:
        return head[:best].rstrip()
    i = head.rfind(" ")
    return head[:i].rstrip() if i > limit // 3 else head


def trim_record(record: str, max_chars: int = PER_RECORD_CHARS) -> str:
    """One record: the title plus the most important fields that fit.

    Priority fields first, then anything else in document order, so an
    unfamiliar field is kept when there is room rather than silently dropped.
    """
    title, fields = _split_fields(record)
    if not title and not fields:
        return ""

    by_label = {}
    for label, value in fields:
        by_label.setdefault(label, value)

    ordered = [(l, by_label[l]) for l in FIELD_PRIORITY if l in by_label]
    ordered += [(l, by_label[l]) for l, _ in fields if l not in FIELD_PRIORITY]

    out, used = ([title], len(title)) if title else ([], 0)
    seen = set()
    for label, value in ordered:
        if label in seen:
            continue
        line = "%s: %s" % (label, _clip(value))
        if used + len(line) > max_chars:
            continue              # a long field must not block the short ones
        out.append(line)
        used += len(line)
        seen.add(label)
    return "\n".join(out)


def assemble_blocks(records, per_record_chars: int = PER_RECORD_CHARS,
                    total_chars: int = TOTAL_CHARS, histories=None):
    """Trimmed blocks that fit the budget, in input order.

    Exposed separately from build_context so a caller can guarantee one
    record a slot before the others compete for what is left. Training data
    needs that: if the record an answer was written from is squeezed out by
    decoys, the example teaches the model to state facts it cannot see.
    """
    histories = list(histories) if histories else []
    blocks, used = [], 0
    for i, rec in enumerate(records):
        trimmed = trim_record(rec, per_record_chars)
        if not trimmed:
            continue
        hist = histories[i] if i < len(histories) else ""
        if hist:
            trimmed += NL + "History: " + hist
        if blocks and used + len(trimmed) > total_chars:
            break
        blocks.append(trimmed)
        used += len(trimmed)
    return blocks


def join_blocks(blocks) -> str:
    """Blocks into the final `Context:` string."""
    if not blocks:
        return ""
    return "Context:" + NL + (NL + NL).join(blocks)


def build_context(records, per_record_chars: int = PER_RECORD_CHARS,
                  total_chars: int = TOTAL_CHARS, histories=None) -> str:
    """The block exactly as the model is trained to read it.

    `records` may be one or several; several teaches the model to pick the
    relevant one, which is what makes a wrong top-1 retrieval survivable.

    `histories` is parallel to `records`. A History excerpt attaches to the
    record it came from, not to the end of the block: with several records the
    answer's record is deliberately not always first, so a trailing History
    line would describe a different character than the block above it.
    """
    return join_blocks(assemble_blocks(records, per_record_chars,
                                       total_chars, histories))
