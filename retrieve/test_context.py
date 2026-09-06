"""The context block is the train/inference contract. Pin it.

A mismatch here does not raise; it produces a fluent wrong answer, and reads
as a bad model rather than a bad prompt.
"""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "context", Path(__file__).resolve().parent / "context.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

REC = ("Lord Hood\n"
       "Full name: Parker Davis Robbins\n"
       "First appearance: Hood Vol 1 1\n"
       "Created by: Brian K. Vaughan; Kyle Hotz\n"
       "History: " + ("a long narrative section. " * 200))


# ------------------------------------------------------- narrative sections

def test_history_is_excluded():
    assert "History:" not in C.trim_record(REC)


def test_synopsis_is_excluded():
    assert "Synopsis" not in C.trim_record("Name\nField: v\nSynopsis\n" + "x " * 500)


def test_fields_above_history_survive():
    out = C.trim_record(REC)
    assert "Full name: Parker Davis Robbins" in out
    assert "Created by: Brian K. Vaughan; Kyle Hotz" in out


# ------------------------------------------------------------- the budget

def test_a_single_oversized_record_is_still_capped():
    """The old code appended the first record unconditionally."""
    huge = "Name\n" + "\n".join(f"Field{i}: {'v' * 80}" for i in range(500))
    assert len(C.build_context([huge])) < C.PER_RECORD_CHARS + 200


def test_total_budget_is_enforced_across_records():
    recs = ["Name%d\n%s" % (i, "\n".join(f"F{j}: {'v'*60}" for j in range(40)))
            for i in range(10)]
    out = C.build_context(recs)
    assert len(out) <= C.TOTAL_CHARS + C.PER_RECORD_CHARS + 100


def test_never_splits_a_line():
    huge = "Name\n" + "\n".join(f"Field{i}: {'v' * 80}" for i in range(500))
    for line in C.trim_record(huge).split("\n"):
        assert line == "" or line in huge.split("\n")


# -------------------------------------------------------------- the shape

def test_starts_with_the_context_header():
    assert C.build_context([REC]).startswith("Context:\n")


def test_records_separated_by_a_blank_line():
    out = C.build_context(["A\nF: 1", "B\nF: 2"])
    assert out == "Context:\nA\nF: 1\n\nB\nF: 2"


def test_no_records_is_empty_string_not_a_bare_header():
    """A lone 'Context:' would signal 'invent something'."""
    assert C.build_context([]) == ""


def test_records_that_trim_to_nothing_are_skipped():
    assert C.build_context(["History: only narrative"]) == ""


def test_history_excerpt_appended_when_asked():
    out = C.build_context([REC], histories=["he stole a cloak"])
    assert out.endswith("\nHistory: he stole a cloak")


def test_no_trailing_whitespace():
    out = C.build_context([REC, REC])
    assert out == out.rstrip()


# ------------------------------------------------------------ determinism

def test_same_input_same_bytes():
    assert C.build_context([REC, REC]) == C.build_context([REC, REC])


# --------------------------------------------------- field selection

LATE_POWERS = ("Wolverine\n"
               "Also known as: " + "Alias, " * 300 + "\n"
               "Occupation: " + "job; " * 200 + "\n"
               "Powers: Regenerative healing factor; adamantium skeleton\n"
               "First appearance: Incredible Hulk Vol 1 180\n"
               "History:\n" + "narrative " * 500)


def test_powers_survives_a_flood_of_earlier_fields():
    """Document-order truncation lost Powers on 8 of 8 major characters."""
    out = C.trim_record(LATE_POWERS)
    assert "Powers: Regenerative healing factor" in out


def test_first_appearance_survives_too():
    assert "First appearance: Incredible Hulk Vol 1 180" in C.trim_record(LATE_POWERS)


def test_a_long_field_does_not_block_the_short_ones_after_it():
    rec = ("Name\nBloat: " + "x " * 5000 + "\nPowers: flight\nReality: Earth-616")
    out = C.trim_record(rec)
    assert "Powers: flight" in out and "Reality: Earth-616" in out


def test_long_values_are_clipped():
    rec = "Name\nOccupation: " + "job; " * 400
    line = [l for l in C.trim_record(rec).split("\n") if l.startswith("Occupation")][0]
    assert len(line) <= C.VALUE_CHARS + len("Occupation: ") + 2


def test_clip_cuts_at_a_boundary_not_mid_name():
    """'Created by: Brian K. Vaughan; Kyle' teaches that half-names are fine."""
    v = "; ".join(f"Creator Number {i} Longname" for i in range(40))
    assert not C.trim_record(f"Name\nCreated by: {v}").rstrip().endswith("Creator")


def test_priority_fields_come_before_unknown_ones():
    out = C.trim_record("Name\nZzzUnknown: v\nPowers: flight")
    assert out.index("Powers") < out.index("ZzzUnknown")


def test_unknown_fields_are_still_kept_when_there_is_room():
    assert "ZzzUnknown: v" in C.trim_record("Name\nZzzUnknown: v\nPowers: flight")


def test_title_is_always_first():
    assert C.trim_record(LATE_POWERS).split("\n")[0] == "Wolverine"


def test_a_record_that_is_only_narrative_yields_nothing():
    assert C.trim_record("History:\n" + "narrative " * 100) == ""


def test_duplicate_labels_keep_the_first_value():
    out = C.trim_record("Name\nReality: Earth-616\nReality: Earth-1610")
    assert "Earth-616" in out and "Earth-1610" not in out


def test_colon_inside_a_value_is_not_a_field_split():
    out = C.trim_record("Name\nPowers: Spider-Physiology: wall-crawling")
    assert "Powers: Spider-Physiology: wall-crawling" in out
