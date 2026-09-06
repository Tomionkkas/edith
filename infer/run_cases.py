"""Score what EDITH SAYS, by driving EDITH.

The two existing harnesses ask one question - did retrieval return the right
doc id - and both sit near ceiling (97.5%, 39/40) while blind to everything
that has gone wrong since. flagship_score compares headline STRINGS, so it
reads `who is beast` as a pass while answering from `Krahllak (Earth-616)`,
an obscure alien headlined `Beast`.

This one drives infer/terminal.py as a subprocess - the product, not
engine.plan() - and keys each case on the `Page:` line that --trace reports,
which is identity rather than a name.

    py infer/run_cases.py              score, one line per failure
    py infer/run_cases.py --show 5     print the transcript of N failures
    py infer/run_cases.py --case 12    run one case, print everything

The scorer is a pure function with its own tests in infer/test_run_cases.py.
A harness that scores wrongly is worse than no harness: the throwaway this
replaces reported 23/27 and then 22/27 on a build that had not changed.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


TRACE = re.compile(r"^trace: doc=(\S+) page=(.*) offered=(\d+)$")

# render.speaker() hardcodes this bar; the label beside it is theme-dependent
# (edith, spidey, logan, we) and the bar is not.
SPEAKER = "▌"
FOOTER = "◈"


def parse_trace(line: str):
    """One `trace:` line as a dict, or None when the line is not one.

    `page` is matched greedily to the last ` offered=` because a Fandom title
    is arbitrary text and may contain `=`. It is then stripped: emit_trace's
    headline fallback is not itself stripped, and an unstripped trailing
    space would otherwise read as a false page mismatch against a case's
    exact page string. `line.strip()` above is also what makes a
    `\\r\\n`-terminated line - the shape Windows stderr actually delivers -
    parse at all.
    """
    m = TRACE.match(line.strip())
    if not m:
        return None
    return {"doc": m.group(1), "page": m.group(2).strip(),
            "offered": int(m.group(3))}


def split_answer(transcript: str) -> str:
    """The answer out of a captured session: the LAST thing EDITH said.

    The FOOTER is the authoritative anchor, not the speaker bar: `◈` is
    printed once per answered turn, at the end of `ask()`, and by nothing
    else - no theme glyph collides with it. `▌`, by contrast, is printed for
    BOTH parties (`▌ you` and `▌ edith` alike), so anchoring on the last
    speaker bar picks up the user's own `/quit` turn that run_case() appends
    after every case, and the "answer" comes back as that instead.

    So: find the LAST footer, then walk backward to the nearest speaker bar
    before it. `ask()` prints in a fixed order:

        blank, retrieval line (absent on chit-chat), blank,
        `▌ <label>`, the answer, [blank, caption], blank, footer

    so the nearest `▌` before a footer is always EDITH's, which excludes the
    retrieval line and the picker menu for free: neither is a claim the
    product made, and checking must_not_include against the whole transcript
    would score the record's own text as EDITH's. A picker turn returns from
    `ask()` before printing anything, so it has no footer, and this correctly
    yields "" - score() judges a picker case on `offered`, never on answer
    text.

    There is no sources BOX any more - 4.12 replaced it with the field block
    that `▌ <label>` now introduces, so what used to sit ABOVE the speaker
    bar and be excluded for free is now the answer itself, deliberately
    inside the window. The caption is the one thing that is inside the window
    without being an answer: `render.caption()` prints between the answer and
    the footer, and a case's must_not_include would score it as something
    EDITH said. That is safe only because `--trace` pins the theme to `plain`
    and `plain` defines no caption pair - a coupling nothing stated or tested
    until 2026-09-05, when the whole-branch review found it. It is now pinned
    by infer/test_run_cases.py::TheScoredWindowExcludesChrome; give `plain` a
    caption, or unpin the theme under `--trace`, and that test fails rather
    than the score quietly drifting.

    render.wrap() is a greedy word wrap, so a two-word fact lands across a
    line break whenever it straddles the margin. Every run of whitespace is
    collapsed to a single space so a wrapped phrase still matches as one
    substring instead of scoring a right answer as a missing string.
    """
    lines = transcript.split("\n")
    ends = [i for i, ln in enumerate(lines) if FOOTER in ln]
    if not ends:
        return ""
    end = ends[-1]
    starts = [i for i in range(end) if SPEAKER in lines[i]]
    if not starts:
        return ""
    return " ".join("\n".join(lines[starts[-1] + 1:end]).split())


def score(case: dict, answer: str, trace, traced: int) -> list:
    """Why this case failed. An empty list is a pass.

    Every applicable rule is reported, not just the first: "right record,
    wrong words" and "wrong record" are different bugs and a score that
    stops at one cannot tell them apart.

    `traced` is how many of the case's turns reached the engine. The spec
    opens its scoring section with "for each case, run every turn in order,
    then judge the LAST turn", so a case whose last turn never ran cannot
    return a pass: the rule that says judge the last turn has nothing to
    judge. Once a picker opened, needs_a_pick (infer/terminal.py, since
    replaced by take_reference) used to swallow every later line without
    calling ask(), and report() labels each case by case["ask"][-1] - so
    until R11 the log printed a PASS beside two strings the product never
    processed. The NOTE disclosed the gap; the score laundered it. Reported
    FIRST, because it is the fact that makes every other reason
    untrustworthy: they describe some earlier turn. The parameter is
    required and undefaulted on purpose - a default meaning "rule not
    applicable" is how a rule gets silently skipped, which is the class of
    bug this phase has now found three times.

    A non-picker case with an empty (or whitespace-only) answer fails on
    that alone, reported before the string checks and instead of them.
    emit_trace runs before the timing footer prints, so a subprocess that
    dies in between - a crash inside `_generate`, a killed process, a
    truncated stream - emits a perfectly good trace and no answer. A case
    with no must_include (cases 25 and 26 carry none) would otherwise make
    no positive assertion at all, and an empty answer would silently pass.
    must_include/must_not_include are skipped in this case as noise against
    an empty string; the `page` check still runs, since the trace exists and
    a wrong record is a separate, useful fact.
    """
    sent = len(case["ask"])
    reasons = []
    if traced < sent:
        reasons.append(f"{sent - traced} turn(s) never reached the engine; "
                       f"scored off turn {traced}")

    if trace is None:
        return reasons + ["no trace: the terminal answered nothing"]

    offered = trace["offered"]
    if case.get("expect") == "picker":
        # `page` is None and there is no answer to match, so a choice being
        # offered is the whole test. Three outcomes, not two: case 24
        # resolves nothing and REFUSES, and "answered instead of offering a
        # choice" was a false statement about it, in a report whose whole
        # purpose is not making false statements (F2).
        if offered:
            return reasons
        if trace["doc"] == "none":
            return reasons + ["refused instead of offering a choice"]
        return reasons + ["answered instead of offering a choice"]
    if offered:
        # Reported as itself. The picker firing where an answer was wanted is
        # a real defect and must not read as a pile of missing strings.
        return reasons + [
            f"offered a picker of {offered}; an answer was expected"]

    if case.get("page") and trace["page"] != case["page"]:
        reasons.append(f"page {trace['page']}, wanted {case['page']}")

    if not answer.strip():
        # True whether no footer printed at all or a footer printed over
        # empty text; the old wording claimed the first and fired on both.
        reasons.append("no answer: the turn produced no text")
        return reasons

    low = answer.lower()
    reasons += [f"missing '{s}'" for s in case.get("must_include", [])
                if s.lower() not in low]
    reasons += [f"forbidden '{s}'" for s in case.get("must_not_include", [])
                if s.lower() in low]
    return reasons


def run_case(case: dict, timeout: int = 300):
    """Drive one case through a fresh terminal.

    Returns (answer, trace, transcript, traced, trouble).

    One subprocess per case, and no batching. /quit exits the process, and
    short of quitting both last_doc and last_offer survive - so `who is
    namor` in one case would prime the pronoun in the next. Slow is the
    correct trade: this runs at the end of a phase, not per commit.

    `traced` is len(traces): how many trace lines came back, against
    len(case["ask"]) turns sent. Once a picker opened, needs_a_pick
    (infer/terminal.py, since replaced by take_reference) used to intercept
    every following line that was not a `/command` - it printed the pick
    prompt and `continue`d the loop without ever calling ask() - so a turn
    typed after an open menu could be sent and never reach the engine at
    all. It goes to score(), which FAILS the case on it (R11, reversing
    R7), and to report(), which NOTEs the count: the verdict and the count
    are different facts and a reader wants both.

    `trouble` is None on an ordinary run, and a short string - "timed out
    after Ns" or "subprocess exited N" - when the process itself misbehaved.
    main() appends it to the case's reasons, because print-only it could be
    passed over entirely: a run whose subprocess died could report 27/27 and
    exit 0 (F3). A timeout additionally leaves `trace` None, which score()
    reports as "no trace: the terminal answered nothing" (R9). report()
    prints the NOTE as well.
    """
    turns = "\n".join(case["ask"]) + "\n/quit\n"
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "infer" / "terminal.py"), "--trace"],
            input=turns, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        # A hung case must not take the other 26 with it (R9). No trace
        # means score() reports "no trace: the terminal answered nothing",
        # which is exactly right - nothing was answered.
        return ("", None, f"<timed out after {timeout}s>", 0,
                f"timed out after {timeout}s")

    traces = [t for t in (parse_trace(ln) for ln in proc.stderr.split("\n")) if t]
    trouble = None if proc.returncode == 0 else f"subprocess exited {proc.returncode}"
    # The LAST trace, never the Nth. Traces are not 1:1 with typed lines: a
    # /command emits none. Before the swallow fix, once a picker opened,
    # needs_a_pick also intercepted every later non-/command turn without
    # calling ask() at all (R7) - take_reference() no longer does that, a
    # non-reference line always reaches ask(), but a /command still emits no
    # trace, so `answer` (the last thing said) and `trace` (the last engine
    # turn) can still point at an earlier turn than the one typed last -
    # which is exactly right, since that earlier turn is the last one that
    # ran.
    return (split_answer(proc.stdout), (traces[-1] if traces else None),
            proc.stdout, len(traces), trouble)


def report(results, show: int = 0, always: bool = False) -> int:
    """Print the score and every failure. Returns the count that passed.

    `show` triages a full run: the transcript of the first `show` FAILURES.
    `always`, set only by `--case`, prints that one case's transcript
    whether it passes or fails (R8) - "run one case, print everything"
    cannot silently mean "only when it happens to be wrong": `--case 8` is
    `who is beast`, which offers a picker rather than answering and may
    well pass, and a verification step that reads the transcript would
    otherwise have nothing to read.
    """
    passed = [r for r in results if not r["reasons"]]
    print(f"\n{len(passed)}/{len(results)}\n")
    shown = 0
    for r in results:
        ask = r["case"]["ask"][-1]
        if r["reasons"]:
            print(f"  FAIL  {ask[:38]:<40}{r['reasons'][0]}")
            for extra in r["reasons"][1:]:
                print(f"        {'':<40}{extra}")
            # Print the page on every failure even when it passed: "right record,
            # wrong words" and "wrong record" are different bugs. Only for a
            # case that HAS a page to check, though: `page  none  ok` on a
            # case whose `page` is None said a record had been checked and
            # found right when nothing was ever checked (F2).
            if (r["case"].get("page") and r["trace"]
                    and not any(x.startswith("page ") for x in r["reasons"])):
                print(f"        page  {r['trace']['page']}  ok")
        # The transcript dump is not nested inside the failure branch (R8):
        # `--show N` triages the first N failures of a full run; `always`
        # (set only for `--case N`) prints regardless of `r["reasons"]`.
        if always or (r["reasons"] and shown < show):
            shown += 1
            print("\n" + "\n".join("      | " + ln
                                   for ln in r["transcript"].split("\n")) + "\n")
        # The count, beside the verdict. score() now fails a case whose turns
        # did not all reach the engine (R11), but the failure says only that
        # it happened - this says how many, and stays outside the failure
        # branch so that nothing can hide it.
        sent = len(r["case"]["ask"])
        if r["traced"] < sent:
            print(f"  NOTE  {ask[:38]:<40}{sent} turns sent, "
                  f"{r['traced']} reached the engine")
            print("        (an open menu swallows later turns "
                  "— infer/terminal.py:199)")
        # A subprocess that misbehaved on shutdown - hung, or exited non-zero
        # after printing a perfectly good footer - must not read as a clean
        # pass (R9/F3). Same NOTE mechanism, unconditional on r["reasons"].
        if r.get("trouble"):
            print(f"  NOTE  {ask[:38]:<40}{r['trouble']}")
    return len(passed)


def main(argv=None) -> int:
    # The console is cp1250 on this machine and the first box-drawing
    # character in a transcript would take the run down. theme.use_utf8()
    # is where this lesson already lives - see PROGRESS.md.
    _load("theme", "infer/theme.py").use_utf8()

    ap = argparse.ArgumentParser(
        prog="run_cases", description="score what EDITH says")
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help="print the full transcript of the first N FAILURES")
    ap.add_argument("--case", type=int, default=0, metavar="N",
                    help="run only case N, and print its transcript "
                         "whether it passes or fails")
    args = ap.parse_args(argv)

    cases = _load("answer_cases", "infer/answer_cases.py").CASES
    always = bool(args.case)
    if args.case:
        if not 1 <= args.case <= len(cases):
            print(f"  case must be 1..{len(cases)}")
            return 2
        cases, args.show = [cases[args.case - 1]], 1

    results = []
    for i, case in enumerate(cases, 1):
        print(f"  {i:>3}/{len(cases)}  {case['ask'][-1][:44]}", flush=True)
        answer, trace, transcript, traced, trouble = run_case(case)
        reasons = score(case, answer, trace, traced)
        if trouble:
            # F3: as a printed annotation this could be passed over, and a
            # run whose subprocess died could report 27/27 and exit 0. Worse,
            # on a multi-turn case that emits its trace and then dies before
            # the footer, split_answer returns the PREVIOUS turn's answer
            # while `trace` is the LAST turn's - two mismatched halves that
            # can agree by accident. Appended here rather than inside score()
            # because R11 fixes that signature at four parameters, and
            # `trouble` is a fact about the subprocess rather than about
            # anything EDITH said.
            reasons.append(trouble)
        results.append({"case": case, "trace": trace, "transcript": transcript,
                        "traced": traced, "trouble": trouble,
                        "reasons": reasons})

    passed = report(results, args.show, always)
    # Non-zero so it can gate a phase.
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
