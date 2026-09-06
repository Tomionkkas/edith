"""EDITH — a Marvel terminal.

    edith                       (Windows: .\\edith)
    edith --ask "who created moon knight"
    edith --theme hulk

Everything factual on screen comes from a retrieved record: a whole-entity
question is answered by the record's own fields, printed as a labelled block
below the speaker bar, and a field question by one sentence rendered from the
same record in code. (There is no sources box any more — 4.12 replaced the
bordered record-above-the-answer with the field block, which is the answer
rather than chrome around one.) The model writes prose and nothing else. A
theme changes colour, the glyph, the wording of the retrieval line and the
speaker's name — never an answer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path.home() / ".edith" / "config.json"
CKPT = ROOT / "checkpoints" / "model.safetensors"
TOKENIZER = ROOT / "tokenizer" / "marvel_bpe_50257.model"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


theme = _load("theme", "infer/theme.py")
render = _load("render", "infer/render.py")
motion = _load("motion", "infer/motion.py")
reference = _load("reference", "infer/reference.py")
bootstrap = _load("bootstrap", "bootstrap.py")

HELP = (
    ("/theme [name]", "switch skin, or list them"),
    # `off` does not hide the record - it swaps the field block for
    # facts.profile() prose built from the SAME fields, which is what
    # ask() falls through to when plan.rows is not used.
    ("/sources on|off", "fields as a block, or as prose"),
    ("/history", "the last few questions"),
    ("/forget", "drop the records you picked"),
    ("/help", "this"),
    ("/quit", "leave (or ctrl-c)"),
)

MISSING = """EDITH needs two things that are not in git:

  {missing}

They are {mb} MB together - the weights from HuggingFace and the corpus the
index is built from. The index itself is never shipped: it is a pickle, and
unpickling executes whatever is in the file.
"""

# Printed instead of MISSING when bootstrap.download_mb() comes back 0: the
# weights and corpus are already there and only the index is gone. Quoting
# an 810 MB total in that case would ask someone to approve a download that
# was never going to happen - the index rebuilds locally, it never fetches.
NOTHING_TO_DOWNLOAD = """EDITH needs the retrieval index, which is not in git:

  {missing}

It is never shipped - unpickling a pickle executes whatever is in the file -
so nothing needs to download for this. It rebuilds locally from the corpus
you already have, in about 30 seconds.
"""

DECLINED = """
No problem. When you want them:

    py install.py          (Windows)
    python3 install.py     (macOS / Linux)
"""

FETCH_FAILED = """
That did not finish: {error}

What already downloaded is kept - running EDITH again resumes rather than
starting over. py install.py does the same job with more output along the
way.
"""

STILL_MISSING = """
Still missing after fetching:

  {missing}

Something did not arrive. Try again, or py install.py for more detail.
"""


# ------------------------------------------------------------------ config

def _took(seconds: float) -> str:
    """A rendered answer lands in a millisecond; "0.0s" hides the best number
    this project has."""
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.1f}s"


def read_config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_config(cfg: dict) -> None:
    """Best effort: a terminal that cannot save a preference still works."""
    try:
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except OSError:
        pass


# ------------------------------------------------------------------- shell

class Terminal:
    HISTORY_MAX = 20
    HISTORY_SHOWN = 10

    def __init__(self, theme_name: str = "", ckpt: Path = CKPT,
                 trace: bool = False):
        # Machine mode. A harness that scores differently depending on the
        # developer's saved preferences is not a measurement. The record's
        # fields are the answer now (below the speaker bar, not a box of raw
        # record text above it), so show_sources is pinned ON under --trace
        # regardless of what a machine has saved. Pinned in process either
        # way: /sources off would write the config file.
        self.trace = trace
        # theme.DEFAULT is "spider-man" (infer/theme.py:85), whose label is
        # `spidey` and whose glyph and verb differ. Machine mode pins `plain`
        # by name so a transcript reads the same on every machine.
        self.theme = (theme.get("plain") if trace else
                      theme.get(theme_name) or theme.get(
                          read_config().get("theme", theme.DEFAULT))
                      or theme.THEMES[theme.DEFAULT])
        self.ckpt = Path(ckpt)
        # Machine mode PINS this, and the pin flipped in 4.12. It used to be
        # off: a raw sources BOX printed record text, so `category:` and
        # `she is` - both forbidden by cases - leaked in from a machine whose
        # saved config had sources on. The field block is the ANSWER now, so
        # forcing it off would make run_cases measure a view no user sees.
        # Still pinned, never read from config under --trace: a harness that
        # scores differently per machine is not a measurement. Spec R1.
        self.show_sources = True if trace else read_config().get("sources", True)
        # render.wrap() breaks the answer at self.width. A harness expectation
        # like `must_include: "new mutants"` is a substring match, and it
        # would fail whenever the measured console width happened to wrap
        # between "new" and "mutants" - a false failure that moves with the
        # terminal size. Pinned so the same answer wraps the same way on
        # every machine.
        self.width = 80 if trace else render.terminal_width()
        self.model = self.sp = self.index = None
        # The record the last question resolved to. A follow-up that
        # names nobody - "more details about his powers" - belongs to it.
        # DELIBERATELY persists unchanged across chitchat and an unresolved
        # question, so the follow-up still has something to attach to. A
        # menu turn is NOT one of those: plan() returns the TOP-RANKED
        # record as doc_id even on a picker turn (engine.py:257), so ask()
        # overwrites last_doc with that row. Consequence: /history shows the
        # menu's top-ranked record beside a turn that only displayed a menu,
        # and a pronoun follow-up after an unanswered menu attaches to that
        # top row rather than to nothing. Pre-existing and defensible -
        # only this comment used to claim otherwise.
        self.last_doc = None
        # What THIS turn resolved to - None when it didn't. Set fresh by
        # every ask() call, unlike last_doc above. /history reads this, not
        # last_doc: last_doc's whole point is to survive a non-resolving
        # turn, which is exactly why it is the wrong field for reporting
        # what that turn actually did.
        self.last_turn_doc = None
        # The rows the last menu showed. A phrase like "the ultimate one"
        # names one of them, and it stays readable for exactly ONE following
        # turn: a menu from three questions ago must never silently capture a
        # phrase typed now.
        self.last_offer = None
        # {name key: doc id} for names the user has explicitly picked. Only a
        # pick settles a key - resolution never does, because that is the
        # engine guessing rather than the user choosing.
        self.settled = {}
        # (question, doc id, headline) per turn, newest last. Bounded: a long
        # session must not grow state without limit. Appended from
        # handle_line(), never from ask() - ask() is also called internally
        # with the sentinel "who is this" from both pick paths (offer()'s
        # console cursor branch and handle_line()'s reference-hit branch),
        # and appending inside ask() would fill /history with a question the
        # user never typed, hiding the one they did.
        self.history = []

    # -- boot ---------------------------------------------------------------

    def offer_bootstrap(self) -> bool:
        """Fetch what is missing, having asked first.

        Not an automatic download: ~800 MB starting the first time somebody
        types `edith`, on whatever connection they are on, is a surprise.
        Not a dead end either - the old message told them to copy the file
        from a machine that has it, which stopped being true on publication
        day.

        Returns True when EDITH can carry on.
        """
        gone = bootstrap.missing(self.ckpt)
        if not gone:
            return True

        def name(p):
            # --ckpt can point outside the repo, and relative_to raises on
            # that rather than falling back.
            try:
                return str(p.relative_to(ROOT))
            except ValueError:
                return str(p)

        names = "\n  ".join(name(p) for p in gone)
        mb = bootstrap.download_mb(self.ckpt)
        if mb:
            print(MISSING.format(missing=names, mb=mb))
        else:
            print(NOTHING_TO_DOWNLOAD.format(missing=names))

        # run_cases.py drives this as a subprocess: prompting a closed stdin
        # hangs the harness or raises EOFError inside a measurement.
        if not sys.stdin.isatty():
            print(DECLINED)
            return False

        try:
            answer = input("  Fetch them now? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-C here is the very first prompt a new user ever sees.
            # terminal.py already treats it as a graceful stop everywhere
            # else it can happen (:447, run(); :713, handle_line(); the
            # main input loop) - this is that same convention, not a new
            # decision.
            answer = "n"
        if answer not in ("y", "yes"):
            print(DECLINED)
            return False

        print()
        try:
            bootstrap.fetch_all()
        except Exception as exc:
            print(FETCH_FAILED.format(error=exc))
            return False
        print()

        # fetch_all() can return normally without actually having finished -
        # trusting that would turn a partial fetch into a confusing
        # downstream model-load error instead of a clear statement here.
        still_gone = bootstrap.missing(self.ckpt)
        if still_gone:
            print(STILL_MISSING.format(
                missing="\n  ".join(name(p) for p in still_gone)))
            return False
        return True

    def boot(self) -> bool:
        """Load everything, narrating the wait. 3.5 s of silence reads as a
        hang; 3.5 s of timings reads as work."""
        t = self.theme
        print()
        motion.wipe(render.banner(t, self.width).split("\n"), delay=0.07)
        print()
        if not self.offer_bootstrap():
            return False

        self.search = _load("search", "retrieve/search.py")
        self.disambiguate = _load("disambiguate", "retrieve/disambiguate.py")
        self.facts = _load("facts", "infer/facts.py")
        self.resolve = _load("resolve", "retrieve/resolve.py")
        self.sft = _load("sft_data", "train/sft_data.py")
        self.sample = _load("sample", "train/sample.py")
        self.engine = _load("engine", "infer/engine.py")

        # The ledger. Loading takes ~4.0 s and the screen used to show
        # nothing for all of it, then dump three timings at once. The
        # spinner costs no wall-clock: it turns during work that was
        # already happening.
        for label, work in (("model", self._load_model),
                            ("index", self._load_index),
                            ("names", self._load_names)):
            start = time.perf_counter()
            motion.spin(label.ljust(7), work,
                        colour=t.dim, paint=theme.paint)
            print(theme.paint(f"   {time.perf_counter() - start:.1f}s", t.faint))

        self.header()
        return True

    def header(self, with_banner: bool = False) -> None:
        """The block that says EDITH is up and ready.

        Printed on boot AND after a theme change. The first version reprinted
        only the wordmark and the theme line, so switching skins left a header
        with no record count and no rule under it - it read as half-loaded,
        as though something was still coming.
        """
        t = self.theme
        if with_banner:
            print()
            print(render.banner(t, self.width))
        print()
        # E.D.I.T.H. is Tony Stark's "Even Dead I'm The Hero" - the model is
        # named for a Marvel AI assistant, so it may as well say so.
        lead = theme.paint("even dead i'm the hero", t.accent)
        if self.index is None:
            print(lead)
        else:
            # The record count is the one number here the loader actually
            # produced, so it is revealed rather than simply printed. A
            # REVEAL, not a progress bar - it plays once the value is known,
            # so it must never be described as measuring anything.
            motion.count(len(self.index),
                         prefix=lead + theme.paint(" · 250M · ", t.dim),
                         suffix=theme.paint(" records indexed", t.dim),
                         colour=t.accent, paint=theme.paint)
        line = "  " + render.chip(t.name, t)
        if t.quip and not theme.caption_of(t):
            line += theme.paint("  " + t.quip, t.faint)
        else:
            line += theme.paint("   /theme to change   /help", t.faint)
        print(line)
        print(render.rule(t, self.width))

    def _load_model(self):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.block, self.step, _ = self.sample.load_model(
            self.ckpt, self.device)

    def _load_index(self):
        self.index = self.search.Index.load()

    def _load_names(self):
        self.engine._NAMES = self.resolve.load() or {}
        self.resolve.token_index(self.engine._NAMES)
        # Walk the whole retrieval path once. First call costs 13 ms in regex
        # compilation and lazy imports, and it would land on the user's first
        # question, where it reads as "retrieval is slow".
        self.engine.plan("who created moon knight", self.index, self.sft,
                         self.search, self.disambiguate, self.facts, self.resolve)

    def take_reference(self, line: str):
        """The doc id `line` names in the last menu, or None.

        Replaces select() and needs_a_pick() together. The old pair could
        only read a bare number, and anything else was swallowed rather than
        answered - so `tell me about the ultimate one`, the failure that
        motivated the picker, never reached the engine at all.
        """
        if not self.last_offer:
            return None
        i = reference.pick_row(line, self.last_offer)
        return None if i is None else self.last_offer[i][2]

    def settle(self, doc_id: int) -> None:
        """Remember that the user chose this record for this name.

        Strips EVERY trailing paren (see the loop below), not just the
        reality suffix. facts.variants() strips only REALITY_SUFFIX
        (infer/facts.py:216-217), so for a headline like "Sasquatch (Beast)
        (Earth-616)" this settles on ("sasquatch",) while variants() keys on
        "sasquatch beast". Pre-existing asymmetry in variants(), not fixed
        here: the visible symptom is a narrower menu on a broad ask after
        such a pick, never a wrong answer.
        """
        head = self.index.headlines[doc_id]
        # PAREN_SUFFIX, looped - the same pattern resolve.py's names_of() uses
        # on the Page: line (resolve.py:136-141) - not a single REALITY_SUFFIX
        # strip. A headline can carry more than one trailing disambiguator,
        # e.g. "Sasquatch (Beast) (Earth-616)": stripping only the reality
        # suffix left "(Beast)" attached, so the settled key was
        # ("sasquatch", "beast") and could never match query_key("beast") -
        # the pick silently did nothing, with no error.
        while True:
            stripped = self.resolve.PAREN_SUFFIX.sub("", head)
            if stripped == head:
                break
            head = stripped
        key = self.resolve.query_key(head.strip())
        if key:
            self.settled[key] = doc_id

    # -- one question -------------------------------------------------------

    def emit_trace(self, doc_id, offered: int) -> None:
        """One machine-readable line per turn, for infer/run_cases.py.

        stderr on purpose: it can never be mistaken for an answer, and the
        transcript stays readable. `page` is the record's identity, which is
        what a case keys on - `Beast` headlines both Henry McCoy and an alien
        called Krahllak, and comparing headlines is why the flagship harness
        scores that wrong answer as a pass.
        """
        if not self.trace:
            return
        page = "none"
        if doc_id is not None:
            # Wikipedia prose and comic issues carry no `Page:` line, so the
            # headline is the best identity they have.
            page = (self.resolve.page_title(self.index.text(doc_id))
                    or self.index.headlines[doc_id])
        sys.stderr.write(f"trace: doc={doc_id if doc_id is not None else 'none'}"
                         f" page={page} offered={offered}\n")
        sys.stderr.flush()

    def ask(self, question: str) -> None:
        t = self.theme
        start = time.perf_counter()
        plan = self.engine.plan(question, self.index, self.sft, self.search,
                                self.disambiguate, self.facts, self.resolve,
                                previous=self.last_doc, settled=self.settled)
        found = time.perf_counter() - start
        # Set BEFORE the choices branch's early return, and unconditionally
        # (including None) - this turn's own outcome, for /history.
        self.last_turn_doc = plan.doc_id
        if plan.doc_id is not None:
            self.last_doc = plan.doc_id

        # BEFORE the choices branch: that branch returns without printing, so
        # a trace emitted at the end would say nothing on the four cases that
        # expect a picker.
        self.emit_trace(plan.doc_id,
                        len(plan.choices) if plan.choices is not None else 0)

        if plan.choices is not None:
            self.offer(plan.choices)
            return

        caught = 1 if plan.doc_id is not None else 0
        print()
        # A greeting is not a lookup, so it does not report one.
        if not plan.chitchat:
            print("  " + theme.paint(t.glyph, t.accent) + " "
                  + theme.paint(f"{t.verb} ", t.dim)
                  + theme.paint(f"{len(self.index):,}", t.text)
                  + theme.paint(" records — ", t.dim)
                  + theme.paint(str(caught), t.text)
                  + theme.paint(" caught · ", t.dim)
                  + theme.paint(f"{found * 1000:.0f} ms", t.dim))

        # The record's fields are printed BELOW the speaker bar, because
        # they are what EDITH is claiming - not chrome around a claim.
        print()
        print(render.speaker(t, t.label, t.second or t.accent))
        if plan.rows and self.show_sources and plan.doc_id is not None:
            motion.wipe(render.answer_rows(
                self.index.text(plan.doc_id), plan.rows, t, self.width))
            source = "from the record"
        elif plan.text is not None:
            print(theme.paint(render.wrap(plan.text, self.width), t.text))
            source = "from the record" if plan.doc_id is not None else "no source"
        else:
            self._generate(plan.prompt)
            source = "grounded" if plan.doc_id is not None else "unprompted"

        pair = theme.caption_of(t)
        if pair and t.quip:
            print()
            print("  " + render.caption(t.quip, t, self.width - 2))

        print()
        print("  " + theme.paint(f"◈ {caught} source{'' if caught == 1 else 's'}"
                                 f"  ·  ⧗ {_took(time.perf_counter() - start)}"
                                 f"  ·  {source}", t.faint))

    def _generate(self, prompt: str) -> None:
        """Stream the answer, cleaned on the way out.

        A terminal cannot un-print, so TidyStream holds each word back until
        the clean-up can no longer change it - see infer/render.py.
        """
        t = self.theme
        colour = theme.rgb(t.text) if theme.COLOUR_ENABLED and t.text else ""
        out = render.TidyStream(
            lambda s: render.write(colour + s + (theme.RESET if colour else "")),
            lambda s: self.engine.tidy(s, self.sft), width=self.width)
        try:
            self.sample.generate(self.model, self.sp, self.device, prompt,
                                 self.block, max_new=220, temp=0.7, top_k=50,
                                 top_p=0.95, repetition_penalty=1.1, stream=out)
        except KeyboardInterrupt:
            render.write(theme.paint(" …stopped", t.faint))
        out.close()
        print()

    # -- commands -----------------------------------------------------------

    def command(self, line: str) -> bool:
        """Handle a /command. Returns False to leave the loop."""
        name, _, arg = line[1:].strip().partition(" ")
        name, arg = name.lower(), arg.strip()
        if name in ("quit", "exit", "q"):
            return False
        if self.trace and name in ("theme", "sources"):
            # Machine mode is pinned so a transcript reads the same on every
            # machine, and so it writes no config file. A piped `/theme` or
            # `/sources` would otherwise un-pin it mid-session and mutate the
            # developer's ~/.edith/config.json.
            print(theme.paint(f"  machine mode: /{name} is pinned, ignoring",
                              self.theme.faint))
            return True
        if name == "help":
            self.print_help()
        elif name == "theme":
            self.set_theme(arg)
        elif name == "sources":
            self.show_sources = arg != "off"
            cfg = read_config(); cfg["sources"] = self.show_sources
            write_config(cfg)
            print(theme.paint(f"  sources {'on' if self.show_sources else 'off'}",
                              self.theme.dim))
        elif name == "history":
            self.print_history()
        elif name == "forget":
            # settled ONLY - history and last_doc are untouched. This drops
            # the picks that steer later answers; it does not erase what was
            # asked, and the conversation you're in the middle of keeps
            # working. Nothing here is disk state, so there is nothing to
            # un-write.
            n = len(self.settled)
            self.settled.clear()
            print(theme.paint(
                f"  forgot {n} picked record{'' if n == 1 else 's'}"
                if n else "  forgot nothing - no records were picked",
                self.theme.dim))
        else:
            print(theme.paint(f"  no such command: /{name}   (/help)",
                              self.theme.faint))
        return True

    def print_help(self) -> None:
        t = self.theme
        print()
        for cmd, what in HELP:
            print("  " + theme.paint(cmd.ljust(18), t.accent)
                  + theme.paint(what, t.dim))
        print()
        print(theme.paint("  Ask about a character, a creator, or a comic. Field "
                          "questions are answered", t.faint))
        print(theme.paint("  from the record in code; anything open-ended is "
                          "written by the model.", t.faint))

    def print_history(self) -> None:
        """What was asked, and which record answered.

        The headline, not the Page: title, because this is what the user saw
        on screen - `/history` is a record of the conversation, not of
        retrieval.
        """
        t = self.theme
        print()
        if not self.history:
            print(theme.paint("  nothing yet", t.faint))
            return
        for question, _doc, head in self.history[-self.HISTORY_SHOWN:]:
            print("  " + theme.paint(question[:40].ljust(42), t.text)
                  + theme.paint(head or "no record", t.faint))

    HIGHLIGHT = "#1b2338"

    def theme_row(self, key: str, cursor: bool) -> str:
        """One row of the picker. The cursor row is a filled band, as in the
        artboards - an arrow alone is easy to lose in six coloured lines."""
        other = theme.THEMES[key]
        mark = "▸" if cursor else " "
        row = (f"  {mark} " + theme.paint(key.ljust(12), other.accent, bold=cursor)
               + theme.paint(theme.BLURBS[key].ljust(34), other.dim)
               + theme.paint("██", other.accent)
               + theme.paint("██", other.second) + "  ")
        if not cursor or not theme.COLOUR_ENABLED:
            return row
        # The band has to be re-applied after every reset the segments emit.
        band = theme.bg(self.HIGHLIGHT)
        return band + row.replace(theme.RESET, theme.RESET + band) + theme.RESET

    def pick_theme(self) -> None:
        """The picker, driven by the arrow keys.

        Retyping the whole command to try a skin is the wrong shape for a
        thing whose whole point is trying skins. Falls back to a printed list
        wherever a keystroke cannot be read - a pipe, or a dumb terminal."""
        keys = theme.names()
        start = keys.index(self.theme.name) if self.theme.name in keys else 0
        print()
        if not render.interactive():
            for key in keys:
                print(self.theme_row(key, key == self.theme.name))
            print()
            print(theme.paint("  /theme <name> to switch", self.theme.faint))
            return
        print(theme.paint("  ↑↓ move · ⏎ apply · esc cancel", self.theme.faint))

        def draw(index):
            for i, key in enumerate(keys):
                print(self.theme_row(key, i == index))

        chosen = render.pick(keys, start, draw)
        if chosen < 0:
            print(theme.paint("  kept " + self.theme.name, self.theme.faint))
            return
        self.apply_theme(theme.THEMES[keys[chosen]])

    def set_theme(self, name: str) -> None:
        t = self.theme
        if not name:
            self.pick_theme()
            return
        chosen = theme.get(name)
        if chosen is None:
            print(theme.paint(f"  no theme called {name!r} — "
                              f"{', '.join(theme.names())}", t.faint))
            return
        self.apply_theme(chosen)

    def choice_row(self, row, cursor: bool) -> str:
        """One offered record: its headline and how much is written about it.

        The cursor row is a chip, not an arrow - the same painted badge the
        header uses to mark the current theme (render.chip, Task 3). An
        arrow beside plain text is easy to lose in a scroll of headlines;
        a filled badge is not.
        """
        _size, head, _doc = row
        t = self.theme
        # FINDING 2026-09-05 (whole-branch review, Important 3): the spec's
        # testing section requires that no printable line exceeds the width
        # at 34/56/78/96 "for every answer shape, picker and refusal", and
        # this one had no bound at all. 6,772 of 104,097 character headlines
        # (6.51%) exceed 30 columns and so overflow at MIN_WIDTH = 34; 81
        # exceed even width 56; the longest is 406 characters. Task 6 fixed
        # exactly this class of bug for the answer head with render._fit.
        # Both branches cost 4 columns - "  " plus a chip that is always two
        # wider than its text, or a bare four-space indent - so both get the
        # same room. `_fit` runs on the UNPAINTED head: len() on a string
        # carrying escapes measures the escapes too. The row kept in
        # `last_offer` is untouched, so a truncated row is still nameable.
        head = render._fit(head, self.width - 4)
        if cursor:
            return "  " + render.chip(head, t)
        return "    " + theme.paint(head, t.text)

    def offer(self, rows) -> None:
        """Show the choice. A cursor picker on a console, a numbered list
        through a pipe - and a bare number selects either way."""
        t = self.theme
        print()
        print(theme.paint(f"  {len(rows)} records could be this. Which?",
                          t.dim))
        if not render.interactive():
            lines = []
            for i, row in enumerate(rows, 1):
                # The first row is the top-ranked candidate - the same one
                # an interactive session would start the cursor on - so it
                # gets the same chip, not just a number. render.chip() is
                # exactly two columns wider than plain text (its own
                # docstring), so its label drops two columns to match - the
                # same 2-vs-4 trade choice_row() makes between its cursor
                # and non-cursor prefixes - keeping every row the same
                # total width.
                #
                # Bounded for the same reason choice_row() is (Important 3),
                # but at width - 6, not - 4: this listing carries a two-digit
                # row number and its full stop as well as the 2-vs-4 prefix
                # trade, so the overhead is six columns either way.
                head = render._fit(row[1], self.width - 6)
                if i == 1:
                    lines.append(f"{i:>2}. " + render.chip(head, t))
                else:
                    lines.append(f"  {i:>2}. " + theme.paint(head, t.text))
            motion.wipe(lines)
            print()
            # Two columns shorter than "…or name one", and it says the same
            # thing: the old wording was 35 printable columns, one past
            # MIN_WIDTH = 34, and the spec's width rule covers the picker's
            # own chrome as well as its rows. Found by the sweep below it
            # (infer/test_terminal.py:NothingOverflowsThePicker), not by eye.
            print(theme.paint("  answer with a number, or a name", t.faint))
            self.last_offer = list(rows)
            return
        print(theme.paint("  ↑↓ move · ⏎ pick · esc cancel",
                          t.faint))

        def draw(index):
            # NOT motion.wipe here: render.pick() calls draw() again on
            # every arrow keypress, and wipe's per-line pause would stagger
            # ~0.055s x len(rows) after each one - on a ten-row offer, over
            # half a second per keystroke, in the one picker people actually
            # navigate. The chip already carries the visual change on
            # redraw; wipe belongs only to the one-shot piped listing below.
            for i, row in enumerate(rows):
                print(self.choice_row(row, i == index))

        chosen = render.pick(list(rows), 0, draw)
        if chosen < 0:
            print(theme.paint("  no pick", t.faint))
            # Escaping the cursor picker leaves the rows nameable: the user
            # saw them, and "the ultimate one" is a reasonable next thing to
            # type. This is the ONLY way the feature is reachable on a
            # console, where render.pick() blocks and nothing can be typed at
            # an open menu.
            self.last_offer = list(rows)
            return
        self.last_doc = rows[chosen][2]
        self.settle(rows[chosen][2])
        self.last_offer = None
        self.ask("who is this")

    def apply_theme(self, chosen) -> None:
        self.theme = chosen
        cfg = read_config(); cfg["theme"] = chosen.name
        write_config(cfg)
        self.header(with_banner=True)

    # -- loop ---------------------------------------------------------------

    def handle_line(self, line: str) -> None:
        """Route one non-empty, non-command line typed at the prompt.

        Pulled out of run()'s loop body so the one-turn lifetime of
        `last_offer` - it survives a miss just long enough to be read once,
        and is gone by the NEXT line no matter which way this one went - can
        be pinned in a unit test without driving `input()`. Both `self.
        last_offer = None` lines below are that contract; delete either one
        and infer/test_terminal.py's LastOfferHasAOneTurnLifetime fails.

        That "gone by the next line" claim covers only lines that reach THIS
        method. A `/command` never does - run() routes it before handle_line
        is called - so a command does NOT clear last_offer. That is
        deliberate, and differs from the old needs_a_pick(), which cancelled
        the choice on any command: it means `/history` can sit between a
        menu and a phrase that still resolves against it.
        """
        doc = self.take_reference(line)
        if doc is not None:
            self.settle(doc)
            self.last_doc = doc
            self.last_offer = None
            self.ask("who is this")
            self._record_turn(line)
            return
        # Not a reference, so it is an ordinary question - and the menu
        # closes. This is the swallow fix: a line that names no row is
        # answered instead of being intercepted.
        self.last_offer = None
        try:
            self.ask(line)
        except KeyboardInterrupt:
            print(theme.paint("\n  stopped.", self.theme.faint))
        self._record_turn(line)

    def _record_turn(self, line: str) -> None:
        """Append one turn to /history: the line the USER typed, and the
        record THIS TURN resolved to.

        Reads `self.last_turn_doc`, set fresh by every ask() call - NOT
        `self.last_doc`. last_doc deliberately persists unchanged across
        chitchat and an unresolved question, so a follow-up still has an
        entity to attach to - but NOT across a menu turn: plan() returns the
        top-ranked record as doc_id even on a picker turn (engine.py:257),
        so ask() sets last_doc (and last_turn_doc) to that row, and
        /history ends up showing the menu's top-ranked record beside a turn
        that only displayed a menu. Reading last_doc here would still be
        wrong for the chitchat case even though it happens to agree with
        last_turn_doc on the menu case: it would make a chitchat turn right
        after a resolved one show the PRIOR turn's record for a question
        that never touched it - worse than showing none, since /history's
        whole purpose is reporting what happened.
        """
        doc = self.last_turn_doc
        self.history.append(
            (line, doc, self.index.headlines[doc] if doc is not None else None))
        del self.history[:-self.HISTORY_MAX]

    def run(self) -> int:
        # Line editing and history, where the platform has it. Windows gets
        # both from the console itself; elsewhere it takes this import.
        try:
            import readline                            # noqa: F401
        except ImportError:
            pass
        while True:
            try:
                print()
                print(render.speaker(self.theme, "you", self.theme.accent))
                line = input("  ").strip()
                if line and not sys.stdin.isatty():
                    # A console echoes what you type; a pipe does not, so a
                    # scripted session read back as answers with no questions.
                    print(line)
            except (EOFError, KeyboardInterrupt):
                print()
                print(theme.paint("  bye.", self.theme.faint))
                return 0
            if not line:
                continue
            if line.startswith("/"):
                if not self.command(line):
                    print(theme.paint("  bye.", self.theme.faint))
                    return 0
                continue
            self.handle_line(line)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="edith", description="a Marvel terminal")
    ap.add_argument("--ask", help="answer one question and exit")
    ap.add_argument("--theme", default="", help=f"one of: {', '.join(theme.names())}")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--trace", action="store_true",
                    help="machine mode for infer/run_cases.py: print which "
                         "record answered, to stderr, and pin the transcript")
    # Registration only - args.no_anim is never read. The gate lives in
    # motion.enable(), which scans raw argv itself; do not "clean up" this
    # entry for looking unused, or `--no-anim` dies with "unrecognized
    # arguments" instead of doing nothing.
    ap.add_argument("--no-anim", action="store_true",
                    help="print everything at once, with no motion")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Motion rides the same test colour already runs: a TTY a person is
    # watching. Called before boot so the banner wipe obeys it.
    motion.enable(argv=sys.argv if argv is None else list(argv))

    theme.enable()
    term = Terminal(args.theme, Path(args.ckpt), trace=args.trace)
    if not term.boot():
        return 1
    if args.ask:
        print()
        print(render.speaker(term.theme, "you", term.theme.accent))
        print("  " + args.ask)
        term.ask(args.ask)
        print()
        return 0
    return term.run()


if __name__ == "__main__":
    raise SystemExit(main())
