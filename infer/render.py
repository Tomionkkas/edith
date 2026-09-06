"""What the terminal draws: the banner, the answer block, and streamed text.

Kept apart from the REPL so it can be tested without a model, a GPU or a
terminal. Everything here is a pure function of (text, theme, width) except
TidyStream, which is a pure function of the order its chunks arrive in - and
that is exactly what its tests pin.
"""
from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("theme", Path(__file__).parent / "theme.py")
theme = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass looks its class up in sys.modules,
# and an unregistered module makes it fail with an unrelated AttributeError.
sys.modules.setdefault("theme", theme)
_spec.loader.exec_module(theme)

# EDITH, in the half-block face the artboards use. The banner is the PRODUCT,
# so it is the one thing a theme never changes - only its colour.
# Five rows, not two. The half-block face from the artboards is legible at
# 19px in a browser and turns to mush at a console's line height - the first
# thing the user said about it was that they could not read it.
# EDITH, outlined with a drop shadow. The five-row solid face before this was
# legible but read as a wall; the artboards' two-row half-block face was worse,
# being drawn for a browser's 19px rather than a console line height.
# The letters carry ONE blank column between each pair. Without it the
# banner has no blank columns at all, and a shadow `╗` landing next to a
# solid `█` reads tighter than one landing next to a space - which is what
# looked crooked. test_render.BannerSpacing pins the invariant.
BANNER = (
    "███████╗ ██████╗  ██╗ ████████╗ ██╗  ██╗",
    "██╔════╝ ██╔══██╗ ██║ ╚══██╔══╝ ██║  ██║",
    "█████╗   ██║  ██║ ██║    ██║    ███████║",
    "██╔══╝   ██║  ██║ ██║    ██║    ██╔══██║",
    "███████╗ ██████╔╝ ██║    ██║    ██║  ██║",
    "╚══════╝ ╚═════╝  ╚═╝    ╚═╝    ╚═╝  ╚═╝",
)
BANNER_WIDTH = 40

# Which record fields reached the sources box, in priority order, with the
# labels a reader expects rather than the corpus's own. NOT authoritative for
# the whole-entity answer as of 4.12 - that is facts.PROFILE_FIELDS, a
# separate, deliberately-not-unified table (see its comment in facts.py).
#
# DEAD IN PRODUCTION. BOX_FIELDS feeds sources_box() below and nothing else,
# and nothing calls sources_box(): terminal.py's ask() prints the field block
# instead, and `/sources off` falls back to facts.profile() PROSE, not to
# this box. Both are kept because their tests still pin the one invariant the
# box ever had - that a row's printed width is exactly the width asserted -
# and deleting them would delete that. An earlier version of this comment
# claimed `/sources off` as a live caller in the same breath as saying it
# falls back to prose; it cannot be both.
BOX_FIELDS = (
    ("Created by", "Created by"),
    ("First appearance", "First appearance"),
    ("Full name", "Real name"),
    ("Formal name", "Formal name"),
    ("Powers", "Powers"),
    ("Occupation", "Occupation"),
    ("Base of operations", "Base"),
)
BOX_ROWS = 4
MIN_WIDTH = 34

# Continuity identifies a character, a team, an event, an arc. It does not
# identify a hammer: Mjolnir exists in Earth-616 and Earth-199999 and is one
# hammer. `unknown` keeps the badge, so a corpus predating the Kind line
# renders exactly as it did.
BADGE_KINDS = frozenset(
    ("character", "team", "event", "story arc", "issue", "unknown"))

# Duplicated from crawl/curate.py's is_main_continuity_reality(), not
# imported: render.py stands alone. Earth-616 is the main comics continuity
# and the unmarked default, so a record listing it among several realities
# (an item can exist in five) is main continuity and gets no badge.
MAIN_CONTINUITY = "Earth-616"


def record_kind(record: str) -> str:
    """Read the curation-written Kind line. Duplicated rather than imported:
    infer/ must stay importable without an index, so it must not import
    retrieve/."""
    for line in record.split(chr(10))[1:]:
        if line.startswith(("History:", "Synopsis")):
            break
        if line.startswith("Kind: "):
            return line[6:].strip() or "unknown"
    return "unknown"


LOCKUP_WIDTH = 56          # 13 emblem + 3 gutter + 40 wordmark


def banner(t, width: int = 78) -> str:
    """The wordmark, with the character's emblem locked up beside it.

    The banner is the PRODUCT: a theme colours it and stands its own mark
    next to it, but never changes the letters. That is what stops the
    terminal looking like seven different tools.

    Three tiers, because it never had a width rule and MIN_WIDTH is 34 -
    the wordmark alone has been overflowing a minimum-width terminal all
    along.
    """
    if width < BANNER_WIDTH:
        return theme.paint("E D I T H", t.accent, bold=True)
    rows = [theme.paint(line, t.accent, bold=True) for line in BANNER]
    # zip() would SHORTEN rows if a sigil were ever less than six rows tall,
    # dropping wordmark rows rather than just decoration. A mis-sized emblem
    # is therefore not drawn at all: no emblem beats half a wordmark.
    if width >= LOCKUP_WIDTH and len(t.sigil) == len(rows):
        rows = [theme.paint(mark, t.second) + "   " + row
                for mark, row in zip(t.sigil, rows)]
    return "\n".join(rows)


# ------------------------------------------------------------------- text

def wrap(text: str, width: int, indent: str = "  ") -> str:
    """Greedy word wrap. Used by TidyStream too, so the streamed answer and
    the finished one cannot disagree about where the lines break."""
    words = text.split()
    if not words:
        return ""
    room = max(width - len(indent), 1)
    lines, line = [], words[0]
    for word in words[1:]:
        if len(line) + 1 + len(word) <= room:
            line += " " + word
        else:
            lines.append(line)
            line = word
    lines.append(line)
    return "\n".join(indent + ln for ln in lines)


class TidyStream:
    """Print a streamed answer already cleaned up.

    `tidy` rewrites text after the fact - ".." becomes "." and "They works"
    becomes "They work" - and a terminal cannot un-print. So nothing goes to
    the screen until no later token can change it, which is everything up to
    the last whitespace: every rewrite is confined to one word, and the two
    that span a space ("They works") only ever alter the second.
    """

    def __init__(self, write, tidy, width: int = 76, indent: str = "  "):
        self._write = write
        self._tidy = tidy
        self._width = width
        self._indent = indent
        self._raw = ""
        self._shown = 0          # characters of the TIDIED text already out
        self._column = 0         # for wrapping across writes
        self._started = False

    def write(self, chunk: str) -> None:
        self._raw += chunk
        cleaned = self._tidy(self._raw)
        cut = cleaned.rfind(" ") + 1
        if cut > self._shown:
            self._emit(cleaned[self._shown:cut])
            self._shown = cut

    def flush(self) -> None:
        pass

    def close(self) -> None:
        """Emit the tail, which no later token will now change."""
        cleaned = self._tidy(self._raw)
        self._emit(cleaned[self._shown:])
        self._shown = len(cleaned)

    def _emit(self, text: str) -> None:
        room = max(self._width - len(self._indent), 1)
        for word in text.split():
            if not self._started:
                self._write(self._indent)
                self._column = 0
                self._started = True
            elif self._column + 1 + len(word) > room:
                self._write("\n" + self._indent)
                self._column = 0
            else:
                self._write(" ")
                self._column += 1
            self._write(word)
            self._column += len(word)


# -------------------------------------------------------------------- box

def fields(record: str) -> dict:
    """Schema lines of a record, stopping where the prose starts."""
    out = {}
    for line in record.split("\n")[1:]:
        if line.startswith(("History:", "Synopsis")):
            break
        label, sep, value = line.partition(": ")
        if sep and value.strip():
            out.setdefault(label.strip(), value.strip())
    return out


def _clip(text: str, room: int) -> str:
    return text if len(text) <= room else text[:max(room - 1, 0)].rstrip() + "…"


def sources_box(record: str, t, width: int = 76) -> str:
    """The retrieved record, as the box that makes the grounding visible.

    Empty when the record carries no usable field - a box with nothing in it
    would claim more than we have.
    """
    width = max(int(width), MIN_WIDTH)
    found = fields(record)
    head_bare = record.split(chr(10), 1)[0].split(" (")[0].strip().lower()

    rows = [(label, found[key]) for key, label in BOX_FIELDS
            if key in found
            # "Real name  Civil War" under a headline reading "Civil War
            # (Event)" is noise, and it pushed a real field off the box.
            and not (key in ("Full name", "Formal name")
                     and found[key].strip().lower() == head_bare)][:BOX_ROWS]
    if not rows:
        return ""

    head = record.split("\n", 1)[0].strip()
    reality = ""
    if record_kind(record) in BADGE_KINDS:
        realities = [r.strip() for r in found.get("Reality", "").split(";") if r.strip()]
        # A record that LISTS SEVERAL realities and includes Earth-616 among
        # them is main continuity - the same rule crawl/curate.py's
        # is_main_continuity_reality() applies to the headline suffix -
        # rather than an alternate one, so it gets no badge: `Godbomb (Story
        # Arc)` with `Earth-14412; Earth-616` badged the ALTERNATE reality.
        # A lone "Earth-616" is unaffected - it is the ordinary, unmarked
        # case, and its badge already read correctly. Exact equality per
        # stripped part, never a substring: Earth-6160 and Earth-61610 are
        # both real realities in this corpus and must not match Earth-616.
        if not (len(realities) > 1 and MAIN_CONTINUITY in realities):
            reality = realities[0] if realities else ""
    inner = width - 4                                  # "│ " + inner + " │"
    pad = max(len(label) for label, _ in rows)

    # Rows are built as (text, colour) segments and painted at the end, so the
    # width is measured on exactly the characters that get printed. Assembling
    # coloured strings and measuring them separately is how a box drifts by
    # one column and nobody notices until it is on screen.
    tail = [(f" {reality} ", t.second), ("─", t.faint)] if reality else []
    tail_width = sum(len(s) for s, _ in tail)
    head = _clip(head, max(inner - tail_width - 2, 1))
    fill = max(inner - len(head) - tail_width - 1, 0)
    rendered = [[("┌─ ", t.faint), (head, t.text), (" " + "─" * fill, t.faint)]
                + tail + [("┐", t.faint)]]
    for label, value in rows:
        value = _clip(value, inner - pad - 1)
        rendered.append([("│ ", t.faint), (label.ljust(pad), t.dim),
                         (" " + value.ljust(inner - pad - 1), t.text),
                         (" │", t.faint)])
    rendered.append([("└" + "─" * (width - 2) + "┘", t.faint)])

    for row in rendered:
        assert sum(len(s) for s, _ in row) == width, row
    return "\n".join("".join(theme.paint(s, c) for s, c in row)
                     for row in rendered)


VALUE_LINES = 3            # a value wraps this far, then stops


def _title(record: str) -> tuple:
    """(person, aside) from a record's headline and Page: line.

    The headline is the STORY-CURRENT alias, which is the least reliable
    thing in the record: Henry McCoy's headline reads `Chairman` and Victor
    von Doom's reads `Emperor Doom`. The `Page:` title is the identity. So
    the person leads and the headline becomes an aside - a presentation fix
    for a curation wart, touching no curated file.
    """
    head = record.split("\n", 1)[0].strip()
    page = field_value_of(record, "Page") or head
    person = page.split(" (")[0].strip()
    alias = head.split(" (")[0].strip()
    return person, ("" if alias.lower() == person.lower() else alias)


def field_value_of(record: str, label: str) -> str:
    """One schema field's value, or "". Local so render imports no facts."""
    return fields(record).get(label, "")


def _wrap_value(value: str, room: int) -> list:
    """A value across at most VALUE_LINES lines, cut only at a separator.

    `_clip` cut mid-token and produced `terrorist;…`. Truncation is allowed
    to lose a clause; it is not allowed to lose half a word.

    textwrap does the wrapping because three things must hold at once and
    the hand-rolled first draft got all three wrong. Whether the value was
    truncated is now TRACKED rather than inferred by comparing a
    whitespace-normalised rejoin against the original - that reported a
    false `…` on any value containing a double space. A single token longer
    than the room is hard-broken rather than allowed to overflow the margin:
    at width 34 the room is 12 columns and `Mutant/Atlantean` is 16. And an
    over-long first token no longer emits a leading blank line.
    """
    room = max(int(room), 1)
    lines = textwrap.wrap(value, room, break_long_words=True,
                          break_on_hyphens=False)
    if not lines:
        return []
    if len(lines) <= VALUE_LINES:
        return lines
    tail = lines[VALUE_LINES - 1]
    mark = max(tail.rfind(";"), tail.rfind(","))
    if mark > 0:
        body = tail[:mark].rstrip(" ;,")
    else:
        space = tail.rfind(" ")
        body = tail[:space].rstrip(" ;,") if space > 0 else ""
    if not body:
        # One unbroken token fills the boundary line. Slicing into it would
        # lose half a word - the defect this function exists to remove - so
        # drop the line and let the ellipsis stand alone.
        return lines[:VALUE_LINES - 1] + ["…"]
    return lines[:VALUE_LINES - 1] + [body + "…"]


def _fit(text: str, room: int) -> str:
    """`text` in at most `room` columns, cut at a word boundary if there is one.

    A name is not a clause, so there is no separator to cut at - but there is
    usually a space. Only a single unbroken token longer than the room gets
    sliced, and then there is no alternative that still fits.
    """
    if len(text) <= room:
        return text
    if room <= 1:
        return "…"[:room]
    cut = text.rfind(" ", 0, room)
    if cut > 0:
        return text[:cut].rstrip(" ;,") + "…"
    return text[:room - 1] + "…"


def answer_rows(record: str, rows, t, width: int) -> list:
    """The record as the answer: a title line, then label/value pairs.

    No border. The box's frame grouped what alignment groups for free, and
    it cost four columns that the values needed.
    """
    person, alias = _title(record)
    reality = ""
    if record_kind(record) in BADGE_KINDS:
        parts = [r.strip() for r in field_value_of(record, "Reality").split(";")
                 if r.strip()]
        if not (len(parts) > 1 and MAIN_CONTINUITY in parts):
            reality = parts[0] if parts else ""

    # DEVIATION from the brief (reported in task-6-report.md): the brief's
    # head line has no width bound at all. For this task's own fixture -
    # person "Henry McCoy", alias "Chairman", reality "Earth-616" - the full
    # head is 36 printable columns, which overflows width=34 by 2. Rather
    # than character-clip it (the exact mid-word defect this task removes),
    # an optional segment that would push the head past `width` is dropped
    # whole instead of shown partially. The person itself is bounded with
    # `_fit`: 707 of 126,652 corpus `Page:` records carry a name that already
    # exceeds width=34 with the indent, and 8 exceed even width=96.
    head = "  " + theme.paint(_fit(person, width - 2),
                              theme.bright_of(t), bold=True)
    # Reality first, alias second: `_title`'s own docstring calls the alias
    # "the least reliable thing in the record", while the reality chip is
    # what tells a reader which version of the character this is - so when
    # both together overflow, the alias is what gets dropped.
    if reality:
        candidate = head + "  " + chip(reality, t)
        if len(theme.strip(candidate)) <= width:
            head = candidate
    if alias:
        candidate = head + "  " + theme.paint(alias, t.dim)
        if len(theme.strip(candidate)) <= width:
            head = candidate
    out = [head, ""]

    pad = max((len(label) for label, _ in rows), default=0)
    indent = 4 + pad + 2
    room = max(width - indent, 1)
    for label, value in rows:
        wrapped = _wrap_value(value, room)
        if not wrapped:
            continue
        out.append("    " + theme.paint(label.ljust(pad), t.dim)
                   + "  " + theme.paint(wrapped[0], t.text))
        for line in wrapped[1:]:
            out.append(" " * indent + theme.paint(line, t.text))
    return out


def terminal_width(default: int = 78) -> int:
    try:
        import shutil
        return max(min(shutil.get_terminal_size().columns - 2, 96), MIN_WIDTH)
    except Exception:                                   # noqa: BLE001
        return default


def rule(t, width: int) -> str:
    """The divider, in the theme's own character.

    Wolverine's is dashed, Hulk's is a thick bar, Venom's is a slime run - the
    artboards give each frame its own rule, and it is most of what makes them
    read as different terminals rather than one terminal recoloured.
    """
    mark = t.rule or "─"
    line = (mark * ((width // len(mark)) + 1))[:width]
    return theme.paint(line, t.accent if t.loud_rule else t.faint)


def chip(text: str, t) -> str:
    """A padded badge on a painted background.

    Always exactly two columns wider than `text`. Without colour it falls
    back to brackets, which are also two columns - so a line's width does
    not depend on whether the terminal paints.
    """
    back, fore = theme.chip_of(t)
    if not theme.COLOUR_ENABLED or not back:
        return f"[{text}]"
    return (theme.bg(back) + theme.rgb(fore) + " " + text + " "
            + theme.RESET)


def caption(text: str, t, width: int) -> str:
    """A full-width painted line - the comic-book narration box.

    Chrome that comments on the record. It is never the answer, and never a
    word of the record: the model was trained to answer, not to impersonate.

    The pair is unpacked BEFORE the guard, and the guard tests the
    background, the way chip() does. Testing the tuple's truthiness instead
    would let a half-filled pair through and emit a foreground escape with
    no background behind it.
    """
    pair = theme.caption_of(t)
    back, fore = pair if len(pair) == 2 else ("", "")
    body = f" {text} "
    body = body[:width] if len(body) > width else body.ljust(width)
    if not theme.COLOUR_ENABLED or not back:
        return body
    return theme.bg(back) + theme.rgb(fore) + body + theme.RESET


def speaker(t, who: str, colour: str) -> str:
    return theme.paint("▌", colour) + theme.paint(" " + who, t.dim)


def write(text: str = "") -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


# ------------------------------------------------------------------ picker

UP, DOWN, ENTER, ESC, OTHER = "up", "down", "enter", "esc", "other"


def read_key():
    """One keystroke as UP / DOWN / ENTER / ESC / OTHER.

    Windows and POSIX disagree about everything here: msvcrt hands back arrows
    as a two-byte sequence led by \xe0, and a POSIX terminal has to be put in
    raw mode to see a keystroke at all. Isolated so the picker itself stays a
    pure state machine that can be tested.
    """
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):                  # arrow prefix
            code = msvcrt.getwch()
            return {"H": UP, "P": DOWN}.get(code, OTHER)
        if ch in ("\r", "\n"):
            return ENTER
        if ch == "\x1b":
            return ESC
        if ch == "\x03":
            raise KeyboardInterrupt
        return OTHER
    import termios
    import tty
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            if sys.stdin.read(1) == "[":
                return {"A": UP, "B": DOWN}.get(sys.stdin.read(1), OTHER)
            return ESC
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    if ch in ("\r", "\n"):
        return ENTER
    if ch == "\x03":
        raise KeyboardInterrupt
    return OTHER


def move(index: int, key: str, count: int) -> int:
    """Where the cursor goes. Wraps, so holding one arrow reaches everything."""
    if key == UP:
        return (index - 1) % count
    if key == DOWN:
        return (index + 1) % count
    return index


def interactive() -> bool:
    """True when a keystroke picker is possible: both ends must be a console."""
    return bool(getattr(sys.stdin, "isatty", lambda: False)()
                and getattr(sys.stdout, "isatty", lambda: False)())


def pick(rows, current: int, draw, keys=None) -> int:
    """Run a cursor over `rows`, redrawing in place. Returns -1 on escape.

    `draw(index)` prints the whole list; the cursor is moved back over it with
    one escape sequence rather than clearing the screen, so the transcript
    above the picker is left alone.
    """
    index = current
    source = iter(keys) if keys is not None else None
    draw(index)
    while True:
        key = next(source, ESC) if source is not None else read_key()
        if key == ENTER:
            return index
        if key == ESC:
            return -1
        moved = move(index, key, len(rows))
        if moved != index or key in (UP, DOWN):
            index = moved
            write(f"\033[{len(rows)}A")              # back to the first row
            draw(index)
