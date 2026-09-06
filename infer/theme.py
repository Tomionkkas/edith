"""Terminal palettes.

A theme changes colour, the spinner glyph, its wording and the speaker's
label. It never changes what is retrieved, which of the record's fields are
printed, or a single character of the answer - the model was trained to
answer, not to impersonate, and a theme that rewrote its sentences would be
inventing text the record does not support.

Colours are taken from the six artboards in `design/`. The one deliberate
departure: the mockups paint a full-bleed background, and the PAGE does not.
Repainting the terminal's background fights the user's own colour scheme,
breaks on resize, and looks wrong the moment someone runs a light profile.
Backgrounds are painted only where they are a component - the `chip` and
`caption` pairs below, and nothing else - so the accents still carry the
identity across the rest of the screen.

    py infer/theme.py            # show every palette
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache

RESET = "\033[0m"
BOLD = "\033[1m"
ANSI_RE = re.compile(r"\033\[[0-9;]*m")


@dataclass(frozen=True)
class Theme:
    name: str
    label: str          # what the assistant is called in the transcript
    accent: str         # the character's colour: banner, highlights
    second: str         # the assistant's prompt glyph
    text: str           # answer prose
    dim: str            # labels, metadata
    faint: str          # box rules, hints
    glyph: str          # marks the retrieval line
    verb: str           # "... 187,580 records"
    sigil: tuple = ()   # six 13-char rows locked up beside the wordmark
    rule: str = "─"     # the divider's character
    loud_rule: bool = False   # rule in the accent colour, not the faint one
    quip: str = ""      # optional aside on the header line
    # Five colours could not carry an identity: every theme read as one
    # terminal recoloured. `bright` sits ABOVE text so the subject's own
    # name can lead, and the two pairs below paint a background - the first
    # use of bg() anywhere in the program.
    bright: str = ""            # the subject's name; falls back to `text`
    chip: tuple = ()            # (background, foreground) - a painted badge
    caption: tuple = ()         # (background, foreground) - a painted line


def _wide(rows) -> tuple:
    """A 3x5 sigil centred in the 6x13 field the lockup uses.

    The three themes nobody has drawn an emblem for yet still have to lock up
    beside a six-row wordmark. Centring what they have beats inventing a
    Hulk emblem in a phase that is not about Hulk.
    """
    body = [r.center(13) for r in rows]
    return tuple([" " * 13] + body + [" " * 13] * (6 - len(body) - 1))


THEMES = {
    "spider-man": Theme(
        name="spider-man", label="spidey",
        accent="#e62429", second="#2f63e0", text="#c6d1e6",
        dim="#6c7a99", faint="#2c3c62",
        glyph="✦", verb="spun through",
        bright="#f3f7ff", chip=("#16295c", "#8fb4ff"),
        sigil=("╲╲   ▄▄▄   ╱╱",
               " ╲╲▄█████▄╱╱ ",
               "══╲███████╱══",
               "══╱███████╲══",
               " ╱╱▀█████▀╲╲ ",
               "╱╱   ▀▀▀   ╲╲")),
    "wolverine": Theme(
        name="wolverine", label="logan",
        accent="#f5c518", second="#2c4a8f", text="#e6e0cf",
        dim="#6b6350", faint="#4a4433",
        glyph="╱╱╱", verb="tore through",
        sigil=_wide(("╱╱╱  ", " ╱╱╱ ", "  ╱╱╱")),
        rule="╌", loud_rule=True),
    "deadpool": Theme(
        name="deadpool", label="pool",
        accent="#e8323b", second="#eee6da", text="#d8cec4",
        dim="#8a7269", faint="#4b3134",
        glyph="◉", verb="rifled through",
        bright="#fffaf2", chip=("#4a1114", "#ff8f8f"),
        caption=("#f2c94c", "#1a1206"),
        sigil=("╲╲  ▄▄▄▄▄  ╱╱",
               " ╲╲█▀▀█▀▀█╱╱ ",
               " ╱╱█▄▄█▄▄█╲╲ ",
               "╱╱ ███████ ╲╲",
               "   ▀█████▀   ",
               "     ▀▀▀     "),
        quip="yes, the model is small. so is my attention span."),
    "moon-knight": Theme(
        name="moon-knight", label="spector",
        accent="#f5f2ea", second="#c9a227", text="#bcc2ca",
        dim="#69707a", faint="#31363e",
        glyph="☾", verb="walked through",
        bright="#ffffff", chip=("#2b2410", "#dcb84d"),
        sigil=("  ▄▄████▄▄   ",
               " ████▀▀▀▀▀   ",
               "▐████        ",
               "▐████        ",
               " ████▄▄▄▄▄   ",
               "  ▀▀████▀▀   "),
        loud_rule=True,
        quip="one of us is answering."),
    "venom": Theme(
        name="venom", label="we",
        accent="#ededed", second="#8cc63f", text="#ededed",
        dim="#565658", faint="#3a3a3c",
        glyph="◍", verb="we tasted",
        sigil=_wide(("╲   ╱", " ◍◍◍ ", "╱   ╲")), rule="▰", loud_rule=True,
        quip="we are in your terminal"),
    "hulk": Theme(
        name="hulk", label="hulk",
        accent="#4c9a2a", second="#6b3fa0", text="#dfe8dc",
        dim="#6a7568", faint="#3b4239",
        glyph="◆", verb="smashed through",
        sigil=_wide(("  ◆  ", " ◆◆◆ ", "  ◆  ")), rule="▬", loud_rule=True),
    "plain": Theme(
        name="plain", label="edith",
        accent="", second="", text="", dim="", faint="",
        glyph="▸", verb="searched"),
}

DEFAULT = "spider-man"

BLURBS = {
    "spider-man": "web lattice, red on midnight blue",
    "wolverine": "claw rules, yellow on tar",
    "deadpool": "breaks the fourth wall",
    "moon-knight": "bone white and Khonshu gold, no hue at all",
    "venom": "speaks as \"we\"",
    "hulk": "green and purple, loud",
    "plain": "no theme, just the terminal",
}


def names() -> list:
    """Theme names in the order the picker shows them."""
    return list(THEMES)


def get(name: str):
    """The theme called `name`, or None. Accepts what people actually type."""
    if not name:
        return None
    key = name.strip().lower().replace("_", "-").replace(" ", "-")
    return THEMES.get(key) or THEMES.get(ALIASES.get(key, ""))


def bright_of(t) -> str:
    """The colour the subject's own name is printed in."""
    return t.bright or t.text


def chip_of(t) -> tuple:
    """(background, foreground) for a painted badge.

    Derived rather than required, so a theme nobody has designed yet still
    renders: the faint rule colour behind ordinary text reads as a plate.
    """
    return t.chip if t.chip else (t.faint, t.text)


def caption_of(t) -> tuple:
    """(background, foreground) for a painted line, or () for no caption.

    Empty is meaningful: a theme with no caption keeps its quip on the
    header line, exactly where it is today.
    """
    return t.caption


ALIASES = {
    "spiderman": "spider-man", "spidey": "spider-man", "spider": "spider-man",
    "logan": "wolverine", "wolvie": "wolverine",
    "dp": "deadpool", "wade": "deadpool",
    "moonknight": "moon-knight", "moon": "moon-knight",
    "khonshu": "moon-knight", "spector": "moon-knight",
    "marc": "moon-knight",
    "eddie": "venom", "symbiote": "venom",
    "banner": "hulk", "green": "hulk",
    "none": "plain", "default": "spider-man", "off": "plain",
}


# ------------------------------------------------------------------ colour

# The xterm-256 colour cube's six levels per channel, and the 24 greys above
# it. Indices 0-15 are deliberately not candidates: they are whatever the
# user's profile has set them to, so matching against them would pick a colour
# we cannot predict.
_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)


@lru_cache(maxsize=512)
def _to_256(r: int, g: int, b: int) -> int:
    """The nearest xterm-256 index to an exact colour.

    Considers the 6x6x6 cube and the grey ramp and takes whichever is closer -
    the greys matter, because `faint` and `dim` are near-neutral and the cube's
    nearest neighbour to a grey can be visibly tinted.
    """
    ci = tuple(min(range(6), key=lambda i: abs(_CUBE_LEVELS[i] - v))
               for v in (r, g, b))
    cube_d = sum((v - _CUBE_LEVELS[i]) ** 2 for v, i in zip((r, g, b), ci))

    gi = min(range(24), key=lambda i: abs(8 + 10 * i - (r + g + b) / 3))
    grey = 8 + 10 * gi
    grey_d = sum((v - grey) ** 2 for v in (r, g, b))

    if grey_d < cube_d:
        return 232 + gi
    return 16 + 36 * ci[0] + 6 * ci[1] + ci[2]


def _channels(colour: str) -> tuple:
    h = colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _escape(colour: str, layer: int) -> str:
    """One SGR escape, 24-bit or 256-colour depending on the terminal.

    A terminal that does not understand `38;2;r;g;b` does not ignore it: it
    drops the `38;2` and reads what is left as separate legacy codes, where
    40-47 mean BACKGROUND. Terminal.app renders the whole palette that way -
    `#e62429` ends in 41, red background, and `#2f63e0` starts with 47, white
    background - so the wordmark filled solid and the rules became bars. The
    256-colour form it does understand, so we send that instead.
    """
    r, g, b = _channels(colour)
    if TRUECOLOR:
        return f"\033[{layer};2;{r};{g};{b}m"
    return f"\033[{layer};5;{_to_256(r, g, b)}m"


def rgb(colour: str) -> str:
    """A #rrggbb string as an ANSI foreground escape; "" for no colour."""
    if not colour:
        return ""
    return _escape(colour, 38)


def bg(colour: str) -> str:
    """A #rrggbb string as an ANSI BACKGROUND escape."""
    if not colour or not COLOUR_ENABLED:
        return ""
    return _escape(colour, 48)


def paint(text: str, colour: str, bold: bool = False) -> str:
    """`text` in `colour`. Uncoloured when the palette has none, so the plain
    theme costs no special case anywhere else."""
    if not colour or not COLOUR_ENABLED:
        return text
    weight = BOLD if bold else ""
    return f"{weight}{rgb(colour)}{text}{RESET}"


def strip(text: str) -> str:
    """Text without escapes - what a width calculation has to measure."""
    return ANSI_RE.sub("", text)


COLOUR_ENABLED = True


def detect_truecolor() -> bool:
    """Whether this terminal can be sent 24-bit colour.

    An allowlist, not a denylist: guessing wrong the optimistic way is what
    produced the filled blocks, and 256-colour on a terminal that could have
    done better is a slightly different red nobody will notice.

    EDITH_COLOR=truecolor|256 forces it either way, for a terminal that
    reports nothing useful.
    """
    forced = os.environ.get("EDITH_COLOR", "").lower()
    if forced in ("truecolor", "24bit", "24"):
        return True
    if forced in ("256", "8bit"):
        return False
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return True
    # Windows has been 24-bit since conhost gained VT processing in 1703, and
    # Windows Terminal sets no COLORTERM - an allowlist alone would downgrade
    # the one platform where this rendered correctly to begin with.
    return sys.platform == "win32"


TRUECOLOR = detect_truecolor()


def use_utf8() -> None:
    """Make stdout carry box-drawing characters.

    Python picks the console's legacy code page on Windows - cp1250 on this
    machine - and the FIRST `┌` raises UnicodeEncodeError, taking the whole
    terminal down. Reconfiguring is not optional decoration.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):        # already wrapped, or a pipe
            pass


def enable(stream=None) -> bool:
    """Turn on colour, and on Windows turn on the console's VT parsing.

    Windows consoles print escape codes literally until
    ENABLE_VIRTUAL_TERMINAL_PROCESSING is set, so a terminal that looks fine
    everywhere else arrives full of `[38;2;` on the user's own machine.
    NO_COLOR is honoured because it costs one line.
    """
    global COLOUR_ENABLED, TRUECOLOR
    TRUECOLOR = detect_truecolor()
    use_utf8()
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") or not getattr(stream, "isatty", lambda: False)():
        COLOUR_ENABLED = False
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:                                 # noqa: BLE001
            pass
    COLOUR_ENABLED = True
    return True


def main() -> int:
    enable()
    for name in names():
        t = THEMES[name]
        swatch = paint("██", t.accent) + paint("██", t.second)
        print(f"  {paint(name.ljust(12), t.accent, bold=True)} {swatch}  "
              f"{paint(BLURBS[name], t.dim)}")
        print(f"    {paint(t.glyph, t.accent)} {paint(t.verb, t.dim)} "
              f"{paint('187,580', t.text)} {paint('records', t.dim)}"
              f"    {paint('▌', t.second)}{paint(' ' + t.label, t.dim)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
