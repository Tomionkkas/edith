"""Motion, and the one flag that turns all of it off.

Every function here is a stagger or a one-line redraw. Nothing enters the
alternate screen buffer, nothing addresses the cursor beyond `\\r`, and
nothing is load-bearing: with ENABLED false, every call prints exactly the
bytes it printed before this module existed, with no delay.

That is not politeness, it is the contract. infer/run_cases.py pipes
questions into the terminal and reads its stdout, and
infer/test_terminal_session.py spawns it as a subprocess. A pipe is not a
TTY, so both get the still version for free.

What is NOT here, deliberately: a scan bar and a typewriter. Retrieval
measures 1-6 ms, so animating it would add most of a second of fake
progress to an operation that already finished.

    py infer/motion.py          # watch the three motions, if on a terminal
"""
from __future__ import annotations

import os
import sys
import threading
import time

ENABLED = False
FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
FRAME_SECONDS = 0.08


def enable(stream=None, argv=None) -> bool:
    """Turn motion on, and only where a person is actually watching.

    Four conditions, all of which must hold. PYTEST_CURRENT_TEST is in the
    list because a terminal test run is still a terminal: without it, the
    suite would sleep.
    """
    global ENABLED
    stream = stream if stream is not None else sys.stdout
    argv = sys.argv if argv is None else argv
    ENABLED = bool(
        getattr(stream, "isatty", lambda: False)()
        and not os.environ.get("NO_COLOR")
        and not os.environ.get("PYTEST_CURRENT_TEST")
        and "--no-anim" not in argv)
    return ENABLED


def pause(seconds: float) -> None:
    """Sleep, but only when motion is on. The single place time passes."""
    if ENABLED:
        time.sleep(seconds)


def wipe(lines, delay: float = 0.055) -> None:
    """Print lines in order, letting the eye follow them down the screen."""
    for line in lines:
        print(line)
        pause(delay)


def spin(label: str, work, colour=None, paint=None):
    """Run `work()` while a spinner turns on `label`'s line. Returns its value.

    `work` runs on THIS thread - it loads torch and touches CUDA, and that
    does not belong on a helper thread. The spinner is the one that gets
    moved off, it owns stdout for exactly as long as work runs, and it is
    joined before anything else prints. If that join times out the spinner is
    still writing, so the checkmark is skipped instead of interleaved with it
    - which is why there is never a second writer.
    """
    paint = paint or (lambda text, _colour: text)
    if not ENABLED:
        sys.stdout.write(paint(f"  {label}", colour))
        sys.stdout.flush()
        return work()

    stop = threading.Event()

    def turn():
        i = 0
        while not stop.is_set():
            sys.stdout.write("\r  " + paint(FRAMES[i % len(FRAMES)], colour)
                             + " " + label)
            sys.stdout.flush()
            i += 1
            stop.wait(FRAME_SECONDS)

    spinner = threading.Thread(target=turn, daemon=True)
    spinner.start()
    ok = False
    try:
        result = work()
        ok = True
        return result
    finally:
        stop.set()
        spinner.join(timeout=0.5)
        # If the join timed out the spinner still owns stdout, so nothing is
        # written rather than interleaved with a frame. Otherwise the mark
        # tells the truth about the stage: a crashed load must never leave a
        # tick behind, and a failure ends its line so the traceback that
        # follows starts on a fresh one.
        if not spinner.is_alive():
            sys.stdout.write("\r  " + paint("✓" if ok else "✗", colour)
                             + " " + label + ("" if ok else "\n"))
            sys.stdout.flush()


def count(value: int, prefix: str = "", suffix: str = "", dur: float = 0.4,
          colour=None, paint=None) -> None:
    """Roll a number up to `value`, then leave it on screen with a newline.

    This is a REVEAL, not a progress bar: it plays after the true value is
    known, so it must never be described as measuring anything.
    """
    paint = paint or (lambda text, _colour: text)
    final = f"{value:,}"
    if ENABLED:
        steps = max(int(dur / 0.045), 1)
        for i in range(1, steps):
            eased = 1 - (1 - i / steps) ** 3
            shown = f"{int(value * eased):,}".rjust(len(final))
            sys.stdout.write("\r" + prefix + paint(shown, colour) + suffix)
            sys.stdout.flush()
            time.sleep(0.045)
    lead = "\r" if ENABLED else ""
    sys.stdout.write(lead + prefix + paint(final, colour) + suffix + "\n")
    sys.stdout.flush()


def main() -> int:
    enable()
    wipe(["  wipe: one", "  wipe: two", "  wipe: three"])
    spin("spin: pretending to load", lambda: time.sleep(1.2) if ENABLED else None)
    print()
    count(202101, prefix="  count: ", suffix=" records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
