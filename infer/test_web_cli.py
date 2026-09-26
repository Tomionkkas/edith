"""`edith --web` - the flag, and that it does not disturb the terminal."""
import unittest
import importlib.util
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "terminal", ROOT / "infer/terminal.py")
T = importlib.util.module_from_spec(spec)
sys.modules["terminal"] = T
spec.loader.exec_module(T)


class StalenessNeverBlocks(unittest.TestCase):
    """A corpus that is merely OUT OF DATE must not stop a launch.

    EDITH answers perfectly well from last month's corpus, and a
    non-interactive caller - run_cases as a subprocess, --ask, --web - would
    otherwise refuse to start the day a new corpus is published. Only a
    MISSING artefact is a reason to stop.
    """

    def test_declining_a_stale_update_still_boots(self):
        import types
        term = T.Terminal.__new__(T.Terminal)
        term.ckpt = pathlib.Path("nonexistent.pt")
        fake = types.SimpleNamespace(
            # offer_bootstrap() migrates a pre-data-dir clone before it
            # decides anything is missing; the fake stands in for the
            # whole module, so it needs that too.
            migrate_legacy=lambda *_a, **_k: [],
            missing=lambda ckpt: [],
            work_pending=lambda ckpt: True,
            corpus_outdated=lambda: True,
            corpus_version=lambda: 1,
            CORPUS_VERSION=2,
            index_stale=lambda: False,
            download_mb=lambda ckpt: 296,
        )
        real, T.bootstrap = T.bootstrap, fake
        real_stdin, T.sys.stdin = T.sys.stdin, types.SimpleNamespace(
            isatty=lambda: False)
        try:
            self.assertTrue(term.offer_bootstrap())
        finally:
            T.bootstrap, T.sys.stdin = real, real_stdin

    def test_declining_a_MISSING_artefact_does_not_boot(self):
        import types
        term = T.Terminal.__new__(T.Terminal)
        term.ckpt = pathlib.Path("nonexistent.pt")
        fake = types.SimpleNamespace(
            # offer_bootstrap() migrates a pre-data-dir clone before it
            # decides anything is missing; the fake stands in for the
            # whole module, so it needs that too.
            migrate_legacy=lambda *_a, **_k: [],
            missing=lambda ckpt: [pathlib.Path("checkpoints/model.safetensors")],
            work_pending=lambda ckpt: True,
            corpus_outdated=lambda: False,
            corpus_version=lambda: 1,
            CORPUS_VERSION=1,
            index_stale=lambda: False,
            download_mb=lambda ckpt: 514,
        )
        real, T.bootstrap = T.bootstrap, fake
        real_stdin, T.sys.stdin = T.sys.stdin, types.SimpleNamespace(
            isatty=lambda: False)
        try:
            self.assertFalse(term.offer_bootstrap())
        finally:
            T.bootstrap, T.sys.stdin = real, real_stdin


class WebFlag(unittest.TestCase):
    def test_web_is_off_by_default(self):
        self.assertFalse(T.build_parser().parse_args([]).web)

    def test_web_takes_a_port(self):
        args = T.build_parser().parse_args(["--web", "--port", "9001"])
        self.assertTrue(args.web)
        self.assertEqual(args.port, 9001)

    def test_the_default_port_is_8420(self):
        self.assertEqual(T.build_parser().parse_args(["--web"]).port, 8420)

    def test_the_terminals_own_flags_still_parse(self):
        """The REPL is not replaced by this; both run from one entry point."""
        args = T.build_parser().parse_args(["--ask", "who is wolverine"])
        self.assertEqual(args.ask, "who is wolverine")
        self.assertFalse(args.web)


if __name__ == "__main__":
    unittest.main()
