"""Round-trip tests for the trained tokenizer settings.

sentencepiece's DEFAULT settings silently destroy a language-model corpus:
the normaliser turns newlines into spaces and collapses repeated whitespace.
Every curated Marvel record depends on line structure --

    Doctor Strange (Earth-199999)
    Also known as: Doctor Strange
    Reality: Earth-199999

-- so losing newlines would make the record format unlearnable. These tests
train a small tokenizer with the project's real settings and check that text
survives a round trip exactly.

Run: py train/test_tokenizer_roundtrip.py
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "train_tokenizer", Path(__file__).resolve().parent / "train_tokenizer.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

CORPUS = """Doctor Strange (Earth-199999)
Also known as: Doctor Strange
Reality: Earth-199999
He casts spells and it's notable.
Spider-Man was created by Stan Lee and Steve Ditko in Amazing Fantasy Vol 1 15.
The Kree-Skrull War reshaped Earth-616 and the Ultimate universe Earth-1610.
Wolverine, Galactus, Wakanda, Mjolnir, S.H.I.E.L.D. and the Infinity Gauntlet.
The Industrial Revolution began in Britain in the late eighteenth century.
It's often described as the moment growth became self-sustaining, though
historians disagree about why it happened there rather than in France.
"""


def _tiny_model():
    import sentencepiece as spm
    d = Path(tempfile.mkdtemp())
    src = d / "corpus.txt"
    src.write_text(CORPUS * 200, encoding="utf-8")
    prefix = d / "tiny"
    spm.SentencePieceTrainer.train(
        input=str(src), model_prefix=str(prefix), vocab_size=400,
        **tt.TRAINER_KWARGS)
    return spm.SentencePieceProcessor(model_file=str(prefix) + ".model")


class RoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp = _tiny_model()

    def rt(self, text):
        return self.sp.decode(self.sp.encode(text))

    def test_newlines_survive(self):
        text = "Doctor Strange\nReality: Earth-616\nHe casts spells."
        self.assertEqual(self.rt(text), text)

    def test_blank_line_between_records_survives(self):
        text = "Spider-Man\nReality: Earth-616\n\nIron Man\nReality: Earth-616"
        self.assertEqual(self.rt(text), text)

    def test_repeated_spaces_survive(self):
        text = "two  spaces and three   spaces"
        self.assertEqual(self.rt(text), text)

    def test_no_leading_space_is_invented(self):
        """add_dummy_prefix would prepend a space to every encoded string."""
        self.assertEqual(self.rt("Wolverine"), "Wolverine")

    def test_apostrophes_survive(self):
        text = "It's Stan Lee's creation, DC's rival."
        self.assertEqual(self.rt(text), text)

    def test_unicode_survives_via_byte_fallback(self):
        text = "Ororo Munroe — Storm — Ξ 東京 🕷"
        self.assertEqual(self.rt(text), text)

    def test_full_record_survives_exactly(self):
        text = ("Doctor Strange (Earth-199999)\n"
                "Also known as: Doctor Strange\n"
                "First appearance: Doctor Strange (film)\n"
                "Reality: Earth-199999\n")
        self.assertEqual(self.rt(text), text)

    def test_eos_token_exists(self):
        self.assertGreaterEqual(self.sp.piece_to_id("<|endoftext|>"), 0)


class EntitySymbols(unittest.TestCase):
    """High-frequency entities that BPE cannot merge get forced into the vocab.

    Dotted acronyms are the worst case: BPE learns merges from frequency, and a
    period between every letter means the merges never form. `S.H.I.E.L.D.`
    costs 12 tokens in GPT-2 and in an unaided BPE of ours, despite appearing
    3,771 times in the corpus.
    """

    @classmethod
    def setUpClass(cls):
        cls.sp = _tiny_model()

    def test_dotted_acronym_is_a_single_token(self):
        self.assertEqual(len(self.sp.encode("S.H.I.E.L.D.")), 1)

    def test_structural_field_label_is_a_single_token(self):
        self.assertEqual(len(self.sp.encode("First appearance")), 1)

    def test_main_continuity_id_is_a_single_token(self):
        self.assertEqual(len(self.sp.encode("Earth-616")), 1)

    def test_entity_symbols_still_round_trip(self):
        text = "S.H.I.E.L.D. agents on Earth-616.\nFirst appearance: unknown"
        self.assertEqual(self.sp.decode(self.sp.encode(text)), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
