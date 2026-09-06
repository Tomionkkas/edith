"""End-to-end assertions against the REAL index.

Every serious bug in this project lived between components that each worked
in isolation, and none was catchable by a unit test:

  * the context builder and retrieval disagreed by 33x - 11,200 tokens against
    a 1,024-token window
  * "hi" retrieved Hi-Lite, a real character, and the model looped forever
  * promote_flagship preferred a fieldless Wikipedia page for "doctor doom"
  * plural stripping turned "does" into "doe" and matched "Doe Eyes"
  * a bare powers pattern matched "power" in Power Man, so "who is power man"
    answered with Iron Man's power list

Each was found by hand, by re-running the same questions. These make that
`pytest`. They need the built index and names (see paths.py) and are skipped
without them; generation is NOT exercised here, because every failure above
was upstream of the model.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_paths_spec = importlib.util.spec_from_file_location(
    "edith_paths", ROOT / "paths.py")
paths = importlib.util.module_from_spec(_paths_spec)
_paths_spec.loader.exec_module(paths)

HAVE_DATA = paths.INDEX.exists() and paths.NAMES.exists()


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@unittest.skipUnless(HAVE_DATA, "needs the built index and names")
class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.search = _load("search", "retrieve/search.py")
        cls.resolve = _load("resolve", "retrieve/resolve.py")
        cls.disambiguate = _load("disambiguate", "retrieve/disambiguate.py")
        cls.sft = _load("sft_data", "train/sft_data.py")
        cls.facts = _load("facts", "infer/facts.py")
        cls.engine = _load("engine", "infer/engine.py")
        cls.index = cls.search.Index.load()
        cls.names = cls.resolve.load()

    def ask(self, question):
        return self.engine.try_facts(question, self.index, self.sft, self.search,
                                     self.disambiguate, self.facts, self.resolve)

    # ------------------------------------------------------------ creators

    def test_creators(self):
        for q, who in [
            ("who created moon knight", ["Doug Moench", "Don Perlin"]),
            ("who created spider-man", ["Stan Lee", "Steve Ditko"]),
            ("who created wolverine", ["Len Wein"]),
            ("who created deadpool", ["Fabian Nicieza", "Rob Liefeld"]),
            ("who created daredevil", ["Stan Lee", "Bill Everett"]),
            ("who created black panther", ["Stan Lee", "Jack Kirby"]),
        ]:
            a = self.ask(q) or ""
            for name in who:
                self.assertIn(name, a, f"{q!r} -> {a!r}")

    def test_doctor_doom_is_in_the_corpus(self):
        """Victor von Doom (Earth-616) was never crawled; a Gwenpool cameo
        answered instead. Re-fetched by crawl/fetch_pages.py."""
        a = self.ask("who created doctor doom") or ""
        self.assertIn("Stan Lee", a)
        self.assertIn("Jack Kirby", a)

    def test_thor_resolves_through_his_real_name(self):
        """Thor is headlined "Thor Odinson (Earth-616)", not "Thor", so
        headline matching alone reached a 4 KB variant."""
        a = self.ask("who created thor") or ""
        self.assertIn("Stan Lee", a)
        self.assertNotIn("Liefeld", a)

    def test_the_hood_reaches_parker_robbins(self):
        """Headlined "Lord Hood"; an article in the query blocked the match."""
        a = self.ask("who created the hood") or ""
        self.assertIn("Vaughan", a)

    # -------------------------------------------------------- other fields

    def test_first_appearances(self):
        for q, comic in [
            ("when did daredevil first appear", "Daredevil Vol 1 1"),
            ("when did doctor doom first appear", "Fantastic Four Vol 1 5"),
            ("when did spider-man first appear", "Amazing Fantasy Vol 1 15"),
        ]:
            self.assertIn(comic, self.ask(q) or "", q)

    def test_real_names(self):
        for q, name in [
            ("what is wolverines real name", "James Howlett"),
            ("what is daredevils real name", "Murdock"),
            ("what is doctor dooms real name", "von Doom"),
        ]:
            self.assertIn(name, self.ask(q) or "", q)

    def test_reality(self):
        self.assertIn("Earth-616", self.ask("where is black panther from") or "")

    def test_venom_resolves_to_the_symbiote_not_a_host(self):
        """Mary Jane's Venom record has more fields than the symbiote's."""
        a = self.ask("what powers does venom have") or ""
        self.assertIn("Symbiote", a)
        self.assertNotIn("Mary Jane", a)

    # ----------------------------------------------------------- routing

    def test_chitchat_never_reaches_the_fact_path(self):
        """Retrieval matches "hi" to Hi-Lite with score 60."""
        for q in ("hi", "hello", "who are you", "thanks", "how are you"):
            self.assertIsNone(self.ask(q), q)

    def test_a_profile_question_is_composed_from_the_record(self):
        """These used to go to the model, and its prose could contradict the
        sources box printed directly above it: asked who the Hood is, it
        credited Paul Jenkins and Humberto Ramos while the record said Brian
        K. Vaughan and Kyle Hotz. "Who is X" names no single field, but the
        record answers it in full, so it is composed rather than generated."""
        a = self.ask("tell me about daredevil in detail")
        self.assertIsNotNone(a)
        self.assertIn("Stan Lee", a)
        self.assertIn("Bill Everett", a)
        b = self.ask("give me a rundown on storm")
        self.assertIn("Ororo Munroe", b)

    def test_a_genuinely_open_question_is_still_the_model(self):
        """The line is "who is X", not "anything with a record attached"."""
        for q in ("what happened in civil war",
                  "why did tony stark build the iron man armor"):
            self.assertIsNone(self.ask(q), q)

    def test_power_man_is_not_a_powers_question(self):
        """A bare powers pattern stripped "power" and answered with Iron Man."""
        a = self.ask("who is power man")
        self.assertTrue(a is None or "Iron Man" not in a, a)

    def test_unknown_character_is_refused_not_invented(self):
        a = self.ask("who created Zyxthaloraxian the Devourer")
        self.assertTrue(a is None or "don't have" in a, a)

    # ------------------------------------------------------- the context

    def test_context_fits_the_model_window(self):
        """It once reached 11,200 tokens against a 1,024-token window."""
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(
            model_file=str(ROOT / "tokenizer" / "marvel_bpe_50257.model"))
        for q in ("who created moon knight", "tell me about spider-man in detail",
                  "what powers does venom have", "who is the hood"):
            prompt = self.engine.build_prompt(q, self.index, self.sft, self.search,
                                              self.disambiguate, self.facts,
                                              self.resolve, k=3)
            self.assertLess(len(sp.encode(prompt)), 900, q)

    def test_chitchat_prompt_carries_no_context(self):
        prompt = self.engine.build_prompt("hi", self.index, self.sft, self.search,
                                          self.disambiguate, self.facts,
                                          self.resolve, k=3)
        self.assertNotIn("Context:", prompt)

    def test_prompt_ends_where_training_masked_it(self):
        """A prompt differing by one separator makes a good model look broken."""
        prompt = self.engine.build_prompt("who is storm", self.index, self.sft,
                                          self.search, self.disambiguate,
                                          self.facts, self.resolve, k=3)
        self.assertTrue(prompt.endswith(self.sft.ASSISTANT_TAG), repr(prompt[-40:]))


class ScaffoldingInTheQuery(Pipeline):
    """Words that are not part of any name must not steer resolution.

    Reported from the terminal: "who is the Hood, explain in detail his
    powers" answered from a 1,986-char Golden Age stub. The words "explain",
    "detail" and "his" stayed in the key, so Parker Robbins' two-word primary
    name ("lord", "hood") could no longer be a subset of it - he matched only
    through his ALIAS, and `primary` outranks `size`, so the stub that owns
    the bare name won. Four ranking orders were measured; the one in use is
    the best of them, so the fix belongs in the key, not the ranking.
    """

    def head(self, question):
        doc = self.engine.resolved_doc(question, self.index, self.facts,
                                       self.resolve)
        return self.index.headlines[doc] if doc is not None else None

    def test_the_reported_question(self):
        self.assertEqual(
            self.head("who is the Hood, explain in detail his powers"),
            "Lord Hood")

    def test_detail_does_not_change_who_is_meant(self):
        for q in ("thor", "tell me about thor", "tell me about thor in detail",
                  "explain thor in detail", "describe thor"):
            self.assertEqual(self.head(q), "Thor Odinson (Earth-616)", q)

    def test_pronouns_do_not_resolve_to_the_characters_named_after_them(self):
        """"Him" is Adam Warlock and "he" is the High Evolutionary."""
        for q in ("what are his powers", "explain her powers in detail"):
            self.assertNotIn(self.head(q) or "", ("Adam Warlock (Earth-616)",
                                                  "High Evolutionary"))

    def test_a_query_that_is_only_scaffolding_names_nobody(self):
        """resolve() falls back to the unstripped query when stripping empties
        it, and that fallback matches the scaffolding itself: "more details
        about his powers" found a character called More. With no previous
        record to carry, the honest answer is that we do not know who is
        meant. It costs "who is him" reaching Adam Warlock, whose alias is
        "Him" - and guessing was worse."""
        self.assertIsNone(self.head("who is him"))
        self.assertIsNone(self.head("more details about his powers"))

    def test_a_pronoun_powers_question_finds_the_field(self):
        self.assertEqual(self.facts.detect_intent("explain in detail his powers")[0],
                         "powers")
        self.assertEqual(self.facts.detect_intent("what are her powers")[0],
                         "powers")

    def test_power_man_is_still_not_a_powers_question(self):
        """The pattern stays tight enough that "who is power man" is not a
        request for a Powers field.

        The second assertion is on the RECORD, not the headline. It used to
        read `== "Power Man"`, which is the headline of
        `Power Man (Steele) (Earth-616)` - Erik Josten, the wrong man, and a
        defect CLAUDE.md carried under Known issues until 4.10 fixed it. A
        headline string cannot tell the two apart; the Page: title can, which
        is why 4.9 re-keyed run_cases the same way."""
        self.assertIsNone(self.facts.detect_intent("who is power man"))
        doc = self.engine.resolved_doc("who is power man", self.index,
                                       self.facts, self.resolve)
        self.assertEqual(self.resolve.page_title(self.index.text(doc)),
                         "Lucas Cage (Earth-616)")


class FollowUps(Pipeline):
    """A question that names nobody is about whoever was just named.

    The model saw only single-turn examples, so it has no dialogue memory and
    cannot be given one without retraining. What CAN carry over is the
    retrieved record - which is where the facts live anyway. Without it "more
    details about his powers" kept the leftover word "powers" and resolved to
    a character called The Power.
    """

    def head(self, question, previous=None):
        doc = self.engine.resolved_doc(question, self.index, self.facts,
                                       self.resolve, previous)
        return self.index.headlines[doc] if doc is not None else None

    def setUp(self):
        self.hood = self.engine.resolved_doc("the hood", self.index,
                                             self.facts, self.resolve)

    def test_a_follow_up_stays_on_the_same_character(self):
        for q in ("more details about his powers", "what about his real name",
                  "who created him", "and his first appearance?",
                  "tell me more"):
            self.assertEqual(self.head(q, self.hood), "Lord Hood", q)

    def test_a_follow_up_that_names_someone_else_switches(self):
        self.assertEqual(self.head("what about storm", self.hood), "Storm")

    def test_an_unknown_name_is_still_refused_mid_conversation(self):
        """Carrying the entity must not swallow a name we do not have."""
        self.assertIsNone(self.head("who is blorptron the unmaker", self.hood))

    def test_the_first_question_of_a_session_has_nothing_to_carry(self):
        self.assertIsNone(self.head("more details about his powers"))

    def test_a_named_question_ignores_the_previous_record(self):
        self.assertEqual(self.head("who created moon knight", self.hood),
                         "Moon Knight")

    def test_a_demonstrative_follow_up_stays_on_the_same_character(self):
        """"who is this" is exactly what the terminal asks right after a
        picker selection (infer/terminal.py's offer()/select()) - it must
        fall back to `previous` the way "who is he" already does above, or
        picking a record from a list answers about the wrong one."""
        self.assertEqual(self.head("who is this", self.hood), "Lord Hood")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_DATA, "needs the built index and names")
class UnknownEntities(Pipeline):
    """Refusing must not become refusing everything.

    The first guard judged leftover query words against NAME vocabulary, so
    "give me a rundown on storm" was refused because no character is called
    "Rundown". The corpus vocabulary is the right test: a word the corpus has
    never seen at all is the signal.
    """

    def test_invented_names_are_refused(self):
        for q in ("who created Blorptron the Unmaker",
                  "who created Zyxthaloraxian the Devourer",
                  "what powers does Grelmaxor have"):
            a = self.ask(q) or ""
            self.assertIn("don't have", a, q)

    def test_the_refusal_names_only_the_entity(self):
        """It once echoed the whole question back."""
        a = self.ask("who created Blorptron the Unmaker") or ""
        self.assertNotIn("who created", a)
        self.assertIn("Blorptron", a)

    def test_unusual_phrasing_about_a_real_character_is_not_refused(self):
        for q in ("give me a rundown on storm",
                  "tell me everything about wolverine",
                  "i want to know more about venom"):
            a = self.ask(q)
            self.assertTrue(a is None or "don't have" not in a, f"{q} -> {a}")

    def test_real_but_obscure_words_do_not_trigger_refusal(self):
        """"unmaker" and "devourer" are real Marvel words; the invented part
        of the name is what must be caught."""
        import importlib.util
        for word in ("unmaker", "devourer", "rundown"):
            self.assertIn(word, self.index.postings, word)


@unittest.skipUnless(HAVE_DATA, "needs the built index and names")
class ConfidenceReadsTheCorpusVocabulary(Pipeline):
    """FINDING 2026-09-02: `resolve.confidence()` and `resolve.rivals()`
    judged their tier-1 unknown-word guard against the NAME vocabulary, while
    `resolve.resolve()` - the function that actually decides which record
    answers, called from `infer/engine.py:145` with `index.postings` - judges
    the same guard against the CORPUS vocabulary (resolve.py:509 says so
    explicitly). A query word that names nobody but IS a real corpus word
    wrongly evicted every tier-1 candidate for the contested name from
    confidence()'s count, which can only push the ratio UP - silently
    suppressing the picker.

    Measured directly against the real index: "symbiotic" is in the corpus
    vocabulary (index.postings) but in no character's name.
    """

    def test_symbiotic_is_a_corpus_word_in_no_name(self):
        self.assertIn("symbiotic", self.index.postings)
        self.assertNotIn("symbiotic", self.resolve.vocabulary(self.names))

    def test_a_query_naming_a_real_corpus_word_gets_the_same_offer_decision(self):
        """"who is venom symbiotic" and "venom" name the same contested
        record (two same-named Venom records close in size - confidence
        1.54). Adding a real corpus word that happens to be in no NAME must
        not change whether EDITH offers a picker or answers outright -
        exactly the parity the fix restores."""
        doc = self.engine.resolved_doc("who is venom symbiotic", self.index,
                                       self.facts, self.resolve)
        self.assertIsNotNone(doc)

        def wants(question):
            d = self.engine.resolved_doc(question, self.index, self.facts,
                                         self.resolve)
            return self.engine.wants_choice(question, self.index, self.facts,
                                            self.resolve, d)

        self.assertEqual(wants("who is venom symbiotic"), wants("venom"))
