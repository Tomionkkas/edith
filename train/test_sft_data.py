"""Tests for stage-3 (instruction tuning) dataset generation.

Stage 3 teaches two things a base model cannot do: reply instead of continue,
and answer *from the Context block* rather than from its own fuzzy memory. The
dataset is generated from the curated field schema, so it costs no labelling -
but it has to be generated in exactly the shape retrieval produces at
inference, or the model learns a format it will never see again.

Run: py train/test_sft_data.py
"""
import importlib.util
import random
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "sft_data", Path(__file__).resolve().parent / "sft_data.py")
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)

RECORD = """Moon Knight
Full name: Marc Spector
Also known as: Moon Knight
First appearance: Werewolf by Night Vol 1 32
Created by: Doug Moench; Don Perlin
Identity status: Secret
Species / origin: Human
Reality: Earth-616
Occupation: Adventurer; former marine; mercenary
Base of operations: New York City
History:
Marc Spector was a mercenary who died and was revived by Khonshu."""

MCU = """Doctor Strange (Earth-199999)
Full name: Stephen Strange
First appearance: Doctor Strange (film)
Created by: Scott Derrickson; C. Robert Cargill
Reality: Earth-199999"""


class FieldParsing(unittest.TestCase):
    def test_reads_a_simple_field(self):
        self.assertEqual(sd.parse_fields(RECORD)["Full name"], "Marc Spector")

    def test_reads_the_headline_as_the_entity_name(self):
        self.assertEqual(sd.entity_name(RECORD), "Moon Knight")

    def test_headline_keeps_its_reality_suffix(self):
        self.assertEqual(sd.entity_name(MCU), "Doctor Strange (Earth-199999)")

    def test_ignores_the_history_body(self):
        self.assertNotIn("History", sd.parse_fields(RECORD))

    def test_missing_field_is_absent_not_empty(self):
        self.assertNotIn("Powers", sd.parse_fields(RECORD))


class ListHumanising(unittest.TestCase):
    def test_two_items_joined_with_and(self):
        self.assertEqual(sd.humanize("Doug Moench; Don Perlin"),
                         "Doug Moench and Don Perlin")

    def test_three_items_use_commas_then_and(self):
        self.assertEqual(sd.humanize("A; B; C"), "A, B and C")

    def test_single_item_unchanged(self):
        self.assertEqual(sd.humanize("Stan Lee"), "Stan Lee")

    def test_empty_is_empty(self):
        self.assertEqual(sd.humanize(""), "")


class QAGeneration(unittest.TestCase):
    def setUp(self):
        self.pairs = sd.qa_pairs(RECORD, seed=1)
        self.qs = [q for q, _ in self.pairs]
        self.as_ = [a for _, a in self.pairs]

    def test_generates_a_creator_question(self):
        self.assertTrue(any("creat" in q.lower() for q in self.qs))

    def test_creator_answer_carries_the_real_names(self):
        ans = " ".join(self.as_)
        self.assertIn("Doug Moench", ans)
        self.assertIn("Don Perlin", ans)

    def test_creator_answer_reads_as_a_sentence_not_a_field_dump(self):
        ans = next(a for q, a in self.pairs if "creat" in q.lower())
        self.assertIn(" and ", ans)
        self.assertNotIn(";", ans)

    def test_generates_a_first_appearance_question(self):
        self.assertTrue(any("appear" in q.lower() for q in self.qs))

    def test_every_question_mentions_the_entity(self):
        for q in self.qs:
            self.assertIn("Moon Knight", q)

    def test_no_pair_for_a_field_the_record_lacks(self):
        self.assertFalse(any("power" in q.lower() for q in self.qs))

    def test_answers_are_never_empty(self):
        self.assertTrue(all(a.strip() for a in self.as_))

    def test_question_phrasing_varies_across_records(self):
        """One template per field would teach the model a single rigid form."""
        seen = set()
        for s in range(12):
            for q, _ in sd.qa_pairs(RECORD, seed=s):
                if "creat" in q.lower():
                    seen.add(q)
        self.assertGreater(len(seen), 1)

    def test_record_with_no_usable_fields_yields_nothing(self):
        self.assertEqual(sd.qa_pairs("Some Headline\nHistory:\nprose only.", seed=0), [])


class NameForm(unittest.TestCase):
    """83% of generated questions carried a reality suffix, but nobody types
    'who created Lectron (Earth-12772)'. The model must mostly see the bare
    name, while still learning that the suffixed form disambiguates."""

    def test_strips_the_reality_suffix(self):
        self.assertEqual(sd.bare_name("Doctor Strange (Earth-199999)"),
                         "Doctor Strange")

    def test_leaves_an_unsuffixed_name_alone(self):
        self.assertEqual(sd.bare_name("Moon Knight"), "Moon Knight")

    def test_keeps_a_parenthetical_that_is_not_a_reality(self):
        self.assertEqual(sd.bare_name("Vision (android)"), "Vision (android)")

    def test_most_questions_use_the_bare_name(self):
        qs = []
        for s in range(60):
            qs += [q for q, _ in sd.qa_pairs(MCU, seed=s)]
        bare = sum(1 for q in qs if "(Earth-" not in q)
        self.assertGreater(bare / len(qs), 0.5)

    def test_suffixed_form_still_appears_sometimes(self):
        qs = []
        for s in range(60):
            qs += [q for q, _ in sd.qa_pairs(MCU, seed=s)]
        self.assertTrue(any("(Earth-199999)" in q for q in qs))


class ValueHygiene(unittest.TestCase):
    def test_drops_sub_label_artifacts(self):
        """Curation leaves 'Formerly:' sub-labels inside some field values."""
        self.assertEqual(sd.clean_value("Formerly:; Adventurer; Mechanic"),
                         "Adventurer; Mechanic")

    def test_keeps_a_normal_value_intact(self):
        self.assertEqual(sd.clean_value("Stan Lee; Steve Ditko"),
                         "Stan Lee; Steve Ditko")

    def test_rejects_a_value_that_is_only_punctuation(self):
        self.assertEqual(sd.clean_value("."), "")

    def test_no_pair_generated_for_a_junk_value(self):
        rec = "Nobody\nAbilities: .\nCreated by: Stan Lee"
        qs = [q for q, _ in sd.qa_pairs(rec, seed=0)]
        self.assertFalse(any("abilit" in q.lower() for q in qs))


class RealisticQuestions(unittest.TestCase):
    """Single-field templates assume a well-formed question. Real users ask
    open-ended and multi-part things, and sometimes about characters that are
    not in the corpus at all."""

    def test_open_ended_question_is_generated(self):
        q, a = sd.open_ended_pair(RECORD, seed=0)
        self.assertIn("Moon Knight", q)
        self.assertIn(q, [t.format(e="Moon Knight") for t in sd.OPEN_ENDED_Q])

    def test_open_ended_answer_combines_several_fields(self):
        _, a = sd.open_ended_pair(RECORD, seed=0)
        self.assertIn("Marc Spector", a)
        self.assertIn("Doug Moench", a)

    def test_open_ended_answer_is_a_paragraph_not_a_field_dump(self):
        _, a = sd.open_ended_pair(RECORD, seed=0)
        self.assertNotIn("\n", a)
        self.assertGreater(len(a), 60)

    def test_open_ended_answer_is_grammatical(self):
        """Gluing field templates after a shared 'is' produced
        'X is real name is Marc Spector'."""
        for s in range(8):
            _, a = sd.open_ended_pair(RECORD, seed=s)
            self.assertNotIn("is real name is", a)
            self.assertNotIn("is  ", a)
            self.assertNotRegex(a, r"\bis\s+\w+\s+is\b")

    def test_multi_field_second_clause_is_lowercased(self):
        """'What species is X and Where is he based?' reads as two sentences
        jammed together."""
        for s in range(8):
            r = sd.multi_field_pair(RECORD, seed=s)
            if not r:
                continue
            after = r[0].split(" and ", 1)[1]
            self.assertTrue(after[0].islower() or not after[0].isalpha(),
                            f"second clause not lowercased: {r[0]}")

    def test_multi_field_question_asks_two_things(self):
        q, a = sd.multi_field_pair(RECORD, seed=3)
        self.assertIn(" and ", q)

    def test_multi_field_answer_covers_both(self):
        q, a = sd.multi_field_pair(RECORD, seed=3)
        self.assertGreaterEqual(len(a.split(".")), 2)

    def test_where_to_read_maps_to_first_appearance(self):
        q, a = sd.where_to_read_pair(RECORD, seed=0)
        self.assertIn("Werewolf by Night Vol 1 32", a)

    def test_records_without_the_field_yield_no_pair(self):
        thin = "Nobody\nReality: Earth-616"
        self.assertIsNone(sd.where_to_read_pair(thin, seed=0))


DETAILED = """The Hood
Full name: Parker Robbins
First appearance: Hood Vol 1 1
Created by: Brian K. Vaughan; Kyle Hotz
Species / origin: Human
Reality: Earth-616
Occupation: Criminal; crime lord
Powers: Invisibility; levitation
History:
Parker Robbins was a small-time crook who stole a demonic cloak and boots
during a botched robbery. The items granted him invisibility and the ability
to walk on air, which he used to build a criminal empire in New York.
He later assembled a syndicate of super-villains during the Civil War."""


class DetailLevel(unittest.TestCase):
    """"Tell me about X in detail" must produce more than "tell me about X".
    If every generated answer is the same length the model can never vary,
    however the question is worded."""

    def test_detailed_answer_is_longer_than_the_brief_one(self):
        _, brief = sd.open_ended_pair(DETAILED, seed=0)
        _, full = sd.open_ended_pair(DETAILED, seed=0, detail=True)
        self.assertGreater(len(full), len(brief) * 1.4)

    def test_detail_question_signals_the_request(self):
        q, _ = sd.open_ended_pair(DETAILED, seed=0, detail=True)
        self.assertTrue(any(w in q.lower() for w in
                            ("detail", "everything", "full", "more about")))

    def test_detailed_answer_draws_on_the_history(self):
        _, full = sd.open_ended_pair(DETAILED, seed=0, detail=True)
        self.assertIn("cloak", full.lower())

    def test_detailed_context_contains_what_the_answer_uses(self):
        """Training an answer on facts absent from its own context teaches
        the model to invent them."""
        ctx = sd.context_for(DETAILED, detail=True)
        self.assertIn("cloak", ctx.lower())

    def test_brief_context_stays_short(self):
        self.assertLess(len(sd.context_for(DETAILED)),
                        len(sd.context_for(DETAILED, detail=True)))

    def test_field_rich_record_answers_detail_without_history(self):
        """No History is fine if the fields carry enough; what disqualifies a
        record is having almost nothing, not lacking prose specifically."""
        rich = ("Somebody\nFull name: Jane Doe\nSpecies / origin: Mutant\n"
                "Powers: Flight; telepathy\nOccupation: Adventurer\n"
                "Reality: Earth-616\nCreated by: Stan Lee")
        r = sd.open_ended_pair(rich, seed=0, detail=True)
        self.assertIsNotNone(r)
        self.assertIn("Stan Lee", r[1])


class CombinedQuestion(unittest.TestCase):
    """'Tell me about The Hood in detail and which comic did he appear in
    first' - an open request plus a specific field, in one breath."""

    def test_asks_both_parts(self):
        q, a = sd.combined_pair(DETAILED, seed=0)
        self.assertIn(" and ", q)

    def test_answer_covers_the_overview_and_the_specific_field(self):
        q, a = sd.combined_pair(DETAILED, seed=0)
        self.assertIn("Parker Robbins", a)
        self.assertIn("Hood Vol 1 1", a)

    def test_returns_none_without_a_first_appearance(self):
        self.assertIsNone(sd.combined_pair("X\nReality: Earth-616", seed=0))


POWERED = """Nightcrawler
Full name: Kurt Wagner
First appearance: Giant-Size X-Men Vol 1 1
Created by: Len Wein; Dave Cockrum
Species / origin: Mutant
Reality: Earth-616
Gender: Male
Identity status: Secret
Powers: Teleportation; prehensile tail; wall-crawling; superhuman agility; \
enhanced flexibility; night vision; camouflage in shadow; superhuman reflexes; \
expert swordsman; acrobatic mastery; limited invisibility in darkness
Occupation: Adventurer"""


class InterestingFieldsFirst(unittest.TestCase):
    """Powers is the field people actually care about, and it is the one most
    often long - 33% of Powers values exceeded the old cap and were dropped
    entirely, while `Identity status: Secret` (median 7 chars) always survived.
    """

    def test_long_powers_are_truncated_not_dropped(self):
        v = sd._usable(sd.parse_fields(POWERED), "Powers")
        self.assertIsNotNone(v)
        self.assertIn("Teleportation", v)

    def test_truncation_keeps_whole_list_items(self):
        v = sd._usable(sd.parse_fields(POWERED), "Powers", cap=60)
        self.assertFalse(v.endswith(("prehen", "superhum", ",")))

    def test_powers_appear_in_a_brief_answer(self):
        _, a = sd.open_ended_pair(POWERED, seed=0)
        self.assertIn("Teleportation", a)

    def test_powers_come_before_dull_metadata(self):
        _, a = sd.open_ended_pair(POWERED, seed=0, detail=True)
        self.assertLess(a.index("Teleportation"), a.index("Earth-616"))

    def test_short_records_still_work(self):
        thin = "Nobody\nReality: Earth-616\nCreated by: Stan Lee"
        self.assertIsNotNone(sd.open_ended_pair(thin, seed=0))


THIN = """Adanna Gui
Full name: Adanna Gui
First appearance: Daken: Dark Wolverine Vol 1 5
Created by: Daniel Way; Marjorie Liu
Species / origin: Human
Reality: Earth-616
Gender: Female"""


class ThinRecords(unittest.TestCase):
    """A record with almost nothing in it must not be used to answer
    "tell me everything", or the model learns that a detail request sometimes
    warrants two sentences."""

    def test_skips_a_tautological_real_name(self):
        """'Adanna Gui's real name is Adanna Gui.' is pure noise."""
        _, a = sd.open_ended_pair(THIN, seed=0)
        self.assertNotIn("real name is Adanna Gui", a)

    def test_keeps_a_real_name_that_differs(self):
        _, a = sd.open_ended_pair(DETAILED, seed=0)
        self.assertIn("Parker Robbins", a)

    def test_thin_record_yields_no_detail_request(self):
        self.assertIsNone(sd.open_ended_pair(THIN, seed=0, detail=True))

    def test_rich_record_still_yields_a_detail_request(self):
        self.assertIsNotNone(sd.open_ended_pair(DETAILED, seed=0, detail=True))

    def test_thin_record_still_answers_a_brief_question(self):
        r = sd.open_ended_pair(THIN, seed=0)
        self.assertIsNotNone(r)
        self.assertIn("Daniel Way", r[1])

    def test_thin_record_yields_no_combined_detail(self):
        self.assertIsNone(sd.combined_pair(THIN, seed=0, detail_only=True))


class Pronouns(unittest.TestCase):
    """Records carry a Gender field, so answers can read like prose instead of
    repeating the name in every sentence - and possessives must not become
    "what is he's real name?"."""

    def test_male_pronouns(self):
        self.assertEqual(sd.pronouns("Male"), ("he", "his", "him"))

    def test_female_pronouns(self):
        self.assertEqual(sd.pronouns("Female"), ("she", "her", "her"))

    def test_unknown_gender_falls_back_to_they(self):
        self.assertEqual(sd.pronouns(""), ("they", "their", "them"))
        self.assertEqual(sd.pronouns("Agender"), ("they", "their", "them"))

    def test_possessive_substitution_is_not_he_s(self):
        for s in range(10):
            r = sd.multi_field_pair(DETAILED, seed=s)
            if r:
                self.assertNotIn("he's real", r[0])
                self.assertNotIn("he's powers", r[0])

    def test_detailed_answer_does_not_repeat_the_name_every_sentence(self):
        _, a = sd.open_ended_pair(DETAILED, seed=0, detail=True)
        self.assertLessEqual(a.count("The Hood"), 3)


class UnknownEntity(unittest.TestCase):
    """The single most important behaviour: when the context does not contain
    the thing asked about, say so instead of inventing it. Without this the
    model confidently answers anything, which is the exact failure retrieval
    exists to prevent."""

    def test_refusal_names_what_it_could_not_find(self):
        """Naming it is better than a vague 'I don't know' - it shows the model
        understood the question and is declining on evidence, not confusion."""
        q, a = sd.unknown_pair(RECORD, "Squirrel Girl", seed=0)
        self.assertIn("Squirrel Girl", q)
        self.assertIn("Squirrel Girl", a)

    def test_refusal_admits_not_knowing(self):
        _, a = sd.unknown_pair(RECORD, "Squirrel Girl", seed=0)
        self.assertTrue(any(p in a.lower() for p in
                            ("don't have", "do not have", "couldn't find",
                             "could not find", "no information", "not in")))

    def test_refusal_does_not_leak_the_context_entity_as_the_answer(self):
        _, a = sd.unknown_pair(RECORD, "Squirrel Girl", seed=0)
        self.assertNotIn("Doug Moench", a)

    def test_refusal_wording_varies(self):
        seen = {sd.unknown_pair(RECORD, "Squirrel Girl", seed=s)[1]
                for s in range(15)}
        self.assertGreater(len(seen), 1)


class ExampleFormatting(unittest.TestCase):
    def test_has_the_three_parts_inference_will_have(self):
        ex = sd.format_example("Context:\nMoon Knight\nCreated by: Doug Moench",
                               "Who created Moon Knight?",
                               "Moon Knight was created by Doug Moench.")
        self.assertIn("Context:", ex)
        self.assertIn("User: Who created Moon Knight?", ex)
        self.assertIn("Assistant: Moon Knight was created by Doug Moench.", ex)

    def test_user_turn_follows_the_context(self):
        ex = sd.format_example("Context:\nX", "Q?", "A.")
        self.assertLess(ex.index("Context:"), ex.index("User:"))
        self.assertLess(ex.index("User:"), ex.index("Assistant:"))

    def test_works_without_context_for_chitchat(self):
        ex = sd.format_example("", "hi", "Hello! Ask me about Marvel.")
        self.assertFalse(ex.startswith("Context:"))
        self.assertTrue(ex.startswith("User: hi"))

    def test_prompt_boundary_is_recoverable_for_loss_masking(self):
        """Training must score only the assistant turn, so the split point
        has to be findable after formatting."""
        ex = sd.format_example("Context:\nX", "Q?", "A long answer.")
        prompt, answer = sd.split_prompt(ex)
        self.assertTrue(prompt.endswith(sd.ASSISTANT_TAG))
        self.assertEqual(answer.strip(), "A long answer.")


class Chitchat(unittest.TestCase):
    def test_covers_greetings(self):
        qs = [q.lower() for q, _ in sd.CHITCHAT]
        self.assertTrue(any(q.startswith("hi") or q.startswith("hello") for q in qs))

    def test_covers_who_are_you(self):
        self.assertTrue(any("who are you" in q.lower() for q, _ in sd.CHITCHAT))

    def test_answers_do_not_claim_to_be_human(self):
        for _, a in sd.CHITCHAT:
            self.assertNotIn("I am a human", a)

    def test_enough_variety_to_learn_the_turn_shape(self):
        self.assertGreaterEqual(len(sd.CHITCHAT), 20)



class SentenceTerminators(unittest.TestCase):
    """19 of 24 sampled answers contained "..".

    Field values usually carry their own full stop - "The full extent of this
    ability are unknown." - and every NARRATIVE template appends another. The
    model learned the doubled stop from the data and reproduces it faithfully,
    so it has to be fixed where the answer is assembled, not where it is shown.
    """

    def test_a_doubled_stop_collapses(self):
        self.assertEqual(sd.one_terminator("The extent is unknown.."),
                         "The extent is unknown.")

    def test_mid_answer_too(self):
        self.assertEqual(
            sd.one_terminator("damaged in battle.. Their powers include X."),
            "damaged in battle. Their powers include X.")

    def test_an_ellipsis_is_not_a_doubled_stop(self):
        """"What If...?" is a real comic title, and 4,000 records cite one."""
        for s in ("first appeared in What If...? Vol 1 1.",
                  "He paused... then left."):
            self.assertEqual(sd.one_terminator(s), s)

    def test_a_stop_appended_after_an_ellipsis_is_removed(self):
        self.assertEqual(sd.one_terminator("He trailed off...."),
                         "He trailed off...")

    def test_a_stop_appended_after_a_question_mark_is_removed(self):
        self.assertEqual(sd.one_terminator("appeared in What If...?."),
                         "appeared in What If...?")
        self.assertEqual(sd.one_terminator("Who knows?."), "Who knows?")

    def test_a_single_terminator_is_untouched(self):
        for s in ("Storm is a Mutant.", "Really!", "Who?", ""):
            self.assertEqual(sd.one_terminator(s), s)

    def test_it_is_idempotent(self):
        once = sd.one_terminator("unknown.. What If...?.")
        self.assertEqual(sd.one_terminator(once), once)

    def test_emit_applies_it(self):
        """The choke point, like agree() - a new template cannot reintroduce it."""
        import inspect
        src = inspect.getsource(sd.build)
        self.assertIn("one_terminator", src)


class PluralVerbAgreement(unittest.TestCase):
    def test_they_works_becomes_they_work(self):
        """Seen in real output: "They works as a Vigilante, Agent of the Cosmos".

        Every other NARRATIVE template uses "is", which DISAGREEMENTS already
        covered; Occupation uses "works" and slipped through.
        """
        self.assertEqual(sd.agree("They works as a Vigilante."),
                         "They work as a Vigilante.")

    def test_singular_works_is_untouched(self):
        self.assertEqual(sd.agree("He works as an Adventurer."),
                         "He works as an Adventurer.")

class FieldDebris(unittest.TestCase):
    """Artifacts seen in real rendered answers.

    These matter twice over: the same renderer generates the stage-3 dataset,
    so anything it leaves in is something the model is taught to reproduce.
    """

    def test_a_category_tag_glued_to_a_value_is_cut(self):
        self.assertEqual(sd.clean_value("KingpinCategory:Crimelords of crime"),
                         "Kingpin")

    def test_a_dangling_lead_in_is_dropped(self):
        """"...from the Kymellian, Whitey, including:" promises a list that
        the field does not contain."""
        self.assertEqual(
            sd.clean_value("Julie received her powers from Whitey, including:"),
            "Julie received her powers from Whitey")
        self.assertEqual(sd.clean_value("Powers such as:"), "Powers")

    def test_a_trailing_colon_goes(self):
        self.assertEqual(sd.clean_value("Android Physiology:"),
                         "Android Physiology")

    def test_ordinary_values_are_untouched(self):
        for v in ("Doug Moench; Don Perlin", "Weather Manipulation",
                  "Adventurer; photographer"):
            self.assertEqual(sd.clean_value(v), v)

    def test_one_enormous_item_is_cut_at_a_sentence(self):
        """Spider-Man's Abilities is a single 2,000-character essay; the item
        cap never fired because the loop only breaks once something is kept."""
        essay = ("Indomitable Will: He has a strong force of will. " * 40)
        out = sd._usable({"Abilities": essay}, "Abilities", cap=240)
        self.assertLessEqual(len(out), 260, out)
        self.assertTrue(out.endswith("."), out)


class HistoryDebris(unittest.TestCase):
    def test_section_headings_are_stripped(self):
        """"===The Hood===;" is curation debris, and the dataset taught the
        model to reproduce it."""
        rec = chr(10).join(["X", "Reality: Earth-616", "History:",
                            "===The Hood===; Parker was a small-time crook. "
                            "He stole a cloak."])
        out = sd.history_excerpt(rec)
        self.assertNotIn("=", out)
        self.assertTrue(out.startswith("Parker"), out)

    def test_ordinary_history_is_untouched(self):
        rec = chr(10).join(["X", "History:", "Marc Spector died in Egypt."])
        self.assertEqual(sd.history_excerpt(rec), "Marc Spector died in Egypt.")


class CitationsAreNotPowers(unittest.TestCase):
    """unwrap_templates() replaced a <ref> with its last argument.

    Spider-Man's powers began "Marvel Super Heroes Secret Wars Vol 1 3" and
    Storm's "House of X Vol 1 1". Only 0.1% of Powers fields corpus-wide, and
    far more among the characters people actually ask about, because a long
    page carries more citations.
    """

    def test_a_citation_item_is_dropped(self):
        out = sd._usable(
            {"Powers": "Marvel Super Heroes Secret Wars Vol 1 3; "
                       "Spider-Physiology: he sticks to walls"}, "Powers")
        self.assertNotIn("Secret Wars", out)
        self.assertIn("Spider-Physiology", out)

    def test_first_appearance_keeps_its_citation(self):
        """That field IS a citation; the filter is for Powers and Abilities."""
        out = sd._usable({"First appearance": "Amazing Fantasy Vol 1 15"},
                         "First appearance")
        self.assertEqual(out, "Amazing Fantasy Vol 1 15")

    def test_a_field_of_nothing_but_citations_is_kept(self):
        """Dropping everything would silently delete the field."""
        out = sd._usable({"Powers": "Web of Spider-Man Vol 1 1"}, "Powers")
        self.assertIn("Web of Spider-Man", out)

    def test_real_powers_are_untouched(self):
        out = sd._usable({"Powers": "Weather Manipulation; Flight"}, "Powers")
        self.assertEqual(out, "Weather Manipulation and Flight")


class PronounSubstitution(unittest.TestCase):
    """Only the SUBJECT of a later sentence becomes a pronoun.

    Replacing the name everywhere turned "the Ancient One possesses" into
    "the He possesses", and "an unnamed order whose elder members are known
    as the Ancient Ones" into "the Hes".
    """

    RECORD = "\n".join([
        "Ancient One",
        "Full name: Yao",
        "Species / origin: Human",
        "Powers: As Sorcerer Supreme the Ancient One possesses great "
        "knowledge; the elders are known as the Ancient Ones",
        "Gender: Male",
        "Created by: Stan Lee; Steve Ditko",
        "First appearance: Strange Tales Vol 1 110",
        "Reality: Earth-616",
        "History:",
        "Yao was born millennia ago.",
    ])

    def test_a_name_inside_a_field_value_survives(self):
        _, answer = sd.open_ended_pair(self.RECORD, seed=1)
        self.assertNotIn("the He ", answer)
        self.assertNotIn("the Hes", answer)
        self.assertIn("Ancient One possesses", answer)

    def test_the_subject_still_becomes_a_pronoun(self):
        """Repeating the name in every sentence is what this exists to avoid."""
        _, answer = sd.open_ended_pair(self.RECORD, seed=1)
        self.assertTrue(" He is " in answer or " His " in answer, answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class VerbAgreement(unittest.TestCase):
    """The previous dataset said "They is" 7,035 times.

    NARRATIVE templates hardcode a singular verb ("{e} is {v}."); the pronoun
    pass rewrote the subject and left the verb behind.
    """

    def test_they_is_becomes_they_are(self):
        self.assertEqual(sd.agree("They is affiliated with SHIELD."),
                         "They are affiliated with SHIELD.")

    def test_they_was_becomes_they_were(self):
        self.assertEqual(sd.agree("They was created by Stan Lee."),
                         "They were created by Stan Lee.")

    def test_they_has_becomes_they_have(self):
        self.assertEqual(sd.agree("They has powers."), "They have powers.")

    def test_lowercase_mid_sentence_is_fixed_too(self):
        self.assertIn("they are", sd.agree("Because they is strong."))

    def test_singular_subjects_are_untouched(self):
        for s in ("He is Human.", "She was created by Stan Lee.",
                  "Wolverine has a healing factor."):
            self.assertEqual(sd.agree(s), s)

    def test_multiple_errors_in_one_answer(self):
        out = sd.agree("They is Human. They has powers. They was created by X.")
        self.assertNotIn("They is", out)
        self.assertNotIn("They has", out)
        self.assertNotIn("They was", out)


class DistractorContexts(unittest.TestCase):
    """Inference sends the top 3 records, so training must show more than one.

    Training on a single record teaches that the context is always about the
    entity asked for, and the model never learns to pick.
    """

    REC = ("Lord Hood\nFull name: Parker Robbins\nPowers: magic cloak\n"
           "History:\nRobbins stole a cloak. " * 3)
    OTHERS = ["Storm\nFull name: Ororo Munroe\nPowers: weather control",
              "Venom\nFull name: Eddie Brock\nPowers: symbiote"]

    def test_all_records_appear(self):
        out = sd.context_with_distractors(self.REC, self.OTHERS,
                                          rng=random.Random(0))
        for name in ("Lord Hood", "Storm", "Venom"):
            self.assertIn(name, out)

    def test_answer_record_is_not_always_first(self):
        """Position must not stand in for reading the context."""
        firsts = set()
        for seed in range(40):
            out = sd.context_with_distractors(self.REC, self.OTHERS,
                                              rng=random.Random(seed))
            firsts.add(out.split("\n")[1])
        self.assertGreater(len(firsts), 1, "answer record always in same slot")

    def test_history_attaches_to_its_own_record(self):
        """A trailing History line would describe whichever record came last."""
        out = sd.context_with_distractors(self.REC, self.OTHERS, detail=True,
                                          rng=random.Random(3))
        block = [b for b in out.split("\n\n") if "Lord Hood" in b][0]
        self.assertIn("History:", block)

    def test_no_history_when_not_detailed(self):
        out = sd.context_with_distractors(self.REC, self.OTHERS,
                                          rng=random.Random(3))
        self.assertNotIn("History:", out)

    def test_works_with_no_decoys_available(self):
        out = sd.context_with_distractors(self.REC, [], rng=random.Random(0))
        self.assertIn("Lord Hood", out)


class IndefiniteArticle(unittest.TestCase):
    """11.79% of answers said "He is Human." and "He works as Adventurer"."""

    def test_consonant_takes_a(self):
        self.assertEqual(sd.with_article("Mutant"), "a Mutant")

    def test_vowel_takes_an(self):
        self.assertEqual(sd.with_article("Asgardian"), "an Asgardian")
        self.assertEqual(sd.with_article("Alien"), "an Alien")

    def test_multiword_uses_the_first_word(self):
        self.assertEqual(sd.with_article("Human mutate"), "a Human mutate")

    def test_plurals_keep_no_article(self):
        """"a Humans" is worse than the original."""
        self.assertEqual(sd.with_article("Humans"), "Humans")

    def test_existing_determiner_is_left_alone(self):
        for v in ("a Mutant", "an Alien", "the Watcher"):
            self.assertEqual(sd.with_article(v), v)

    def test_double_s_is_not_a_plural(self):
        self.assertEqual(sd.with_article("Princess"), "a Princess")

    def test_empty_value(self):
        self.assertEqual(sd.with_article(""), "")

    def test_narrative_template_renders_the_article(self):
        rec = ("Storm (Earth-616)\nFull name: Ororo Munroe\n"
               "Species / origin: Mutant\nGender: Female\n"
               "Occupation: Adventurer\nCreated by: Len Wein\n"
               "First appearance: Giant-Size X-Men Vol 1 1\nReality: Earth-616")
        out = sd.open_ended_pair(rec, seed=1)
        self.assertIsNotNone(out)
        self.assertNotIn(" is Mutant", out[1])
        self.assertIn("a Mutant", out[1])


class ChitchatCoverage(unittest.TestCase):
    """18,000 examples over 24 exact strings memorises strings, not categories.

    "who are you?" was answered correctly while "who are you" fell through to
    the "not in my sources" template.
    """

    def test_variants_cover_missing_punctuation(self):
        v = sd.chitchat_variants("who are you")
        self.assertIn("who are you", v)
        self.assertIn("who are you?", v)

    def test_variants_cover_capitalisation(self):
        self.assertIn("Who are you?", sd.chitchat_variants("who are you"))

    def test_variants_do_not_double_punctuate(self):
        for v in sd.chitchat_variants("how's it going?"):
            self.assertFalse(v.endswith("??"), v)

    def test_variants_are_deduplicated(self):
        v = sd.chitchat_variants("hi")
        self.assertEqual(len(v), len(set(v)))

    def test_identity_questions_are_covered(self):
        prompts = {u for u, _ in sd.CHITCHAT}
        for q in ("who are you", "what are you", "what's your name",
                  "what can you do", "are you an ai"):
            self.assertIn(q, prompts, q)

    def test_the_model_is_named_in_its_own_answers(self):
        answers = " ".join(a for _, a in sd.CHITCHAT)
        self.assertIn("EDITH", answers)

    def test_no_answer_claims_certainty(self):
        """It answers from retrieval and must not promise it cannot be wrong."""
        for user, answer in sd.CHITCHAT:
            self.assertNotIn("never wrong", answer.lower())


class QuestionCasing(unittest.TestCase):
    """"Who created Wolverine?" worked; "who created wolverine" did not.

    Questions are built from record titles, so training only ever saw the
    record's own capitalisation.
    """

    def variants(self, q, n=400):
        rng = random.Random(0)
        return {sd.question_case(q, rng) for _ in range(n)}

    def test_produces_lowercase_form(self):
        self.assertIn("who created spider-man?",
                      self.variants("Who created Spider-Man?"))

    def test_produces_form_without_question_mark(self):
        self.assertIn("who created spider-man",
                      self.variants("Who created Spider-Man?"))

    def test_keeps_the_original_form(self):
        self.assertIn("Who created Spider-Man?",
                      self.variants("Who created Spider-Man?"))

    def test_never_alters_the_words(self):
        for v in self.variants("Who created Spider-Man?"):
            self.assertIn("spider-man", v.lower())
            self.assertIn("created", v.lower())

    def test_no_trailing_space_after_stripping(self):
        for v in self.variants("Tell me about Storm ?"):
            self.assertEqual(v, v.rstrip())

    def test_answers_are_untouched(self):
        """Only the question varies; the answer keeps proper capitalisation."""
        rec = ("Storm (Earth-616)\nFull name: Ororo Munroe\n"
               "Species / origin: Mutant\nGender: Female\n"
               "Created by: Len Wein\nFirst appearance: Giant-Size X-Men Vol 1 1\n"
               "Reality: Earth-616")
        q, a = sd.open_ended_pair(rec, seed=1)
        self.assertIn("Ororo Munroe", a)
        self.assertIn("Storm", a)


class KindAwareComposition(unittest.TestCase):
    """MEASURED FAILURES 2026-09-02: "Latveria's real name is Königruch
    Latverien. They are from Earth-616." and "who is she" about a team
    answering "They are...". The composer assumed a character and nothing
    told it otherwise."""

    def location(self):
        nl = chr(10)
        return nl.join([
            "Wakanda", "Kind: location", "Page: Wakanda",
            "Formal name: Kingdom of Wakanda", "Reality: Earth-616",
            "Country: Wakanda", "Capital: Birnin Zana",
            "Created by: Stan Lee; Jack Kirby", "History:", "A nation.",
        ])

    def test_a_place_has_no_real_name(self):
        out = sd.open_ended_pair(self.location())[1]
        self.assertNotIn("real name", out.lower())

    def test_a_place_is_not_a_they(self):
        out = sd.open_ended_pair(self.location())[1]
        self.assertNotIn("they are", out.lower())

    def test_a_place_keeps_its_formal_name_as_a_fact(self):
        out = sd.open_ended_pair(self.location())[1]
        self.assertIn("Kingdom of Wakanda", out)

    def test_a_character_is_unchanged(self):
        nl = chr(10)
        rec = nl.join([
            "Wolverine", "Kind: character", "Page: James Howlett (Earth-616)",
            "Full name: James Howlett", "Gender: Male",
            "Species / origin: Mutant", "History:", "A man.",
        ])
        out = sd.open_ended_pair(rec)[1]
        self.assertIn("real name", out.lower())


class FormalNameTautology(unittest.TestCase):
    """FIX ROUND 1 2026-09-02: Task 5 renamed the name field to `Formal name`
    for items and locations, but `_tautology()` only ever special-cased
    `Full name`. Most items' Formal name equals their own title, so the
    composer said "Infinity Stones is also called Infinity Stones." -- the
    exact counterpart of the fix already made on the render side
    (`infer/render.py`'s skip condition widened from `key == "Full name"` to
    `key in ("Full name", "Formal name")`)."""

    def item(self):
        nl = chr(10)
        return nl.join([
            "Infinity Stones", "Kind: item", "Page: Infinity Stones",
            "Formal name: Infinity Stones", "Reality: Earth-616",
            "Type: Cosmic artifact", "Created by: Stan Lee; Jack Kirby",
            "History:", "Six gems of immense power.",
        ])

    def location(self):
        nl = chr(10)
        return nl.join([
            "Wakanda", "Kind: location", "Page: Wakanda",
            "Formal name: Kingdom of Wakanda", "Reality: Earth-616",
            "Country: Wakanda", "Capital: Birnin Zana",
            "Created by: Stan Lee; Jack Kirby", "History:", "A nation.",
        ])

    def test_an_item_whose_formal_name_repeats_its_title_says_nothing_extra(self):
        out = sd.open_ended_pair(self.item())[1]
        self.assertNotIn("is also called Infinity Stones", out)

    def test_a_location_whose_formal_name_differs_still_carries_it(self):
        out = sd.open_ended_pair(self.location())[1]
        self.assertIn("Kingdom of Wakanda", out)


class LocationCountryTautology(unittest.TestCase):
    """FINDING 2 2026-09-02: 453 of 10,645 location records compose a
    self-referential clause. The `location` narrative maps `Country` to
    "{e} is in {v}.", so a country whose own `Country` field repeats its
    headline composes `Abysmia is in Abysmia.` -- and for many of those it is
    the ONLY narrative sentence. `_tautology()` guarded only `Full name` and
    `Formal name`; it needs the location geography labels too."""

    def location(self, headline, country):
        nl = chr(10)
        return nl.join([
            headline, "Kind: location", f"Page: {headline}",
            f"Country: {country}", "Reality: Earth-616",
            "Created by: Stan Lee; Jack Kirby", "History:", "A place.",
        ])

    def test_a_country_repeating_the_headline_drops_the_clause(self):
        out = sd.open_ended_pair(self.location("Abysmia", "Abysmia"))[1]
        self.assertNotIn("is in Abysmia", out)

    def test_a_country_that_genuinely_differs_is_still_kept(self):
        out = sd.open_ended_pair(self.location("Birnin Zana", "Wakanda"))[1]
        self.assertIn("Wakanda", out)
