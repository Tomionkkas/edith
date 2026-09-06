"""Hand-written expectations for what a RIGHT answer looks like.

Both existing harnesses ask one question: did retrieval return the right
doc_id. Every defect found by actually talking to EDITH on 2026-09-02 happened
AFTER the right record was found, so neither harness could see any of them:

  - Domino's powers came back as "telepathy" - invented by the model on top of
    a correctly retrieved record. Her power is probability manipulation.
  - Namor's occupation read "King of AtlantisCategory:Dictators".
  - Cable's first appearance was the animated series, not New Mutants 87.
  - Latveria, a country, was described as having a "real name" and being a
    "they".

These cases are the other half of the measurement. They are deliberately
hand-written rather than generated from the records: a harness built out of
the corpus can only ask whether the corpus agrees with itself, which is the
bias that made the first round-trip eval flatter the ranking it was meant to
judge.

FIELDS
  ask             one or more turns, in order. The last one is scored; the
                  earlier ones set up context, which is how a follow-up is
                  tested at all.
  page            the `Page:` line of the record that must answer - the
                  source page title, which is unique per Fandom page.
                  Identity, NOT the headline: two records are headlined
                  "Beast" and only one of them is Henry McCoy.
  must_include    substrings the answer must contain, lowercased. Facts, not
                  phrasing.
  must_not_include  claims that would be FALSE. This is the hallucination
                  guard, and the only field that catches a fluent wrong answer.
  expect          optional. "picker" means the right response is to OFFER a
                  choice rather than answer. `page` is then None, because no
                  single record is correct. Absent means an answer is
                  expected.
  why             what this case is defending. If it is not worth a sentence
                  it is not worth a case.

STATUS is not recorded here on purpose. A case says what is correct; whether
EDITH currently manages it belongs in the run, not in the expectation.
"""

CASES = [
    # ---------------------------------------------------------- the basics
    {
        "ask": ["who created taskmaster"],
        "page": "Anthony Masters (Earth-616)",
        "must_include": ["david michelinie", "george p"],
        "must_not_include": [],
        "why": "The plainest fact path: a creator credit, straight off the record.",
    },
    {
        "ask": ["when did blade first appear"],
        "page": "Eric Brooks (Earth-616)",
        "must_include": ["tomb of dracula"],
        "must_not_include": [],
        "why": "First appearance, answered in one line rather than a profile.",
    },
    {
        "ask": ["who is kitty pryde"],
        "page": "Katherine Pryde (Earth-616)",
        "must_include": ["pryde"],
        "must_not_include": [],
        "why": "Headlined by real name, so the page title and headline agree.",
    },
    {
        "ask": ["who is emma frost"],
        "page": "Emma Frost (Earth-616)",
        "must_include": ["telepath"],
        "must_not_include": [],
        "why": "A power that is genuinely hers, to balance the Domino case.",
    },

    # ------------------------------------------- the hallucination guards
    {
        "ask": ["who is domino"],
        "page": "Neena Thurman (Earth-616)",
        "must_include": ["probability"],
        "must_not_include": ["telepathy", "read the thoughts"],
        "why": "MEASURED FAILURE 2026-09-02. The model composed 'Telepathy: "
               "Neena is a mutant with the ability to read the thoughts of "
               "others' over a correctly retrieved record, and the footer "
               "said 'grounded'. Asking the same thing again as 'what are "
               "her powers' returned the correct probability manipulation "
               "from the record. Retrieval was never wrong, so no existing "
               "harness sees this.",
    },
    {
        "ask": ["who is cable"],
        "page": "Nathan Summers (Earth-616)",
        "must_include": ["new mutants"],
        "must_not_include": ["animated series"],
        "why": "MEASURED FAILURE 2026-09-02. Answered 'first appeared in "
               "X-Men: The Animated Series Season 1 7 (First Historical "
               "Appearance)'. Cable debuted in New Mutants Vol 1 87. The "
               "record carries more than one first-appearance field and the "
               "picker took the wrong one.",
    },
    {
        "ask": ["who is zarblaxian the unmaker"],
        "page": None,
        "must_include": ["don't have", "sources"],
        "must_not_include": ["created by", "first appeared"],
        "why": "Refusing beats inventing. Currently passes; it is here so it "
               "keeps passing.",
    },

    # ------------------------------------------------ who people MEAN
    {
        "ask": ["who is beast"],
        "page": "Henry McCoy (Earth-616)",
        "must_include": ["mccoy"],
        "must_not_include": ["krahllak"],
        "why": "MEASURED FAILURE 2026-09-02. Returns a 6,580-character record "
               "headlined 'Beast' whose page is `Krahllak (Earth-616)`, an "
               "obscure alien. The flagship harness scores this as a PASS "
               "because it compares the headline STRING, which is the single "
               "clearest argument for keying these cases on `page`.",
    },
    {
        "ask": ["who is power man"],
        "page": "Lucas Cage (Earth-616)",
        "must_include": ["cage"],
        "must_not_include": [],
        "why": "MEASURED FAILURE 2026-09-02. Returns `Power Man (Steele) "
               "(Earth-616)` - Erik Josten. Also scored as a flagship PASS. "
               "Erik Josten held the name first; Luke Cage is who is meant.",
    },
    {
        "ask": ["who is miles morales"],
        "page": None,
        "expect": "picker",
        "must_include": [],
        "must_not_include": ["ultimatum"],
        "why": "REWRITTEN 2026-09-02 when the picker landed. It used to "
               "assert a direct answer of `Miles Morales (Earth-1610)`, and "
               "it passed - the notability oracle put the comics Miles "
               "ahead of the Earth-616 character who is also called Miles "
               "Morales and aliased Ultimatum. But the case was written when "
               "the only outcomes were a right answer and a wrong one. "
               "Measured now, the comics Miles at 72,776 characters is only "
               "1.83x the Earth-1048 game Miles at 39,730, with Earth-1610B "
               "behind them - three substantial records for three different "
               "versions of him. Offering the choice is the right outcome, "
               "not a regression. `ultimatum` stays forbidden: the obscure "
               "Earth-616 namesake must still not be what answers.",
    },
    {
        "ask": ["tell me about jean grey"],
        "page": "Jean Grey (Earth-616)",
        "must_include": ["grey"],
        "must_not_include": [],
        "why": "Her 224 KB page is headlined `Phoenix`, so the flagship set - "
               "which wants the string 'Jean Grey' - marks the right answer "
               "wrong. Keyed on page, the right answer is simply right.",
    },
    {
        "ask": ["who is eddie brock"],
        "page": "Edward Brock (Earth-616)",
        "must_include": ["brock"],
        "must_not_include": [],
        "why": "Retrieval is correct and the card says `Sleeper Agent`, which "
               "is Fandom's current alias for him. Keyed on page this passes, "
               "which is the honest result: the record is right and the "
               "LABEL is the open question.",
    },

    # ---------------------------------------- places and items are not people
    {
        "ask": ["what is wakanda"],
        "page": "Wakanda",
        "must_include": ["wakanda"],
        "must_not_include": ["real name", "they are from"],
        "why": "MEASURED FAILURE 2026-09-02 (as Latveria and Genosha). A "
               "country was given a 'real name' and called 'they'. The "
               "record only exists at all because of the locations crawl.",
    },
    {
        "ask": ["what is latveria"],
        "page": "Latveria",
        "must_include": ["latveria"],
        "must_not_include": ["real name"],
        "why": "Same defect, and the formal name (Kingdom of Latveria) is "
               "real information that deserves a label other than 'Real name'.",
    },
    {
        "ask": ["what are the infinity stones"],
        "page": "Infinity Stones",
        "must_include": ["infinity"],
        "must_not_include": [],
        "why": "Nothing in 187,584 records owned this name before the items "
               "crawl; it fell through to a character called Stone.",
    },
    {
        "ask": ["what is adamantium"],
        "page": "Adamantium",
        "must_include": ["adamantium"],
        "must_not_include": ["earth-616; earth-1610"],
        "why": "MEASURED FAILURE 2026-09-02. The headline read 'Adamantium "
               "(Earth-616; Earth-1610; Earth-41578; Earth-TRN1400; "
               "Earth-199999)' and the prose repeated all five. An item "
               "listing many realities is not an alternate continuity.",
    },
    {
        "ask": ["what is mjolnir"],
        "page": "Mjolnir",
        "must_include": ["mjolnir"],
        "must_not_include": ["earth-616; earth-199999"],
        "why": "The same multi-reality headline defect, smaller.",
    },

    # ---------------------------------------------------------- follow-ups
    {
        "ask": ["who is namor", "what are his powers"],
        "page": "Namor McKenzie (Earth-616)",
        "must_include": ["atlantean"],
        "must_not_include": [],
        "why": "A pronoun follow-up on a settled entity. Verified working "
               "2026-09-02, and it is worth a case precisely because it was "
               "doubted.",
    },
    {
        "ask": ["who is cable", "what about domino", "what are her powers"],
        "page": "Neena Thurman (Earth-616)",
        "must_include": ["probability"],
        "must_not_include": ["cable", "telepathy"],
        "why": "Switching entity mid-conversation, then a pronoun on the NEW "
               "entity. The failure this guards is the pronoun sticking to "
               "the old subject.",
    },
    {
        "ask": ["who is namor", "who created him"],
        "page": "Namor McKenzie (Earth-616)",
        "must_include": ["everett"],
        "must_not_include": [],
        "why": "The follow-up must change which FIELD is answered, not just "
               "which entity.",
    },
    # --- the picker rule, decided by the user 2026-09-02 ---------------
    # A broad ask opens the picker. A specific ask is answered directly ONLY
    # when we are certain it is the record meant; any doubt opens the picker
    # too. So the picker is the answer to both kinds of uncertainty, and
    # guessing is never the answer to either.
    {
        "ask": ["who is spider-man", "show me the variants of spider-man"],
        "page": None,
        "expect": "picker",
        "must_include": [],
        "must_not_include": [],
        "why": "The broad ask. 908 records are called Spider-Man; today it "
               "prints the 10 largest as text, which cannot be chosen from. "
               "Nothing is ambiguous about the REQUEST - it is a request to "
               "be offered the choice.",
    },
    {
        "ask": ["who is spider-man 2099"],
        "page": "Miguel O'Hara (Earth-928)",
        "must_include": ["hara"],
        "must_not_include": [],
        "why": "The certain specific ask. One record answers to Spider-Man "
               "2099, so offering a menu would be friction, not care. This "
               "is the case that stops the picker becoming a toll booth.",
    },
    {
        "ask": ["who is spider-man", "what are the variants",
                "tell me about the ultimate one"],
        "page": "Miles Morales (Earth-1610)",
        "expect": None,
        "must_include": [],
        "must_not_include": ["ultimates", "iron lad"],
        "why": "MEASURED FAILURE 2026-09-02, and the case that motivates the "
               "whole picker: 'the ultimate one' matched the WORD ultimate "
               "and returned `Ultimates`, an Earth-6160 TEAM. "
               "REWRITTEN 2026-09-04, after 4.8. It expected a picker, and "
               "the reasoning was that nobody can know whether Ultimate "
               "Spider-Man means Peter Parker or Miles Morales. That was "
               "true when the phrase arrived with nothing on screen. It is "
               "not true now: turn 2 puts a menu up, `Spider-Man "
               "(Earth-1610)` is ONE row of it, and turn 3 points at that "
               "row - so the menu has already collapsed the ambiguity the "
               "old expectation was deferring. Answering the row the reader "
               "named is the correct behaviour, and offering a second menu "
               "over the first would be the toll booth case 22 forbids. "
               "must_not_include keeps the original guard: the Earth-6160 "
               "TEAM must still never come back.",
    },
    {
        "ask": ["who is spider-man earth-1610"],
        "page": None,
        "expect": "picker",
        "must_include": [],
        "must_not_include": [],
        "why": "A specific ask that LOOKS certain and is not: Peter Parker "
               "(Earth-1610) and Miles Morales (Earth-1610) both exist. "
               "Precision in the question is not the same as certainty in "
               "the answer, and this is the case that proves the difference.",
    },
    {
        "ask": ["who is doctor octopus", "who is she"],
        "page": "Otto Octavius (Earth-616)",
        "must_include": [],
        "must_not_include": ["she is", "her powers"],
        "why": "A pronoun that does not match the entity should not silently "
               "change the answer's gender. Related: 'who is she' after a "
               "TEAM answered 'They are...', which is the same bug wearing a "
               "different hat.",
    },

    # ------------------------------------------------------- corpus hygiene
    {
        "ask": ["who is namor"],
        "page": "Namor McKenzie (Earth-616)",
        "must_include": [],
        "must_not_include": ["category:"],
        "why": "MEASURED FAILURE 2026-09-02. Occupation read 'King of "
               "AtlantisCategory:Dictators and adventurer'. `[[Category:X]]` "
               "was being unwrapped to its label like an ordinary wikilink. "
               "Fixed in curate.py and applied on 2026-09-02; this keeps it "
               "applied, since the first fix passed its unit tests while the "
               "corpus on disk still carried 3,997 leaked lines.",
    },
    {
        "ask": ["who is squirrel girl"],
        "page": "Doreen Green (Earth-616)",
        "must_include": ["squirrel"],
        "must_not_include": ["vol 2 40;"],
        "why": "Her Powers field opens with a stray citation "
               "('Unbeatable Squirrel Girl Vol 2 40; Squirrel Powers: ...'), "
               "which is a reference that survived curation.",
    },
]
