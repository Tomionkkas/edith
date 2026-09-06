#!/usr/bin/env python3
"""Generate the stage-3 (instruction tuning) dataset.

Stage 3 teaches a base model two things it cannot do:

  1. **Reply, not continue.** A base model given "hi how are you" writes more
     text that looks like a document containing that phrase.
  2. **Answer from the Context block**, not from its own fuzzy memory. A 250M
     model cannot hold 261 MB of facts, so the facts arrive by retrieval and
     the model's job is to read and phrase them.

The examples are generated from the curated field schema, so they cost no
labelling -- and they are generated in exactly the shape `retrieve/search.py`
produces at inference, because a format the model never sees again is worse
than useless.

  py train/sft_data.py --build          # -> data/sft/stage3.jsonl
  py train/sft_data.py --preview        # show a few examples
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
import random
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CURATED = BASE_DIR / "curated"
OUT_DIR = BASE_DIR / "data" / "sft"
OUT_FILE = OUT_DIR / "stage3.jsonl"
SEPARATOR = "=" * 60

# Always kept when present: these are the questions people actually ask.
PRIORITY_FIELDS = ("Created by", "First appearance")

ASSISTANT_TAG = "Assistant:"
USER_TAG = "User:"

# Several phrasings per field: one template would teach a single rigid form,
# and the model would fail on any question worded differently.
TEMPLATES = {
    "Created by": (
        ["Who created {e}?", "Who made {e}?", "Which creators are behind {e}?",
         "Who is {e} created by?", "Tell me who created {e}."],
        ["{e} was created by {v}.", "{e} was created by {v}.",
         "{v} created {e}."],
    ),
    "First appearance": (
        ["Where did {e} first appear?", "What was {e}'s first appearance?",
         "In which comic did {e} debut?", "When did {e} first show up?"],
        ["{e} first appeared in {v}.", "{e} debuted in {v}."],
    ),
    "Reality": (
        ["Which universe is {e} from?", "What reality does {e} belong to?",
         "Which continuity is {e} in?"],
        ["{e} is from {v}.", "{e} belongs to {v}."],
    ),
    "Full name": (
        ["What is {e}'s real name?", "Who is {e} really?",
         "What is the full name of {e}?"],
        ["{e}'s real name is {v}.", "{e} is {v}."],
    ),
    "Powers": (
        ["What are {e}'s powers?", "What can {e} do?",
         "Which abilities does {e} have?"],
        ["{e} has {v}.", "{e}'s powers include {v}."],
    ),
    "Abilities": (
        ["What abilities does {e} have?", "What is {e} skilled at?"],
        ["{e} has {v}.", "{e} is skilled in {v}."],
    ),
    "Affiliation": (
        ["Who is {e} affiliated with?", "Which teams has {e} been part of?",
         "What groups does {e} belong to?"],
        ["{e} is affiliated with {v}.", "{e} has been part of {v}."],
    ),
    "Occupation": (
        ["What does {e} do for a living?", "What is {e}'s occupation?"],
        ["{e} works as {a}.", "{e}'s occupation is {v}."],
    ),
    "Species / origin": (
        ["What species is {e}?", "What is {e}'s origin?"],
        ["{e} is {a}.", "{e}'s origin is {v}."],
    ),
    "Base of operations": (
        ["Where is {e} based?", "Where does {e} operate from?"],
        ["{e} is based in {v}.", "{e} operates from {v}."],
    ),
    "Identity status": (
        ["Is {e}'s identity secret?", "What is {e}'s identity status?"],
        ["{e}'s identity is {v}.", "{e} has a {v} identity."],
    ),
}

# Hand-written, so the model is not a Marvel-only vending machine. Kept honest:
# it does not claim to be human and it does not overpromise its own accuracy.
CHITCHAT = [
    ("hi", "Hello! Ask me anything about Marvel comics."),
    ("hello", "Hi there. What would you like to know about the Marvel universe?"),
    ("hey", "Hey. Ask me about any Marvel character, comic or creator."),
    ("yo", "Hey. Who do you want to know about?"),
    ("hey there", "Hey. Ask me about anyone in the Marvel universe."),
    ("good morning", "Good morning. What can I look up for you?"),
    ("good evening", "Good evening. Who would you like to know about?"),
    ("how are you", "I'm running fine, thanks. What can I look up for you?"),
    ("how's it going", "All good here. Ask me about any Marvel character."),
    ("how are you doing", "Fine, thanks. What would you like to know?"),
    ("who are you", "I'm EDITH, a small language model trained on Marvel comics. "
                    "I look answers up in my sources rather than recalling them."),
    ("what are you", "A small language model with a Marvel comics index. "
                     "Ask me about a character and I'll look them up."),
    ("what is your name", "EDITH. Ask me about anyone in the Marvel universe."),
    ("what's your name", "EDITH. Who would you like to know about?"),
    ("introduce yourself", "I'm EDITH. I answer questions about Marvel comics "
                           "from a search index, so I can quote sources rather "
                           "than guess."),
    ("tell me about yourself", "I'm EDITH, a Marvel comics assistant. I look "
                               "things up in my index instead of relying on memory."),
    ("what do you do", "I answer questions about Marvel characters, comics and "
                       "creators, using a search index."),
    ("what can you do", "I can tell you about Marvel characters, who created "
                        "them, where they first appeared and what powers they have."),
    ("what do you know", "Marvel comics, mostly. Characters, creators, first "
                         "appearances, powers and continuities."),
    ("are you human", "No, I'm a language model."),
    ("are you an ai", "Yes. A small one, trained on Marvel comics."),
    ("are you real", "I'm a language model, not a person."),
    ("are you chatgpt", "No. I'm EDITH, a much smaller model that only knows Marvel."),
    ("do you know everything about marvel", "Not everything. I look things up, "
                                            "and I'll say so when I can't find them."),
    ("can you be wrong", "Yes. I answer from retrieved sources, so ask me to "
                         "point at the comic if it matters."),
    ("thanks", "You're welcome."),
    ("thank you", "Any time. Ask me anything else about Marvel."),
    ("cheers", "You're welcome."),
    ("bye", "See you around."),
    ("goodbye", "Goodbye."),
    ("see you", "See you."),
    ("ok", "Anything else you'd like to know?"),
    ("cool", "Ask me about another character whenever you like."),
    ("nice", "Anything else?"),
    ("sorry", "No problem. What would you like to know?"),
    ("never mind", "No problem. Ask me something else when you're ready."),
    ("help", "Ask me about a Marvel character, a comic, or a creator. "
             "For example: who created Moon Knight?"),
    ("what should I ask", "Try a character you like: who created them, where "
                          "they first appeared, or what powers they have."),
    ("tell me a fact", "Ask me about any Marvel character and I'll pull what my "
                       "sources have on them."),
]


def chitchat_variants(user: str):
    """Case and punctuation variants of one prompt.

    The previous dataset repeated 24 exact strings ~750 times each, so the
    model memorised the strings: "who are you?" was answered correctly while
    "who are you" fell through to the "not in my sources" template. Variants
    teach the category instead.
    """
    base = user.strip()
    out = {base, base.lower(), base.capitalize()}
    if not base.endswith(("?", "!", ".")):
        q = base + "?"
        out |= {q, q.lower(), q.capitalize()}
    return sorted(out)


def entity_name(record: str) -> str:
    return record.strip().split("\n", 1)[0].strip()


def parse_fields(record: str) -> dict:
    """`Label: value` lines above the body. Stops at History/Synopsis."""
    fields = {}
    for line in record.split("\n")[1:]:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(("History:", "Synopsis", "Notes:", "Trivia:")):
            break
        if ": " in line:
            label, _, value = line.partition(": ")
            label, value = label.strip(), value.strip()
            if label and value and len(label) < 30:
                fields.setdefault(label, value)
    return fields


REALITY_SUFFIX_RE = re.compile(r"\s*\(Earth-[0-9A-Za-z\-]+\)\s*$")
SUB_LABELS = ("formerly:", "currently:", "originally:", "later:", "previously:")


def bare_name(entity: str) -> str:
    """`Doctor Strange (Earth-199999)` -> `Doctor Strange`.

    Only a reality suffix is stripped; `Vision (android)` keeps its
    parenthetical because that is part of the name, not a continuity tag.
    """
    return REALITY_SUFFIX_RE.sub("", entity).strip()


CATEGORY_TAG = re.compile(r"Category:")
DANGLING_LEAD_IN = re.compile(
    r"[,;]?\s*(?:including|such as|namely|these include)\s*:?\s*$", re.I)


def clean_value(value: str) -> str:
    """Drop sub-label artifacts and values that are only punctuation.

    Curation leaves `Formerly:` markers inside some field values, which turn
    into answers like "affiliated with Psigns of the Times, Formerly: and ...".
    """
    parts = []
    for p in (x.strip() for x in value.split(";")):
        # "KingpinCategory:Crimelords of crime" - a wiki category glued to the
        # end of the value by curation. Everything from the tag on is markup.
        p = CATEGORY_TAG.split(p)[0].strip()
        # "...from the Kymellian, Whitey, including:" promises a list the
        # field does not contain, so the promise comes off.
        # SUB_LABELS carry their colon ("Formerly:"), so the check comes
        # before the colon is stripped - and after, for the bare form.
        if p.lower() in SUB_LABELS:
            continue
        p = DANGLING_LEAD_IN.sub("", p).rstrip(": ").strip()
        if not p or p.lower() in SUB_LABELS or f"{p.lower()}:" in SUB_LABELS:
            continue
        if not any(c.isalnum() for c in p):
            continue
        parts.append(p)
    return "; ".join(parts)


def humanize(value: str) -> str:
    """`A; B; C` -> `A, B and C`, so answers read as prose not field dumps."""
    parts = [p.strip() for p in value.split(";") if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def qa_pairs(record: str, seed: int = 0, max_pairs: int = 4) -> list:
    """Question/answer pairs from whichever fields the record actually has.

    A record can match eight templates but only `max_pairs` survive, so the
    two questions people actually ask -- who created this, and where did it
    first appear -- are kept whatever else is dropped.
    """
    rng = random.Random(f"{entity_name(record)}|{seed}")
    full = entity_name(record)
    bare = bare_name(full)
    fields = parse_fields(record)
    priority, rest = [], []
    for label, (questions, answers) in TEMPLATES.items():
        raw = fields.get(label)
        if not raw:
            continue
        value = humanize(clean_value(raw))
        if not value or len(value) > 220:
            continue
        # Ask mostly by the bare name, since that is what a person types, but
        # keep the suffixed form in the mix so disambiguation is still learned.
        asked = full if (full != bare and rng.random() < 0.25) else bare
        # "What is X's real name?" -> "X's real name is X." teaches nothing
        if _tautology(label, value, bare) or _tautology(label, value, full):
            continue
        pair = (rng.choice(questions).format(e=asked),
                rng.choice(answers).format(e=asked, v=value,
                                                 a=with_article(value)))
        (priority if label in PRIORITY_FIELDS else rest).append(pair)
    rng.shuffle(rest)
    return (priority + rest)[:max_pairs]


# --------------------------------------------------- realistic question forms

OPEN_ENDED_Q = [
    "Tell me about {e}.", "Who is {e}?", "What do you know about {e}?",
    "Can you tell me about {e}?", "Give me a rundown on {e}.",
    "What's the deal with {e}?", "I'm interested in {e}.",
]

DETAIL_Q = [
    "Tell me about {e} in detail.", "Tell me everything about {e}.",
    "Give me a full rundown on {e}.", "I want to know more about {e}.",
    "Explain {e} in detail.", "Who is {e}? Give me the details.",
]

WHERE_TO_READ_Q = [
    "Where can I read about {e}?", "Which comic should I buy to read {e}?",
    "Where do I start with {e}?", "Which issue introduces {e}?",
    "What comic is {e} in?",
]

# Wording varies so the model learns the behaviour, not one magic sentence.
UNKNOWN_A = [
    "I don't have any information about {e} in my sources.",
    "I couldn't find {e} in the reference material I have.",
    "There's no information about {e} in what I can see.",
    "I don't have {e} in my sources, so I can't say.",
    "{e} isn't in the material I have access to.",
]

# Complete sentences, not fragments sharing one verb: gluing fragments after a
# single "is" produced "The Hood is real name is Parker Robbins".
# Ordered by how much a reader actually cares. Only the first few survive a
# brief answer, so powers must outrank "Identity status: Secret".
# "an" before a vowel sound. Marvel species are ordinary nouns - Asgardian,
# Alien, Inhuman, Eternal - so the letter test is enough here.
VOWEL_LETTERS = "aeiou"
ALREADY_DETERMINED = ("a ", "an ", "the ", "one ", "two ", "several ")


def with_article(value: str) -> str:
    """'Mutant' -> 'a Mutant'; 'Asgardian' -> 'an Asgardian'.

    The templates read "{e} is {v}." and "{e} works as {v}.", which produced
    "He is Human." and "He works as Adventurer" in 11.8% of answers. A plural
    keeps no article, because "a Humans" is worse than the original.
    """
    if not value:
        return value
    low = value.lower()
    if low.startswith(ALREADY_DETERMINED):
        return value
    first = low.split()[0] if low.split() else ""
    if not first:
        return value
    if first.endswith("s") and not first.endswith(("ss", "us", "is")):
        return value
    return ("an " if first[0] in VOWEL_LETTERS else "a ") + value


NARRATIVE = {
    "Full name": "{e}'s real name is {v}.",
    "Species / origin": "{e} is {a}.",
    "Powers": "{e}'s powers include {v}.",
    "Abilities": "{e} is skilled in {v}.",
    "Occupation": "{e} works as {a}.",
    "Affiliation": "{e} is affiliated with {v}.",
    "Reality": "{e} is from {v}.",
    "Base of operations": "{e} is based in {v}.",
}


def kind_of(record: str) -> str:
    """Read the curation-written Kind line. Duplicated rather than imported:
    train/ must not depend on retrieve/."""
    for line in record.split(chr(10))[1:]:
        if line.startswith(("History:", "Synopsis")):
            break
        if line.startswith("Kind: "):
            return line[6:].strip() or "unknown"
    return "unknown"


# Only the clauses that differ. Anything absent falls back to NARRATIVE, so a
# kind nobody has thought about yet reads exactly as it does today.
NARRATIVE_BY_KIND = {
    "location": {
        "Formal name": "{e}'s formal name is {v}.",
        "Country": "{e} is in {v}.",
        "Continent": "{e} is on {v}.",
        "Capital": "Its capital is {v}.",
        "Population": "Its population is {v}.",
        "Reality": "{e} is in {v}.",
    },
    "item": {
        "Formal name": "{e} is also called {v}.",
        "Type": "{e} is {a}.",
        "Material": "It is made of {v}.",
        "Current owner": "It is currently held by {v}.",
        "Previous owners": "It has been held by {v}.",
        "Reality": "{e} appears in {v}.",
    },
}

# A place is an "it"; a team is a "they". Neither is a "he" or a "she", and
# the Gender field they do not have was resolving to they/their by default.
KIND_SUBJECTS = {
    "location": ("it", "its", "it"),
    "item": ("it", "its", "it"),
    "team": ("they", "their", "them"),
    "event": ("it", "its", "it"),
    "story arc": ("it", "its", "it"),
}


def subject_for(kind: str, gender: str):
    """(subject, possessive, object) for a record of this kind."""
    if kind in KIND_SUBJECTS:
        return KIND_SUBJECTS[kind]
    return pronouns(gender)


CITATION = re.compile(r"^[A-Z][\w' :.,-]{2,60} Vol \d+ \d+$")


def _usable(fields: dict, label: str, cap: int = 240):
    """Field value as prose, truncated at a list boundary rather than dropped.

    Dropping over-long values silently deleted the most interesting field:
    33% of `Powers` and 27% of `Abilities` exceeded the old cap, while
    `Identity status: Secret` (median 7 chars) always survived.
    """
    cleaned = clean_value(fields.get(label, ""))
    if not cleaned:
        return None
    items = [p.strip() for p in cleaned.split(";") if p.strip()]
    if label in ("Powers", "Abilities"):
        # unwrap_templates() replaced a <ref> with its last argument, so a
        # citation ends up as an item: Spider-Man's powers began "Marvel Super
        # Heroes Secret Wars Vol 1 3", Storm's "House of X Vol 1 1". Rare
        # across the corpus and common among the characters people ask about,
        # because long pages carry more citations. A comic title is never a
        # power; First appearance keeps its citations because it IS one.
        kept = [it for it in items if not CITATION.match(it)]
        items = kept or items
    kept, used = [], 0
    for it in items:
        if kept and used + len(it) + 2 > cap:
            break
        kept.append(it)
        used += len(it) + 2
    if not kept:
        return None
    # A single item can be the whole cap and more: Spider-Man's Abilities is
    # one 2,000-character essay, and the loop above never breaks on it because
    # it only breaks once something is already kept.
    if len(kept[0]) > cap:
        cut = kept[0].rfind(". ", 0, cap)
        kept = [kept[0][:cut + 1] if cut > 40 else kept[0][:cap].rstrip() + "."]
    return humanize("; ".join(kept))


PRONOUNS = {"male": ("he", "his", "him"), "female": ("she", "her", "her")}


def pronouns(gender: str):
    """(subject, possessive, object). Unknown or non-binary -> they/their/them."""
    return PRONOUNS.get((gender or "").strip().lower(), ("they", "their", "them"))


DETAIL_MIN_FIELDS = 4      # below this a record cannot support "tell me everything"


ANY_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


TAUTOLOGY_LABELS = ("Full name", "Formal name",
                     # Location geography labels: a country, city or planet
                     # can be its own entry, e.g. Country: Abysmia on the
                     # Abysmia page, composing "Abysmia is in Abysmia."
                     "Country", "City", "State", "Province", "Region",
                     "Planet", "Continent", "Locale")


def _tautology(label: str, value: str, entity: str) -> bool:
    """`Adanna Gui's real name is Adanna Gui.` carries nothing."""
    if label not in TAUTOLOGY_LABELS:
        return False
    # Compare names with ANY parenthetical stripped, not just a reality tag.
    # "Civil War (Event)" carries Full name "Civil War", so the old check saw
    # two different strings and the answer opened "Civil War (Event)'s real
    # name is Civil War."
    return (ANY_PAREN.sub("", value).strip().lower()
            == ANY_PAREN.sub("", entity).strip().lower())


HEADING = re.compile(r"=+[^=]+=+\s*;?")


def history_excerpt(record: str, max_chars: int = 420) -> str:
    """First prose of the History section, trimmed at a sentence boundary."""
    _, _, body = record.partition("History:")
    # "===The Hood===;" is a section heading curation left in place. It reads
    # as debris in an answer, and the dataset taught the model to emit it.
    body = HEADING.sub(" ", body)
    body = " ".join(body.split()).lstrip("; ").strip()
    if not body:
        return ""
    if len(body) <= max_chars:
        return body
    cut = body[:max_chars]
    stop = cut.rfind(". ")
    return (cut[:stop + 1] if stop > 80 else cut).strip()


def open_ended_pair(record: str, seed: int = 0, detail: bool = False):
    """'Tell me about X' -> a couple of sentences.

    `detail=True` answers "tell me about X in detail" with more fields plus a
    History excerpt. Without this every answer is the same length and the model
    can never respond to how the question was asked.
    """
    rng = random.Random(f"open|{entity_name(record)}|{seed}|{detail}")
    fields = parse_fields(record)
    ent = bare_name(entity_name(record))
    kind = kind_of(record)
    subj, poss, _ = subject_for(kind, fields.get("Gender", ""))
    narrative = dict(NARRATIVE)
    narrative.update(NARRATIVE_BY_KIND.get(kind, {}))
    raw = [(lab, t) for lab, t in narrative.items()
           if _usable(fields, lab) and not _tautology(lab, _usable(fields, lab), ent)]
    # A detail request answered from a near-empty record teaches the model that
    # "tell me everything" sometimes warrants two sentences.
    if detail and len(raw) < DETAIL_MIN_FIELDS and not history_excerpt(record):
        return None
    sentences = []
    for i, (lab, t) in enumerate(raw[:6 if detail else 3]):
        # name the subject once, then use pronouns: repeating "The Hood" in
        # every sentence would teach a robotic register
        who = ent if i == 0 else subj.capitalize()
        who_poss = f"{ent}'s" if i == 0 else poss.capitalize()
        value = _usable(fields, lab)
        s = t.format(e=ent, v=value, a=with_article(value))
        if i > 0:
            # SUBJECT position only. Replacing the name everywhere rewrote it
            # inside the field's own prose: "the Ancient One possesses" became
            # "the He possesses", and "known as the Ancient Ones" became
            # "the Hes".
            if s.startswith(f"{ent}'s"):
                s = who_poss + s[len(ent) + 2:]
            elif s.startswith(ent):
                s = who + s[len(ent):]
        sentences.append(s)
    created = _usable(fields, "Created by")
    first = _usable(fields, "First appearance")
    if not sentences and not created and not first:
        return None
    body = " ".join(sentences)
    if created:
        tail = f" {ent} was created by {created}"
        tail += f" and first appeared in {first}." if first else "."
        body += tail
    elif first:
        body += f" {ent} first appeared in {first}."
    if detail:
        hist = history_excerpt(record)
        if hist:
            body += " " + hist
    q = rng.choice(DETAIL_Q if detail else OPEN_ENDED_Q).format(e=ent)
    return q, body.strip()


def combined_pair(record: str, seed: int = 0, detail_only: bool = False):
    """'Tell me about X in detail and which comic did he appear in first?'"""
    rng = random.Random(f"comb|{entity_name(record)}|{seed}")
    fields = parse_fields(record)
    first = _usable(fields, "First appearance")
    if not first:
        return None
    detail = True if detail_only else rng.random() < 0.5
    base = open_ended_pair(record, seed=seed, detail=detail)
    if not base:
        return None
    ent = bare_name(entity_name(record))
    q = base[0].rstrip("?.") + " and " + rng.choice(
        ["which comic did he first appear in?",
         "where did he first show up?",
         "what was his first appearance?",
         "which issue should I read first?"])
    a = base[1]
    if first not in a:
        a += f" {ent} first appeared in {first}."
    return q, a


def multi_field_pair(record: str, seed: int = 0):
    """'Where is X from and what powers does he have' - two asks, one answer."""
    rng = random.Random(f"multi|{entity_name(record)}|{seed}")
    fields = parse_fields(record)
    ent = bare_name(entity_name(record))
    have = [lab for lab in TEMPLATES if _usable(fields, lab)]
    if len(have) < 2:
        return None
    a_lab, b_lab = rng.sample(have, 2)
    qa = TEMPLATES[a_lab][0][0].format(e=ent).rstrip("?.")
    qb = TEMPLATES[b_lab][0][0].format(e=ent).rstrip("?.")
    # Second clause drops the repeated name and is lowercased, or it reads as
    # two sentences jammed together. The possessive has to be replaced first,
    # or "What is X's real name" becomes "what is he's real name".
    subj, poss, _ = subject_for(kind_of(record), fields.get("Gender", ""))
    qb_short = qb.replace(f"{ent}'s", poss).replace(ent, subj)
    qb_short = qb_short.replace("  ", " ").strip()
    qb_short = qb_short[:1].lower() + qb_short[1:]
    question = f"{qa} and {qb_short}?"
    va, vb = _usable(fields, a_lab), _usable(fields, b_lab)
    ans = (TEMPLATES[a_lab][1][0].format(e=ent, v=va, a=with_article(va)) + " "
           + TEMPLATES[b_lab][1][0].format(e=ent, v=vb, a=with_article(vb)))
    return question, ans


def where_to_read_pair(record: str, seed: int = 0):
    """'Where can I read about X' maps onto first appearance."""
    rng = random.Random(f"read|{entity_name(record)}|{seed}")
    fields = parse_fields(record)
    first = _usable(fields, "First appearance")
    if not first:
        return None
    ent = bare_name(entity_name(record))
    return (rng.choice(WHERE_TO_READ_Q).format(e=ent),
            f"{ent} first appeared in {first}, which is the place to start.")


def unknown_pair(record: str, missing_entity: str, seed: int = 0):
    """Context about someone else -> admit ignorance rather than invent."""
    rng = random.Random(f"unk|{missing_entity}|{seed}")
    q = rng.choice(OPEN_ENDED_Q + ["Who created {e}?", "What are {e}'s powers?"])
    return (q.format(e=missing_entity),
            rng.choice(UNKNOWN_A).format(e=missing_entity))


def format_example(context: str, user: str, assistant: str) -> str:
    head = f"{context}\n\n" if context else ""
    return f"{head}{USER_TAG} {user}\n{ASSISTANT_TAG} {assistant}"


def split_prompt(example: str):
    """(prompt, answer) so training can score only the assistant turn."""
    i = example.rindex(ASSISTANT_TAG) + len(ASSISTANT_TAG)
    return example[:i], example[i:]


def read_records(path: Path):
    buf = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(SEPARATOR):
                rec = "".join(buf).strip()
                if rec:
                    yield rec
                buf = []
            else:
                buf.append(line)
    rec = "".join(buf).strip()
    if rec:
        yield rec


_CTX_MODULE = None


def _ctx():
    """The one context builder, shared with retrieval at inference time.

    Cached: this is called once per generated example, and the dataset runs to
    hundreds of thousands of them.
    """
    global _CTX_MODULE
    if _CTX_MODULE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "context", BASE_DIR / "retrieve" / "context.py")
        _CTX_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_CTX_MODULE)
    return _CTX_MODULE


# "They" takes a plural verb. The NARRATIVE templates hardcode the singular
# ("{e} is {v}."), and the pronoun pass rewrote the subject but not the verb,
# so the previous dataset said "They is affiliated with..." 7,035 times.
DISAGREEMENTS = (
    ("They is", "They are"), ("They was", "They were"),
    ("They has", "They have"), ("They does", "They do"),
    ("They isn't", "They aren't"), ("They wasn't", "They weren't"),
    ("They hasn't", "They haven't"), ("They doesn't", "They do not"),
    # Every other NARRATIVE template uses "is", which the pairs above
    # cover. Occupation uses "works", and "They works as a Vigilante"
    # reached real output.
    ("They works", "They work"),
)


def question_case(question: str, rng) -> str:
    """Vary how a question is typed, not what it asks.

    Questions are built from record titles, so every training question carried
    the record's own capitalisation and a trailing "?". A lowercase query was
    then out of distribution: "Who created Wolverine?" answered correctly with
    Len Wein and John Romita, while "who created wolverine" returned Jeff
    Parker and Gabriel Hardman - a different creative team entirely. Same
    failure shape as "who are you?" working while "who are you" did not.

    The ANSWER is never lowercased; only the question, which is what varies in
    real use.
    """
    r = rng.random()
    if r < 0.40:
        return question
    if r < 0.70:
        return question.lower()
    if r < 0.85:
        return question.rstrip("?").rstrip()
    return question.lower().rstrip("?").rstrip()



# A sentence ends once. Field values usually carry their own full stop - "The
# full extent of this ability are unknown." - and every template appends
# another, so 19 of 24 sampled answers contained "..". The model learns the
# doubled stop from the data and reproduces it faithfully.
TERMINATOR_RUN = re.compile(r"[.?!]{2,}")


def _single(match) -> str:
    run = match.group(0)
    mark = max(run.rfind("?"), run.rfind("!"))
    if mark >= 0:
        return run[:mark + 1]              # "What If...?." -> "What If...?"
    # All dots. Two is a doubled stop; three is an ellipsis, and "What If..."
    # is a real title, so only the appended fourth comes off.
    return "." if len(run) == 2 else "..."


def one_terminator(text: str) -> str:
    """Collapse a doubled sentence terminator, leaving ellipses alone."""
    return TERMINATOR_RUN.sub(_single, text)


def agree(text: str) -> str:
    """Fix subject-verb agreement left behind by a pronoun substitution."""
    for wrong, right in DISAGREEMENTS:
        text = text.replace(wrong, right).replace(wrong.lower(), right.lower())
    return text


def context_for(record: str, max_chars: int = 700, detail: bool = False) -> str:
    """The record as retrieval hands it over.

    `detail=True` also carries the History excerpt, because a detailed answer
    drawing on facts absent from its own context trains the model to invent
    them.

    The formatting lives in retrieve/context.py so that what training sees and
    what inference sends cannot diverge.
    """
    return _ctx().build_context(
        [record], per_record_chars=max_chars,
        histories=[history_excerpt(record)] if detail else None)


def context_with_distractors(record, others, detail=False, rng=None):
    """The answer's record plus decoys, shuffled.

    Inference sends the top 3 retrieved records, so training on a single
    record would teach the model that the context is always about the entity
    it was asked about. With decoys it must read rather than assume, which is
    what makes a wrong top-1 retrieval survivable. The answer's record is
    deliberately not always first, so position cannot substitute for reading.

    The answer's record is assembled FIRST and always keeps its slot. Letting
    it compete with decoys for the budget dropped or truncated it in ~0.4% of
    examples, each of which taught the model to state a fact its own context
    did not contain.
    """
    ctx = _ctx()
    mine = ctx.assemble_blocks(
        [record], histories=[history_excerpt(record)] if detail else None)
    if not mine:
        return ""
    blocks, used = list(mine), sum(len(b) for b in mine)
    for block in ctx.assemble_blocks(list(others)):
        if used + len(block) > ctx.TOTAL_CHARS:
            break
        blocks.append(block)
        used += len(block)
    if rng is not None:
        rng.shuffle(blocks)
    return ctx.join_blocks(blocks)



def build(limit: int = 0, chitchat_repeats: int = 30000) -> int:
    """Mix of question shapes, because people do not ask in one form.

    Single-field templates alone would teach the model to expect a tidy
    "who created X?" every time. Real questions are open-ended, multi-part,
    ask for detail, or are about characters that do not exist at all.

    The rates below are a rebalance. The previous dataset was 75.7%
    single-field with a median answer of 8 words, and only 2.6% detailed, so
    the model would have learned terseness as its default register. Rich
    shapes are now the majority and chit-chat is no longer 0.3%, which was
    too thin for "hi" to be answered naturally.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260830)
    counts = Counter()
    recent_names = []
    recent_records = []
    n = 0
    with OUT_FILE.open("w", encoding="utf-8") as fh:
        def emit(ctx, q, a, kind):
            nonlocal n
            # agree() here rather than in each generator: every answer passes
            # through this one place, so a template that grows a new singular
            # verb cannot reintroduce "They is".
            # question_case here rather than in each generator: every
            # question passes through this one place, so a new question
            # shape cannot quietly reintroduce the fixed-case assumption.
            # Chit-chat brings its own variants and keeps them.
            user = q if kind == "chitchat" else question_case(q, rng)
            fh.write(json.dumps({"context": ctx, "user": user,
                                 "assistant": one_terminator(agree(a)),
                                 "kind": kind},
                                ensure_ascii=False) + "\n")
            counts[kind] += 1
            n += 1

        def decoys(rec, want=2):
            pool = [r for r in recent_records if r is not rec]
            return rng.sample(pool, min(want, len(pool))) if pool else []

        for f in sorted(CURATED.glob("*.txt")):
            if f.stat().st_size < 1000:
                continue
            for rec in read_records(f):
                seed = rng.randrange(10000)
                others = decoys(rec)
                ctx = context_with_distractors(rec, others, rng=rng)
                ctx_detail = context_with_distractors(rec, others, detail=True, rng=rng)

                # Short factual lookups still matter: they are what teaches the
                # model to READ the context instead of recalling. Capped at two
                # per record so they inform the model without defining it.
                if rng.random() < 0.55:
                    for q, a in qa_pairs(rec, seed=seed, max_pairs=2):
                        emit(ctx, q, a, "single-field")

                r = open_ended_pair(rec, seed=seed)
                if r and rng.random() < 0.85:
                    emit(ctx, r[0], r[1], "open-ended")

                if rng.random() < 0.55:
                    r = open_ended_pair(rec, seed=seed, detail=True)
                    if r:
                        # detailed answers draw on History, so the context must
                        # carry it or we are training the model to invent detail
                        emit(ctx_detail, r[0], r[1], "detailed")

                if rng.random() < 0.30:
                    r = multi_field_pair(rec, seed=seed)
                    if r:
                        emit(ctx, r[0], r[1], "multi-field")

                if rng.random() < 0.12:
                    r = where_to_read_pair(rec, seed=seed)
                    if r:
                        emit(ctx, r[0], r[1], "where-to-read")

                if rng.random() < 0.30:
                    r = combined_pair(rec, seed=seed)
                    if r:
                        emit(ctx_detail, r[0], r[1], "combined")

                # "not in my sources" - the behaviour that stops the model
                # confidently inventing an answer for anything it is asked
                name = bare_name(entity_name(rec))
                if recent_names and rng.random() < 0.12:
                    other = rng.choice(recent_names)
                    if other != name:
                        q, a = unknown_pair(rec, other, seed=seed)
                        emit(ctx, q, a, "unknown")
                recent_names.append(name)
                if len(recent_names) > 500:
                    recent_names.pop(0)
                recent_records.append(rec)
                if len(recent_records) > 300:
                    recent_records.pop(0)

                if limit and n >= limit:
                    break
            if limit and n >= limit:
                break

        # chit-chat carries no context, teaching the model that an empty
        # Context block is normal rather than a signal to invent one
        # Sample a VARIANT, not the exact prompt: the model must learn the
        # category, not 24 memorised strings.
        for _ in range(chitchat_repeats):
            user, answer = rng.choice(CHITCHAT)
            emit("", rng.choice(chitchat_variants(user)), answer, "chitchat")

    mb = OUT_FILE.stat().st_size / 1e6
    print(f"wrote {n:,} examples ({mb:,.1f} MB) -> {OUT_FILE}\n")
    print(f"{'kind':<16}{'count':>10}{'share':>8}")
    for kind, c in counts.most_common():
        print(f"{kind:<16}{c:>10,}{c/n*100:>7.1f}%")
    return 0



def preview(k: int = 3) -> int:
    if not OUT_FILE.exists():
        raise SystemExit("no dataset â€” run: py train/sft_data.py --build")
    import itertools
    with OUT_FILE.open(encoding="utf-8") as fh:
        rows = list(itertools.islice(fh, 4000))
    rng = random.Random(7)
    for line in rng.sample(rows, min(k, len(rows))):
        r = json.loads(line)
        print("=" * 66)
        print(format_example(r["context"], r["user"], r["assistant"]))
    print("=" * 66)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.build:
        return build(args.limit)
    if args.preview:
        return preview()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())




