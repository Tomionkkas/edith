#!/usr/bin/env python3
"""Marvel SLM — curation step.

Converts raw Fandom wikitext (crawl/raw/*.jsonl) into clean training text:
  - characters: consistent "key: value" fact lines + cleaned History prose
  - issues: publication line, per-story credits line, synopsis prose,
    chronology notes

Why this shape: a small model encodes repeated, explicit patterns far better
than dense prose it has to mine relationships out of. Consistent labels are
learnable.

Usage:
  python curate.py --demo            # fetch Spider-Man, curate + ASM Vol 1 300, print
  python curate.py --run characters  # curate a whole raw file -> curated/<name>.txt
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "raw")
CURATED_DIR = os.path.join(ROOT, "curated")
SAMPLES_DIR = os.path.join(CURATED_DIR, "samples")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
BASE = "https://marvel.fandom.com/api.php"

# ---------------------------------------------------------------- stripping

REF_RE = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
FILE_PREFIX_RE = re.compile(r"\[\[\s*(?:File|Image|Media)\s*:", re.I)
WIKILINK2_RE = re.compile(r"\[\[([^\]|]*)\|([^\]]*)\]\]")
WIKILINK1_RE = re.compile(r"\[\[([^\]]*)\]\]")
BOLD_RE = re.compile(r"'{2,}")   # wiki bold/italic only; keep lone apostrophes
TAG_RE = re.compile(r"<[^>]+>")
MATH_RE = re.compile(r"<math>.*?</math>", re.S)
NOWIKI_RE = re.compile(r"</?nowiki>")
# `[[Category:X]]` is filing, not prose. Unwrapping it to its label the way an
# ordinary wikilink is unwrapped leaves the literal words "Category:X" glued to
# the text before it. Unlike a File link it carries no caption, so it never
# nests and a flat match is enough.
CATEGORY_LINK_RE = re.compile(r"\[\[\s*Category\s*:[^\]]*\]\]", re.I)
# `[[:Category:X]]` - note the leading colon - is the opposite: a link TO the
# category rather than filing under it, and it is how the corpus writes
# `Relatives: [[:Category:Anu Family]]`. Dropping it empties the field, so the
# label is what survives.
CATEGORY_REF_RE = re.compile(
    r"\[\[\s*:\s*Category\s*:([^\]|]*)(?:\|[^\]]*)?\]\]", re.I)


# Templates carrying no prose value: citations, navboxes, magic words.
# Everything else is treated as a LINK template and unwrapped to its label.
DROP_TEMPLATES = {
    "r", "citation", "cite", "cite web", "cite book", "cite news",
    "navigation", "pagename", "reflist", "image wiki", "marvel database",
    "seealso", "see also", "main", "further",
}


def strip_file_links(text):
    """Remove [[File:...]] / [[Image:...]] blocks, caption and all.

    Captions routinely contain their own [[wikilinks]], so a regex that stops
    at the first ]] ends the match early and leaves the caption tail behind as
    debris. Match brackets by depth instead.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i:i + 2] == "[[" and FILE_PREFIX_RE.match(text, i):
            depth, j = 0, i
            while j < n:
                if text[j:j + 2] == "[[":
                    depth += 1; j += 2
                elif text[j:j + 2] == "]]":
                    depth -= 1; j += 2
                    if depth <= 0:
                        break
                else:
                    j += 1
            i = j
        else:
            out.append(text[i]); i += 1
    return "".join(out)


def _split_top_level(s):
    """Split on '|' that sits at brace/bracket depth 0."""
    parts, buf, depth, i, n = [], [], 0, 0, len(s)
    while i < n:
        two = s[i:i + 2]
        if two in ("{{", "[["):
            depth += 1; buf.append(two); i += 2; continue
        if two in ("}}", "]]"):
            depth -= 1; buf.append(two); i += 2; continue
        if s[i] == "|" and depth == 0:
            parts.append("".join(buf)); buf = []; i += 1; continue
        buf.append(s[i]); i += 1
    parts.append("".join(buf))
    return parts


def unwrap_templates(text):
    """{{Link|Target|Label}} -> Label, {{Navbox|...}} -> '' (depth-aware).

    Marvel Fandom uses templates as inline entity links ({{m}}, {{cl}}, {{DC}},
    {{Power}}, {{wc}}...). Deleting them wholesale removes exactly the proper
    nouns the corpus exists to teach, so keep the last positional argument.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i:i + 2] == "{{":
            depth, j = 0, i
            while j < n:
                if text[j:j + 2] == "{{":
                    depth += 1; j += 2
                elif text[j:j + 2] == "}}":
                    depth -= 1; j += 2
                    if depth <= 0:
                        break
                else:
                    j += 1
            args = _split_top_level(text[i + 2:j - 2])
            name = args[0].strip().lower()
            if name in DROP_TEMPLATES:
                out.append("")
            elif len(args) == 1:
                out.append(args[0].strip())          # bare {{Word}} -> Word
            else:
                positional = [a for a in args[1:] if "=" not in a]
                label = unwrap_templates(positional[-1]).strip() if positional else ""
                # a label spanning lines / carrying params is navbox debris
                nl = chr(10)
                out.append("" if (nl in label or "=" in label) else label)
            i = j
        else:
            out.append(text[i]); i += 1
    return "".join(out)


def strip_markup(text):
    """Wikitext -> clean plain text."""
    if not text:
        return ""
    t = COMMENT_RE.sub("", text)
    t = REF_RE.sub("", t)
    t = MATH_RE.sub("", t)
    t = NOWIKI_RE.sub("", t)
    t = strip_file_links(t)
    t = CATEGORY_REF_RE.sub(r"\1", t)
    t = CATEGORY_LINK_RE.sub("", t)
    # Templates first (depth-aware, keeps the display label), then wikilinks.
    t = unwrap_templates(t)
    while True:
        t2 = WIKILINK2_RE.sub(r"\2", t)
        t2 = WIKILINK1_RE.sub(r"\1", t2)
        if t2 == t:
            break
        t = t2
    t = t.replace("<br", "\n<br")
    t = t.replace("'''", "").replace("''", "")
    t = TAG_RE.sub("", t)
    t = BOLD_RE.sub("", t)
    t = t.replace("__NOTOC__", "").replace("__NOEDITSECTION__", "")
    # collapse whitespace, keep paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in t.split("\n")]
    out, prev_blank = [], False
    for ln in lines:
        if not ln:
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(ln)
            prev_blank = False
    return "\n".join(out).strip()


# ------------------------------------------------------------- field parser

FIELD_LINE_RE = re.compile(r"^\|\s*([A-Za-z0-9_\-]+)\s*=\s*(.*)$")
LIST_ITEM_RE = re.compile(r"^\*+\s*(.*)$")


def find_template_span(wikitext):
    """Locate the first {{...Template...}} block via brace-depth matching.

    Returns (start, end) char indices inclusive of braces, or None.
    A line-based "stop at first bare }}" breaks on nested templates
    (e.g. Affiliation = {{Navigation|...}}).
    """
    n = len(wikitext)
    idx = 0
    start = None
    while True:
        i = wikitext.find("{{", idx)
        if i == -1:
            return None
        stop = n
        for marker in ("|", "}}"):
            p = wikitext.find(marker, i)
            if p != -1 and p < stop:
                stop = p
        if "template" in wikitext[i:stop].lower():
            start = i
            break
        idx = i + 2
    depth = 0
    j = start
    while j < n - 1:
        if wikitext[j:j + 2] == "{{":
            depth += 1
            j += 2
        elif wikitext[j:j + 2] == "}}":
            depth -= 1
            j += 2
            if depth == 0:
                return start, j
        else:
            j += 1
    return start, n - 1




def parse_template_fields(wikitext):
    """Return ordered {field: value-or-list} for the first template block.

    Values are lists of display strings; sub-labels (": Formerly:") are
    prefixed onto the items that follow them. A one-item list is collapsed
    to a scalar.
    """
    span = find_template_span(wikitext)
    if span is None:
        return {}
    a, b = span
    inner = wikitext[a + 2:b - 1] if b > a + 3 else ""
    inner = unwrap_templates(inner)
    fields = {}
    order = []
    cur = None
    for ln in inner.split("\n"):
        s = ln.strip()
        if not s or s in ("|", "||"):
            continue
        m = FIELD_LINE_RE.match(s)
        if m:
            cur = m.group(1)
            if cur not in fields:
                fields[cur] = []
                order.append(cur)
            val = m.group(2).strip()
            lm = re.match(r"^:\s*(.*)$", val)
            if lm and lm.group(1).strip():
                # e.g. `| Formerly: = Captain America` -> label "Formerly"
                fields[cur].append((lm.group(1).strip().rstrip(":"), ""))
            elif val:
                for it in val.split(";"):
                    it = re.sub(r"^[*]+\s*", "", it).strip()
                    if it:
                        fields[cur].append((None, it))
            continue
        if cur is None:
            continue
        lm = re.match(r"^:(\s*)(.*)$", s)
        if lm and lm.group(2).strip():
            fields[cur].append((lm.group(2).strip().rstrip(":"), ""))
            continue
        li = LIST_ITEM_RE.match(s)
        if li:
            it = li.group(1).strip()
            if it:
                it = re.sub(r"^[*]+\s*", "", it.split(";")[0]).strip()
                if it:
                    fields[cur].append((None, it))
            continue
        # plain prose continuation: attach to the current field as its own item
        fields[cur].append((None, s))
    shaped = {}
    for k in order:
        parts = []
        cur_label = None
        for tag, txt in fields[k]:
            if tag is not None:
                cur_label = tag
                if not txt:
                    continue
            if txt:
                parts.append(f"{cur_label}: {txt}" if cur_label else txt)
        shaped[k] = parts[0] if len(parts) == 1 else parts
    return shaped


def body_sections(wikitext):
    """{section-name: raw-body} for == Heading == sections outside templates."""
    lines = wikitext.split("\n")
    # drop the template block if present (brace-depth aware)
    span = find_template_span(wikitext)
    if span is not None:
        a, b = span
        cut_lines = wikitext.count("\n", 0, a)
        tail_start_line = wikitext.count("\n", 0, b + 1)
        lines = lines[:cut_lines] + lines[tail_start_line:]
    sections = {}
    cur = None
    buf = []
    for ln in lines:
        m = re.match(r"^==+([^=]+)==+\s*$", ln)
        if m:
            if cur is not None:
                sections[cur] = "\n".join(buf).strip()
            cur = m.group(1).strip()
            buf = []
        elif cur is not None:
            buf.append(ln)
    if cur is not None:
        sections[cur] = "\n".join(buf).strip()
    return sections


def joinval(v):
    if isinstance(v, list):
        parts = [strip_markup(x) for x in v]
        return "; ".join(p for p in parts if p)
    return strip_markup(v)


# ------------------------------------------------------------- character

# A publication, not a person. EditorialNames carries comic titles beside
# real names - `2020 Machine Man Vol 1` sits next to `Spider-Man 2099` - and
# indexing one as a character name answers a question about that comic with a
# character card, which is the defect build()'s MIN_FIELDS branch already
# exists to prevent.
PUBLICATION = re.compile(r"\bVol\s+\d")


def codenames(fields):
    """The names people actually type, from the two fields that carry them.

    `Codenames` is where a character's codename lives and curation never read
    it: "Beast" appeared in no field of Henry McCoy's 105,792-character
    record, so `who is beast` could not reach him at all and answered from an
    obscure alien headlined the same. `EditorialNames` carries the rest -
    Miguel O'Hara is "Spider-Man 2099" there and nowhere else - at the cost
    of needing the publication filter above.

    Split on newlines as well as `;`: parse_template_fields already strips
    bullets and splits a `;`-list, but a multi-line `* [[A]]\\n* [[B]]` value
    reaches joinval() whole and comes back one-per-line with its bullets on.
    """
    out = []
    for key in ("Codenames", "EditorialNames"):
        for part in re.split(r"[;\n]", joinval(fields.get(key, ""))):
            part = re.sub(r"^[*]+\s*", "", part).strip()
            if part and not PUBLICATION.search(part) and part not in out:
                out.append(part)
    return "; ".join(out)


CHAR_FIELD_MAP = [
    ("Name", "Full name"),
    ("CurrentAlias", "Also known as"),
    ("Codenames", "Codename"),
    ("Aliases", "Other aliases"),
    ("First", "First appearance"),
    ("Creators", "Created by"),
    ("Role", "Role"),
    ("Identity", "Identity status"),
    ("Origin", "Species / origin"),
    ("Reality", "Reality"),
    ("Citizenship", "Citizenship"),
    ("PlaceOfBirth", "Place of birth"),
    ("Education", "Education"),
    ("Occupation", "Occupation"),
    ("BaseOfOperations", "Base of operations"),
    ("Affiliation", "Affiliation"),
    ("Relatives", "Relatives"),
    ("MaritalStatus", "Marital status"),
    ("Gender", "Gender"),
    ("Powers", "Powers"),
    ("Abilities", "Abilities"),
    ("Weaknesses", "Weaknesses"),
    ("Equipment", "Equipment"),
    ("Weapons", "Weapons"),
    ("Transportation", "Transportation"),
    ("UnusualFeatures", "Unusual features"),
    ("Notes", "Notes"),
    ("Trivia", "Trivia"),
]


# Items and Locations use their own Marvel Database templates, so their fields
# are absent from CHAR_FIELD_MAP and would be dropped in silence. Both lists
# are taken from what the 3,655 item and 10,647 location pages actually carry,
# not from the template documentation: every key below appears on at least 5%
# of them, and the ones at 100% are the schema.
ITEM_FIELD_MAP = [
    ("Name", "Formal name"),
    ("Aliases", "Other aliases"),
    ("Type", "Type"),
    ("First", "First appearance"),
    ("Creators", "Created by"),
    ("Reality", "Reality"),
    ("Origin", "Origin"),
    ("LeadDesigner", "Lead designer"),
    ("AdditionalDesigners", "Additional designers"),
    ("PlaceOfCreation", "Place of creation"),
    ("PlaceOfDestruction", "Place of destruction"),
    ("CurrentOwner", "Current owner"),
    ("PreviousOwners", "Previous owners"),
    ("AlternateOwners", "Alternate owners"),
    ("Material", "Material"),
    ("Dimensions", "Dimensions"),
    ("Weight", "Weight"),
    ("Model", "Model"),
    ("Version", "Version"),
    ("Properties", "Properties"),
    ("Notes", "Notes"),
    ("Trivia", "Trivia"),
]

LOCATION_FIELD_MAP = [
    ("Name", "Formal name"),
    ("Aliases", "Other aliases"),
    ("First", "First appearance"),
    ("Creators", "Created by"),
    ("Reality", "Reality"),
    ("Galaxy", "Galaxy"),
    ("StarSystem", "Star system"),
    ("Planet", "Planet"),
    ("Continent", "Continent"),
    ("Country", "Country"),
    ("Region", "Region"),
    ("State", "State"),
    ("Province", "Province"),
    ("City", "City"),
    ("Locale", "Locale"),
    ("Capital", "Capital"),
    ("Demonym", "Demonym"),
    ("Language", "Language"),
    ("Religion", "Religion"),
    ("Government", "Government"),
    ("Population", "Population"),
    ("Dimension", "Dimension"),
    ("PointsOfInterest", "Points of interest"),
    ("Residents", "Residents"),
    ("Notes", "Notes"),
    ("Trivia", "Trivia"),
]

MAIN_CONTINUITY = "Earth-616"
REALITY_IN_TITLE_RE = re.compile(r"\((Earth-[0-9A-Za-z\-]+)\)")


def is_main_continuity_reality(reality: str) -> bool:
    """Earth-616 anywhere in the list means main continuity.

    An item's Reality field is a LIST - Adamantium exists in five - and an
    exact comparison against "Earth-616" called every one of them an
    alternate continuity.
    """
    return any(part.strip() == MAIN_CONTINUITY for part in (reality or "").split(";"))


def normalise_reality(value):
    """`6160` -> `Earth-6160`. Leaves `Earth-616` and prose values alone."""
    v = (value or "").strip()
    if re.fullmatch(r"\d{2,7}", v):
        return "Earth-" + v
    return v


def reality_of(fields, title):
    """Best available continuity id: the Reality field, else the page title."""
    r = normalise_reality(joinval(fields.get("Reality", "")))
    if not r:
        m = REALITY_IN_TITLE_RE.search(title)
        if m:
            r = m.group(1)
    return r


def curate_character(rec, field_map=CHAR_FIELD_MAP, kind="character"):
    """Raw record -> clean training text for one templated page.

    `field_map` is what makes it work for items and locations too: the
    headline rule, the reality rule, the `Page:` line and the History
    fallback are the same for every Marvel Database template, and only the
    schema differs. Items and locations carry no CurrentAlias, so their
    headline is the page title, which is what people call them.
    """
    title = rec["title"]
    wt = rec["wikitext"]
    fields = parse_template_fields(wt)
    sections = body_sections(wt)

    # Marvel has many Doctor Stranges. The reality shows up as a field further
    # down, but the first line is what anchors a record, so alternate
    # continuities carry theirs up front. Earth-616 is the unmarked default.
    reality = reality_of(fields, title)
    # `fields.get(..., "")`, not `fields["CurrentAlias"]`: a CurrentAlias can
    # be present but strip to nothing (its value is an HTML comment, e.g.
    # `<!-- Psykos -->`), and `or title` catches both that case and the
    # field's ordinary absence the same way.
    display = joinval(fields.get("CurrentAlias", "")) or title
    if reality and not is_main_continuity_reality(reality) and reality not in display:
        display = f"{display} ({reality})"

    # The page title is a name too, and the headline above threw it away.
    # 90.9% of character pages have a CurrentAlias that differs from their
    # title, and the alias is the right headline - but `Jean Grey (Earth-616)`
    # is headlined `Phoenix` and carries the married Full name `Jean Elaine
    # Grey-Summers`, so nothing in the record said "Jean Grey" and her 224 KB
    # page was unreachable by her own name.
    out = []
    for key, label in field_map:
        # The one label built from two raw keys, and the only one that must
        # be emitted when its own key is absent: Miguel O'Hara carries
        # "Spider-Man 2099" in EditorialNames and has no Codenames at all.
        if key == "Codenames":
            v = codenames(fields)
            if v:
                out.append(f"{label}: {v}")
            continue
        if key in fields:
            v = joinval(fields[key])
            if key == "Reality":
                v = normalise_reality(v)
            if v:
                if key == "History":
                    continue  # handled separately
                out.append(f"{label}: {v}")

    # History: prefer the big template field; fall back to body sections
    hist = ""
    if fields.get("History"):
        hist = joinval(fields["History"])
    if not hist:
        hist = strip_markup(sections.get("History", ""))
    if hist:
        out.append("History:")
        out.append(hist)

    if not out:
        return ""
    # Prepended after the emptiness check, so a page with no fields at all is
    # still dropped rather than emitted as a headline and its own title.
    return (display + "\n"
            + "\n".join([f"Kind: {kind}", f"Page: {title}"] + out) + "\n")


def curate_item(rec, kind="item"):
    """Raw record -> clean training text for one item."""
    return curate_character(rec, ITEM_FIELD_MAP, kind)


def curate_location(rec, kind="location"):
    """Raw record -> clean training text for one location."""
    return curate_character(rec, LOCATION_FIELD_MAP, kind)


# ----------------------------------------------------------------- issue

ISSUE_SIMPLE = [
    ("ReleaseDate", "Released"),
]


def issue_credit_fields(fields, role):
    vals = []
    for k, v in fields.items():
        if re.fullmatch(role + r"\d+_\d+", k):
            j = joinval(v)
            if j and j not in vals:
                vals.append(j)
    return vals


def curate_issue(rec, kind="issue"):
    """Raw record -> clean training text for one issue."""
    title = rec["title"]
    wt = rec["wikitext"]
    fields = parse_template_fields(wt)
    sections = body_sections(wt)

    out = []
    rel = joinval(fields.get("ReleaseDate", ""))
    head = f"{title} is a Marvel comic."
    if rel:
        head += f" It was published on {rel}."
    out.append(head)

    # per-story credits
    story_titles = []
    for k in sorted(fields):
        m = re.fullmatch(r"StoryTitle(\d+)", k)
        if m:
            st = joinval(fields[k])
            if st:
                story_titles.append((m.group(1), st))
    if story_titles:
        for num, st in story_titles:
            line = f'Story {num} is titled "{st}".'
            writers = issue_credit_fields(fields, "Writer")
            pencils = issue_credit_fields(fields, "Penciler")
            inkers = issue_credit_fields(fields, "Inker")
            if writers:
                line += f" It was written by {', '.join(writers)}."
            if pencils:
                line += f" The art was by {', '.join(pencils)}."
            if inkers:
                line += f" It was inked by {', '.join(inkers)}."
            out.append(line)
    else:
        writers = issue_credit_fields(fields, "Writer")
        if writers:
            out.append(f"The issue was written by {', '.join(writers)}.")

    # synopsis
    syns = []
    for k in sorted(fields):
        m = re.fullmatch(r"Synopsis(\d+)", k)
        if m:
            s = joinval(fields[k])
            if s:
                syns.append(s)
    if syns:
        out.append("Synopsis:")
        out.append("\n\n".join(syns))

    # body sections: Chronology Notes / Continuity
    for name in ("Chronology Notes", "Continuity Notes", "Continuity"):
        if name in sections and sections[name]:
            out.append(f"{name}:")
            out.append(strip_markup(sections[name]))

    return "\n".join([out[0], f"Kind: {kind}"] + out[1:]) + "\n" if out else ""


# ------------------------------------------------------------------ demo

def fetch_title(title):
    p = urllib.parse.urlencode({"action": "query", "titles": title,
                                "prop": "revisions", "rvprop": "content",
                                "redirects": "1", "format": "json"})
    req = urllib.request.Request(BASE + "?" + p, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    for pg in data.get("query", {}).get("pages", {}).values():
        return {"title": pg.get("title", title),
                "wikitext": (pg.get("revisions") or [{}])[0].get("*") or ""}
    return None


def demo():
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    sm = fetch_title("Spider-Man (Earth-616)")
    if sm:
        text = curate_character(sm)
        with open(os.path.join(SAMPLES_DIR, "character_spiderman.txt"),
                  "w", encoding="utf-8") as f:
            f.write(text)
        print("=" * 70)
        print("CURATED CHARACTER — Spider-Man (Earth-616)")
        print(f"(raw wikitext: {len(sm['wikitext'])} chars -> "
              f"clean: {len(text)} chars)")
        print("=" * 70)
        print(text)

    sample_file = os.path.join(ROOT, "state", "asm300_sample.json")
    if os.path.exists(sample_file):
        with open(sample_file, encoding="utf-8") as f:
            rec = json.load(f)
        text = curate_issue(rec)
        with open(os.path.join(SAMPLES_DIR, "issue_asm_vol1_300.txt"),
                  "w", encoding="utf-8") as f:
            f.write(text)
        print("=" * 70)
        print("CURATED ISSUE — Amazing Spider-Man Vol 1 300 (Venom debut)")
        print(f"(raw wikitext: {rec['chars']} chars -> clean: {len(text)} chars)")
        print("=" * 70)
        print(text)


# The raw file IS the kind. run_phase() knows it and nothing downstream did,
# which is how a country came to be described as a person.
PHASE_KINDS = {
    "characters": "character",
    "comics": "issue",
    "teams": "team",
    "events": "event",
    "story_arcs": "story arc",
    "items": "item",
    "locations": "location",
    "mw_character_backstop": "character",
    "mw_marvel_backstop": "character",
    "patch": "character",
}


def kind_for(name):
    """Kind for a raw phase. Anything unlisted is character-shaped, which is
    what teams, events and story arcs already are."""
    return PHASE_KINDS.get(name, "character")


def curator_for(name):
    """Which curator a raw phase goes through.

    Anything not listed is a Marvel Database character-shaped template -
    teams, events and story arcs all share that schema.
    """
    return {"comics": curate_issue,
            "items": curate_item,
            "locations": curate_location}.get(name, curate_character)


def run_phase(name):
    """Curate raw/<name>.jsonl -> curated/<name>.txt (one block per page)."""
    os.makedirs(CURATED_DIR, exist_ok=True)
    src = os.path.join(RAW_DIR, name + ".jsonl")
    if not os.path.exists(src):
        print(f"no such raw file: {src}")
        return
    curator = curator_for(name)
    kind = kind_for(name)
    out_path = os.path.join(CURATED_DIR, name + ".txt")
    n = 0
    empty = 0
    with open(src, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = curator(rec, kind=kind)
            if not text:
                empty += 1
                continue
            fout.write(text + "\n" + "=" * 60 + "\n\n")
            n += 1
    print(f"{name}: curated {n} pages -> {out_path} ({empty} empty skipped)")


def run_all():
    """Curate every raw/<name>.jsonl that exists."""
    names = sorted(
        f.name[:-len(".jsonl")] for f in Path(RAW_DIR).glob("*.jsonl")
    )
    if not names:
        print(f"no raw jsonl files in {RAW_DIR} — nothing to curate")
        return
    for name in names:
        run_phase(name)


def main():
    if "--demo" in sys.argv:
        demo()
    elif "--run-all" in sys.argv:
        run_all()
    elif "--run" in sys.argv:
        name = sys.argv[sys.argv.index("--run") + 1]
        run_phase(name)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()