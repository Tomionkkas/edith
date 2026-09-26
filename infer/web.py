"""`edith --web` - the same answers as the terminal, with the art.

One decision, two renderers. This calls `engine.plan()` directly, which is
what stops the web answer drifting from the terminal answer - the same
argument `retrieve/context.py` settled for training vs inference, and the
reason there is no second copy of the retrieval logic here. Everything below
is transport: JSON out, static files out, nothing thought about.

The page is three files in `infer/web/`; every question is one POST. There is
no framework and no build step, because the package installs as a `uv` tool
and a dependency for one route is not worth carrying.

**One lock around plan().** The model is a single resident process; two
overlapping generations would interleave their sampling. One user asking one
question at a time is the whole concurrency story, so a lock is the correct
amount of machinery.
"""
import importlib.util
import json
import pathlib
import random
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / "web"

# The picker shows more than the terminal can. facts.variants() defaults to
# 10 because ten rows is what 80 columns hold; a page has room, and
# spider-man has 782 DISTINCT headlines behind it.
VARIANT_LIMIT = 40
LOCK = threading.Lock()

# One browser, one session: the picks made in it, name key -> doc id, exactly
# what terminal.Terminal.settled holds. Module-level for the same reason LOCK
# is: this serves one person on their own machine.
SETTLED = {}

# doc id -> sentences of that record already quoted this session, so a second
# "explain in more detail" reads on instead of repeating itself.
SHOWN = {}

CONTENT_TYPES = {".html": "text/html", ".css": "text/css",
                 ".js": "text/javascript", ".svg": "image/svg+xml"}


def _images():
    """`retrieve/images.py`, loaded by path like every other module here."""
    if "images" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "images", ROOT / "retrieve" / "images.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["images"] = module
        spec.loader.exec_module(module)
    return sys.modules["images"]


def _first_field(record: str, label: str) -> str:
    """The first entry of a field. Fields are joined with "; " corpus-wide."""
    return _field(record, label).split(";")[0].strip()


def _field(record: str, label: str) -> str:
    for line in record.split("\n"):
        if line.startswith(label + ":"):
            return line[len(label) + 1:].strip()
    return ""


def provenance_of(term, record, headline, key):
    """How this record holds the name that was ASKED: identity, codename or
    alias - 4.15's distinction, which is the one thing a picker of ten
    identically-headlined records can be sorted out by.

    `who is ghost rider` offers Johnny Blaze and Danny Ketch, who ARE Ghost
    Rider, beside records that are merely CALLED it. The terminal cannot show
    that; a badge on a card can.
    """
    if not (term.resolve and key):
        return None
    identity, codename, _alias = term.resolve.names_by_provenance(headline, record)
    if key in identity:
        return "identity"
    if key in codename:
        return "codename"
    return "alias"


def _generate(term, plan) -> str:
    """The model's answer, not the prompt that asks for it.

    `Plan.text` is set only when a record answered outright; otherwise the
    plan carries a PROMPT for the model to continue, and the terminal streams
    that continuation through `render.TidyStream`. This page had been
    rendering `plan.prompt` itself, so a question with no field answer
    printed the whole scaffold - "Context: ... User: cosmic spider
    Assistant:" - as though that were the reply.

    Same sampler settings the terminal uses (terminal.py:_generate), and the
    same `engine.tidy()` on the way out, which repairs what stage 3's dataset
    taught: 32,345 doubled full stops and 2,621 "They works".
    """
    if not plan.prompt or getattr(term, "model", None) is None:
        return plan.prompt or ""
    raw = term.sample.generate(term.model, term.sp, term.device, plan.prompt,
                               term.block, max_new=220, temp=0.7, top_k=50,
                               top_p=0.95, repetition_penalty=1.1)
    return term.engine.tidy(raw, term.sft).strip()


def _row(term, doc_id, provenance, images) -> dict:
    record = term.index.text(doc_id)
    headline = term.index.headlines[doc_id]
    return {"headline": headline, "size": len(record), "doc_id": doc_id,
            "page": _field(record, "Page") or None,
            "provenance": provenance,
            "images": images.urls_for(record, headline, term.images)}


def others_of(term, question, exclude, images, limit=24) -> list:
    """Who ELSE goes by this name - a different character, not another
    reality of the same one.

    `facts.variants()` answers "which Hulk", and every row is Bruce Banner
    from a different earth. It cannot answer "who else has been the Hulk",
    which is the more interesting question and the one the corpus already
    holds: She-Hulk, Brawn and Rick Jones all carry `Hulk` on a Codename or
    alias line. Spider-Man gives Chasm and Scorpion; Iron Man gives Emperor
    Doom and War Machine.

    Read straight off the names index that 4.11 gave provenance to, so this
    costs a dict lookup. build() keeps at most 8 entries per name, ranked, so
    a very long legacy is cut - Red Hulk and Maestro do not survive the cap.
    """
    if not (term.resolve and term.facts):
        return []
    key = term.resolve.query_key(term.facts.entity_text(question))
    if not key:
        return []

    # The sidecar first: it is uncapped where the names index keeps eight,
    # which is the difference between "Chasm and Scorpion" and the whole
    # legacy. Absent (not built, or an older install), fall back to the names
    # index so the group degrades rather than disappears.
    legacy = getattr(term, "legacy", None)
    if legacy:
        rows = legacy.get(" ".join(key), [])
        out = [_row(term, doc_id,
                    "codename" if prov == term.resolve.PROV_CODENAME else "alias",
                    images)
               for doc_id, prov in rows if doc_id not in exclude]
        return out[:limit]

    names = getattr(term.engine, "_NAMES", None) or {}
    out = []
    for doc_id, _size, _main, prov, _page, _famous in names.get(key, []):
        if prov >= term.resolve.PROV_IDENTITY or doc_id in exclude:
            continue
        out.append(_row(term, doc_id,
                        "codename" if prov == term.resolve.PROV_CODENAME
                        else "alias", images))
        if len(out) == limit:
            break
    return out


def _empty() -> dict:
    """A turn that asked nothing, in the shape the page renders from."""
    return {"answer": "", "doc_id": None, "headline": None, "page": None,
            "kind": None, "size": 0, "fields": [], "quoted": None,
            "quoted_leads": False, "exhausted": False, "corrected": None,
            "images": [], "choices": None, "others": None,
            "first_appearance": None, "variants_total": 0}


def answer(term, question, previous=None, settled=None) -> dict:
    """One turn, as JSON the page can render.

    `previous` is the record already on screen, threaded exactly as the
    terminal threads `self.last_doc`, so "what are his powers" answers from
    the record the last turn resolved instead of resolving the pronoun.
    """
    images = _images()
    if not question.strip():
        # The model is a continuation engine: asked to continue nothing it
        # invents something, and the page would render it as an answer.
        # MEASURED 2026-09-26 against the real endpoint - a request whose
        # question arrived empty came back with a paragraph about a comic
        # nobody had mentioned. The browser never sends one; an endpoint
        # answers whoever asks.
        return _empty()
    with LOCK:
        plan = term.engine.plan(question, term.index, term.sft, term.search,
                                term.disambiguate, term.facts, term.resolve,
                                previous=previous, settled=settled,
                                shown=SHOWN)
        text = plan.text if plan.text is not None else _generate(term, plan)

    doc_id = plan.doc_id
    record = term.index.text(doc_id) if doc_id is not None else ""
    headline = term.index.headlines[doc_id] if doc_id is not None else None

    choices = None
    variants_total = 0
    if plan.choices:
        # plan.choices decided WHETHER to offer; this asks the same function
        # for a longer list to SHOW. The engine is untouched.
        deep = []
        if term.facts is not None and hasattr(term.facts, "variants"):
            deep, variants_total = term.facts.variants(
                term.index, term.resolve, doc_id, limit=VARIANT_LIMIT)
        plan_choices = deep or plan.choices
    if plan.choices:
        # The name as ASKED, which is what provenance is judged against -
        # `facts.entity_text()` is the same string resolution used.
        key = None
        if term.resolve and term.facts:
            key = term.resolve.query_key(term.facts.entity_text(question))
        choices = []
        for size, choice_headline, choice_doc in plan_choices:
            choice_record = term.index.text(choice_doc)
            choices.append({
                "headline": choice_headline,
                "size": size,
                "doc_id": choice_doc,
                "page": _field(choice_record, "Page") or None,
                "provenance": provenance_of(term, choice_record,
                                            choice_headline, key),
                "images": images.urls_for(choice_record, choice_headline,
                                          term.images),
            })

    # Asked for more and got nothing back: the record has run out. Saying so
    # is the whole reason select() returns None past the end rather than
    # looping to the opening, which would read as the system forgetting.
    exhausted = bool(
        plan.quoted is None and doc_id is not None and SHOWN.get(doc_id)
        and term.resolve and term.facts
        and not term.resolve.query_key(term.facts.entity_text(question)))

    # The issue an answer CITES, shown rather than named - the comics tier
    # is 99.997% covered, so this nearly always has art.
    first = None
    if record:
        title = _first_field(record, "First appearance")
        cover = term.images.get(title) if title else None
        if title:
            first = {"title": title,
                     "image": images.url_for_name(cover) if cover else None}

    others = []
    if plan.choices:
        others = others_of(term, question,
                           {c["doc_id"] for c in choices}, images)

    return {
        "answer": text or "",
        "doc_id": doc_id,
        "headline": headline,
        "page": (_field(record, "Page") or None) if record else None,
        "kind": (_field(record, "Kind") or None) if record else None,
        "size": len(record),
        "fields": [[str(label), str(value)] for label, value in (plan.rows or [])],
        "quoted": plan.quoted,
        "quoted_leads": bool(plan.quoted_leads),
        "exhausted": exhausted,
        "corrected": plan.corrected,
        "images": images.urls_for(record, headline, term.images) if record else [],
        "choices": choices,
        "others": others or None,
        "first_appearance": first,
        "variants_total": variants_total,
    }


# A comic's page title is "<series> Vol <n> <issue>". Used only to pick the
# cold open's wall out of the sidecar without loading the index for it.
COMIC_TITLE = re.compile(r" Vol \d+ \d+$")
HERO = "James Howlett (Earth-616)"


def wall(term, count: int = 15, seed=None) -> dict:
    """Covers for the cold open, sampled from the corpus rather than chosen.

    Sampled, not curated, for the same reason the harnesses sample: a
    hand-picked wall shows the corpus at its best and tells you nothing. The
    browser fetches them; a machine with no network gets the mark alone,
    which is correct - EDITH itself runs offline.
    """
    images = _images()
    covers = [title for title in term.images if COMIC_TITLE.search(title)]
    rng = random.Random(seed)
    chosen = rng.sample(covers, min(count, len(covers))) if covers else []
    hero = term.images.get(HERO)
    return {
        "covers": [images.url_for_name(term.images[t]) for t in chosen],
        "hero": images.url_for_name(hero) if hero else None,
        "total": len(term.index.docs),
        "comics": len(covers),
    }


def settle(term, doc_id: int, settled: dict) -> None:
    """Remember that the user chose this record for this name.

    Copied rule-for-rule from `terminal.Terminal.settle()`, including the
    LOOP over PAREN_SUFFIX: a headline can carry more than one trailing
    disambiguator ("Sasquatch (Beast) (Earth-616)"), and stripping only the
    reality suffix leaves a key that can never match `query_key("beast")` -
    the pick then silently does nothing, with no error. If that rule changes
    there, it changes here.
    """
    head = term.index.headlines[doc_id]
    while True:
        stripped = term.resolve.PAREN_SUFFIX.sub("", head)
        if stripped == head:
            break
        head = stripped
    key = term.resolve.query_key(head.strip())
    if key:
        settled[key] = doc_id


def static_name(path: str) -> str:
    """The file a request path asks for. `/` is the page itself."""
    return "page.html" if path in ("", "/") else path.lstrip("/").split("?")[0]


def static_path(path: str):
    """The file under `infer/web/`, or None when the path climbs out of it.

    A request that resolves outside STATIC is a traversal attempt, not a
    typo, and gets a 404 rather than an explanation.
    """
    candidate = (STATIC / static_name(path)).resolve()
    if STATIC.resolve() not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


class Handler(BaseHTTPRequestHandler):
    term = None

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("", "/"):
            # Loading the page starts a session, the way running `edith`
            # starts one. Without this a pick made an hour ago still
            # silences the picker for that name, across reloads.
            SETTLED.clear()
            SHOWN.clear()
        if self.path == "/wall":
            body = json.dumps(wall(self.term)).encode("utf-8")
            return self._send(200, body, "application/json; charset=utf-8")
        path = static_path(self.path)
        if path is None:
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        ctype = CONTENT_TYPES.get(path.suffix, "text/plain")
        self._send(200, path.read_bytes(), ctype + "; charset=utf-8")

    def do_POST(self):
        if self.path != "/ask":
            return self._send(404, b"not found", "text/plain; charset=utf-8")
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._send(400, b"bad json", "text/plain; charset=utf-8")

        pick = body.get("pick")
        if pick is not None:
            # A click on a picker card IS the bare number the terminal takes:
            # settle the name, then ask the same sentinel it asks itself
            # (terminal.py:748), so the chosen record answers and the menu
            # does not re-open on the next turn.
            settle(self.term, int(pick), SETTLED)
            out = answer(self.term, "who is this", previous=int(pick),
                         settled=SETTLED)
        else:
            # 500 characters is longer than any question EDITH answers well
            # and short enough that nothing here is a memory decision.
            out = answer(self.term, str(body.get("question", ""))[:500],
                         previous=body.get("previous"), settled=SETTLED)
        self._send(200, json.dumps(out).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, *args):
        """Quiet: the terminal this runs in belongs to the user."""


def serve(term, host: str = "127.0.0.1", port: int = 8420) -> int:
    Handler.term = term
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"  EDITH on http://{host}:{port}   (ctrl-c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye.")
    finally:
        httpd.server_close()
    return 0
