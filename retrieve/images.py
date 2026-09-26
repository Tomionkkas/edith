"""Which picture belongs to a record, and where the browser gets it.

The join is the product. The image is keyed on the record that ANSWERED - not
on the words of the question - so `who is spider-man 2099` shows Miguel
O'Hara and `who is wolverine` shows James Howlett rather than whichever
namesake happens to be largest. `doc_id` in and a URL out.

**Nothing here downloads anything.** `urls_for()` returns URLs for the
BROWSER to fetch: the wiki serves the bytes and this app serves a string.

**The URL is the CDN's, not `Special:FilePath`, and that is not a
preference.** Measured in a real browser 2026-09-26: a page on
`http://127.0.0.1:8420` asking `marvel.fandom.com/wiki/Special:FilePath/...`
gets nothing - the request fails even with `referrer=no-referrer` - while the
same file from `static.wikia.nocookie.net` loads at full size. The wiki host
refuses browser-initiated hotlinks; the CDN serves them. curl gets both,
which is exactly why this was invisible until the page ran.

**A Referer is refused by both.** With `Referer: http://127.0.0.1:8420/` the
CDN returns a 1,976-byte 404 placeholder; with none it returns the image.
The page therefore sets `<meta name="referrer" content="no-referrer">`, and
20 of 20 sampled files served under that rule.

A server-side prefetch remains impossible anyway: Fandom refused `urllib` 40
times out of 40 - bare User-Agent, Accept headers, full browser UA string -
while curl fetched 39 of the same 40. They fingerprint the client. Hotlinking
asks no such question, and it is the weaker claim on image licensing, which
ROADMAP's licensing note leaves open for art.

Coverage, measured over all 202,207 records:

    the record's own art                       90.84%
    the cover of its `First appearance`        + 4.37%
    nothing - the page draws a plate             4.79%
"""
import gzip
import hashlib
import importlib.util
import json
import pathlib
import sys
import urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
IMAGES_PATH = HERE / "images.json.gz"
CDN = "https://static.wikia.nocookie.net/marveldatabase/images"


def load(path: pathlib.Path = IMAGES_PATH) -> dict:
    """The sidecar, or `{}` when it was not shipped.

    Missing is not fatal: the app runs with no art rather than refusing to
    start, because the sidecar is a garnish on an engine that answers fine
    without it.
    """
    if not path.exists():
        return {}
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def _resolve():
    """`retrieve/resolve.py`, loaded the way every caller in this repo loads
    it - by path, so `train/` is never forced to import `retrieve/`."""
    if "resolve" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "resolve", HERE / "resolve.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["resolve"] = module
        spec.loader.exec_module(module)
    return sys.modules["resolve"]


def title_of(record: str, headline: str) -> str:
    """The sidecar key for a record: its `Page:` line, else its title.

    THE SAME RULE `resolve.build()` USES, and it must stay the same rule.
    Issue records carry no `Page:` line, so keying on it alone drops the
    whole comics tier without erroring - measured at 55.98% coverage before
    `untitled()` was added here, 95.21% after. This is the `Kind:`
    triplication hazard wearing a different hat.
    """
    resolve = _resolve()
    return resolve.page_title(record) or resolve.untitled(headline)


def _field(record: str, label: str) -> str:
    """The first entry of a field. Fields are joined with "; " corpus-wide."""
    for line in record.split("\n"):
        if line.startswith(label + ":"):
            return line[len(label) + 1:].split(";")[0].strip()
    return ""


def url_for_name(name: str) -> str:
    """The CDN path for a file, derived rather than looked up.

    MediaWiki stores an upload under the first one and first two characters
    of the md5 of its underscored name - `Wolverine Vol 8 22 Virgin
    Variant.jpg` hashes to `bb...`, so it lives at `b/bb/`. Verified against
    what `Special:FilePath` redirects to, then checked on 20 random files:
    20/20 served. Deriving it skips three redirect hops AND the wiki host,
    which is the one that refuses browsers.
    """
    underscored = name.replace(" ", "_")
    digest = hashlib.md5(underscored.encode("utf-8")).hexdigest()
    return (f"{CDN}/{digest[0]}/{digest[:2]}/"
            f"{urllib.parse.quote(underscored)}/revision/latest")


def urls_for(record: str, headline: str, images: dict) -> list:
    """The image chain for a record, most specific first.

    Both rungs are returned even when the first resolves: the browser walks
    the chain on `onerror`, which is what covers a file the wiki serves under
    a different name than the record gives. An empty list means the page
    should draw its plate - about one record in twenty-one.
    """
    chain = []
    own = images.get(title_of(record, headline))
    if own:
        chain.append(url_for_name(own))
    first = _field(record, "First appearance")
    cover = images.get(first) if first else ""
    if cover:
        chain.append(url_for_name(cover))
    return chain
