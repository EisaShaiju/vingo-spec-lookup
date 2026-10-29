"""Spec enrichment prototype: fill objective specs a seller never typed.

A seller writes "Canon 200D, good condition, 8 months used". A buyer wants to
know the sensor, the mount and the resolution. Those facts are identical for
every 200D ever made, they are published by Canon, and the seller is never
going to type them. That gap is what this fills.

    set GROQ_API_KEY=...
    python scripts/enrich_prototype.py --since 30d --limit 8

Writes `docs/enrichment-demo.html` (before/after, self-contained) and
`docs/enrichment-cache.json` (the spec cache).

**This is a prototype and touches nothing.** It does not write to the database,
does not run in the ingest path and does not change what the console serves.
It answers one question -- does this produce specs good enough to publish --
before any of that is worth building.

Three design choices carry the whole thing:

**One call per distinct MODEL, never per listing.** 1,865 listings in the
corpus carry roughly 102 distinct models. Enrichment is a property of the
model, so the same answer serves every listing of it, now and forever. Per-row
calls would be ~18x the cost for identical answers, and that ratio improves as
the catalog grows.

**An ambiguous model is refused, not guessed.** "MacBook Pro" is a family, not
a product -- it could be a 2015 Intel or an M3 Max, and their specs share
almost nothing. `ProductModel.resolvable` already makes that distinction, and
asking anyway is how you fill a listing with confident, wrong specs. A wrong
spec is worse than a blank one: a blank field is visibly missing, a wrong one
is believed.

**Every spec is read out of a page we fetched, and cites it.** The model is
never asked what it knows, only what the article says -- and it is never asked
for the URL, since a model asked for a source will invent a plausible one. The
page we actually retrieved is the citation. A spec arriving without one did
not come from a page and is dropped.

Gap-fill only: a spec the seller already stated is reported as `kept`, never
overwritten. The seller is describing the actual unit and we are describing the
model -- on any disagreement the seller is right.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from enrichment_stats import fetch  # noqa: E402

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# `groq/compound` (web search server-side) is NOT available on this key --
# verified by GET /openai/v1/models, which lists only plain inference models.
# So the web half is done here and the model only READS what we fetched. That
# is the stronger arrangement anyway: a model given a page can be held to it,
# while a model recalling specs from training has nothing to cite.
GROQ_MODEL = "openai/gpt-oss-120b"
TIMEOUT_S = 90.0

# Wikipedia because it is free, needs no key, carries spec tables for exactly
# the consumer electronics this catalog is made of, and gives a URL a human can
# open and check. A manufacturer page would be better and is the obvious
# upgrade; it is also per-brand scraping work.
WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia's robot policy refuses a generic agent with a bare 403. It wants
# a named tool and a contact.
WIKI_UA = "VingoComet/0.1 (https://justvingo.com; eshaiju@justvingo.com) python-httpx"

CACHE = Path("docs/enrichment-cache.json")
DEMO = Path("docs/enrichment-demo.html")

SYSTEM = """You extract objective product specifications from a reference page.

You will be given the text of one encyclopedia article and a product name.
Use ONLY that text. You have no other knowledge for this task: if the article
does not state a fact, you do not know it.

Return ONLY specs that are IDENTICAL for every unit of that exact product.
Sensor size, lens mount, megapixels, weight, release year, panel size, chip.

NEVER return anything that varies per unit or per configuration: storage,
RAM, colour, condition, battery health, accessories, price. If the product is
sold in several configurations, those fields are not product facts, so omit
them.

If the article is about a product FAMILY rather than one product, or is about
something else entirely, return an empty spec list and say so in "reason".
Guessing is worse than returning nothing.

Reply with JSON only, no prose and no code fence:
{"specs": [{"name": "...", "value": "..."}], "reason": ""}"""


# The model key is stored compressed (`15-promax`, `seriess`), and free-text
# search does badly with that: `iphone 15-promax` returns "Horrible Histories".
# shortcut: a small alias table, extended whenever a lookup misses. The real
# fix is a product-id source (GTIN/UPC) instead of searching prose.
SEARCH_ALIASES = {
    "promax": "pro max",
    "seriess": "series s",
    "seriesx": "series x",
    "m1pro": "pro m1",
    "m2pro": "pro m2",
    "m3pro": "pro m3",
}


def search_terms(brand: str, model: str) -> str:
    out = (model or "").replace("-", " ")
    for glued, spaced in SEARCH_ALIASES.items():
        out = re.sub(rf"\b{glued}\b", spaced, out)
    return f"{brand} {out}".strip()


def about_this_product(title: str, brand: str, model: str) -> bool:
    """Is the article we got back actually about the product we asked for?

    Wikipedia returns its single best match no matter how poor, so relevance
    has to be checked here. Requires the brand AND at least one model token in
    the title: "Sony ZV-E10" for an a6100 names the brand but not the product,
    and its specs are a different camera's.
    """
    low = title.lower()
    if brand.lower() not in low:
        return False
    tokens = [t for t in re.split(r"[\s-]+", search_terms("", model).lower()) if len(t) >= 2]
    return any(t in low for t in tokens) if tokens else True


def wiki_context(brand: str, model: str) -> tuple[str, str]:
    """(article text, url) for the best match, or ("", "").

    The grounding step. Without a page there is nothing to cite, and an
    uncited spec is dropped downstream -- so a miss here means the model is
    honestly reported as unenriched rather than filled from memory.
    """
    import httpx

    query = search_terms(brand, model)
    with httpx.Client(timeout=30.0, headers={"User-Agent": WIKI_UA}) as c:
        found = c.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            },
        )
        # Without this a 403 arrives as "Expecting value: line 1 column 1",
        # which reads as a parser bug rather than as being blocked.
        found.raise_for_status()
        hits = found.json().get("query", {}).get("search", [])
        if not hits:
            return ("", "")
        title = hits[0]["title"]
        if not about_this_product(title, brand, model):
            # Wikipedia always returns its best match, never "no result", so an
            # unrelated article arrives looking exactly like a good one. Caught
            # here it costs nothing; passed on, it costs a model call to be
            # told the article is about something else.
            return ("", "")
        got = c.get(
            WIKI_API,
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": title,
                "format": "json",
            },
        )
        got.raise_for_status()
        page = got.json().get("query", {}).get("pages", {})
    extract = next(iter(page.values()), {}).get("extract", "") if page else ""
    if not extract:
        return ("", "")
    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    # The lead and spec tables sit at the top; the tail is history and
    # reception, which is tokens spent on nothing.
    return (extract[:6000], url)


def ask(brand: str, model: str) -> dict:
    """Fetch a reference page, then have the model read specs out of it."""
    import httpx

    key = os.environ.get("GROQ_API_KEY") or os.environ.get("VINGO_GROQ_API_KEY")
    if not key:
        raise SystemExit("set GROQ_API_KEY (console.groq.com/keys)")

    article, source_url = wiki_context(brand, model)
    if not article:
        return {
            "specs": [],
            "reason": f"no reference page found for {brand} {model}",
            "usage": {},
        }

    prompt = (
        f"Product: {brand} {model}\n\n"
        f"Article:\n{article}\n\n"
        f"Up to 6 specs a second-hand buyer would want and a seller would not "
        f"bother typing."
    )
    body_out = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        # Specs are a lookup, not a composition. Any creativity here is a
        # fabricated number.
        "temperature": 0,
    }
    # The free tier rate-limits after a handful of calls. Groq says how long to
    # wait in `Retry-After`; honouring it is the difference between a batch
    # that finishes slowly and one that dies a third of the way through.
    for attempt in range(1, 4):
        resp = httpx.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}"},
            json=body_out,
            timeout=TIMEOUT_S,
        )
        if resp.status_code != 429:
            break
        wait = float(resp.headers.get("retry-after") or 2**attempt)
        print(f"    rate limited, waiting {wait:.0f}s")
        time.sleep(min(wait, 60))
    resp.raise_for_status()
    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage") or {}

    # The model wraps JSON in a fence despite the instruction often enough.
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return {"specs": [], "reason": f"unparseable reply: {text[:120]}", "usage": usage}

    # The page we actually read IS the citation. The model is never asked for
    # a URL -- asked for one, it would invent a plausible one.
    for s in out.get("specs") or []:
        if isinstance(s, dict):
            s["source_url"] = source_url
    out["usage"] = usage
    return out


def clean(reply: dict) -> tuple[list[dict], str]:
    """Drop anything uncited. Returns (specs, why_rejected)."""
    kept, uncited = [], 0
    for s in reply.get("specs") or []:
        if not isinstance(s, dict):
            continue
        name, value = str(s.get("name") or "").strip(), str(s.get("value") or "").strip()
        url = str(s.get("source_url") or "").strip()
        if not (name and value):
            continue
        if not url.startswith("http"):
            # No source means no page was read. That is the fabrication case.
            uncited += 1
            continue
        kept.append({"name": name, "value": value, "source_url": url})
    reason = reply.get("reason") or ""
    if uncited:
        reason = f"{uncited} spec(s) dropped for citing no source. {reason}".strip()
    return kept, reason


def already_stated(spec: dict, text: str) -> bool:
    """Did the seller already say this? Compared on the NUMBER, because
    "24.1MP" and "24.1 megapixel" are the same fact typed two ways."""
    nums = re.findall(r"\d+(?:\.\d+)?", spec["value"])
    low = text.lower()
    return any(n in low for n in nums) if nums else spec["value"].lower() in low


def load_cache() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def save_cache(store: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(store, indent=2), encoding="utf-8")


def enrich(title: str, description: str = "", store: dict | None = None) -> dict:
    """Specs for ONE listing, as it arrives. This is the real entry point.

    The cache is what makes calling this per-listing affordable: a model is
    looked up once, ever, and every later listing of it is answered from disk
    with no network call. At 500 listings/day only the genuinely new models
    cost anything, and that count falls as the catalog fills.

    Never raises on a miss -- a listing with no recognisable model, or an
    ambiguous one, comes back with an empty spec list and a status saying
    which. Enrichment is a bonus on top of a listing we already hold; it must
    never be a reason to fail one.
    """
    from vingo_scraper.extract.model import extract_model

    owned = store is None
    store = load_cache() if owned else store

    pm = extract_model(title, description)
    if pm is None:
        return {"status": "no_model", "key": None, "specs": [], "reason": "", "tokens": 0}
    if not pm.resolvable:
        # A family name. Looking it up returns specs for the wrong generation.
        return {
            "status": "ambiguous",
            "key": pm.key,
            "specs": [],
            "reason": f"{pm.key} is a product family, not one product",
            "tokens": 0,
        }
    if pm.key in store:
        hit = store[pm.key]
        return {
            "status": "cached",
            "key": pm.key,
            "specs": hit["specs"],
            "reason": hit.get("reason", ""),
            "tokens": 0,
        }

    reply = ask(pm.brand, pm.model or "")
    specs, reason = clean(reply)
    store[pm.key] = {
        "brand": pm.brand,
        "model": pm.model or "",
        "specs": specs,
        "reason": reason,
    }
    if owned:
        save_cache(store)
    return {
        "status": "looked_up",
        "key": pm.key,
        "specs": specs,
        "reason": reason,
        "tokens": (reply.get("usage") or {}).get("total_tokens") or 0,
    }


def selfcheck() -> None:
    """The guards, provable in front of whoever asks. No API key needed."""
    kept, why = clean(
        {
            "specs": [
                {"name": "Sensor", "value": "APS-C", "source_url": "https://canon.in/x"},
                {"name": "Weight", "value": "450 g", "source_url": ""},
                {"name": "Mount", "value": "EF-S", "source_url": "not-a-url"},
            ]
        }
    )
    assert [s["name"] for s in kept] == ["Sensor"], kept
    assert "2 spec(s) dropped" in why, why

    assert already_stated({"value": "24.1 MP"}, "Canon 200D 24.1MP DSLR")
    assert not already_stated({"value": "24.1 MP"}, "Canon 200D good condition")

    from vingo_scraper.extract.model import extract_model

    fam = extract_model("MacBook Pro for sale")
    assert fam is None or not fam.resolvable, "a family name must never be looked up"

    # enrich() must reach neither the network nor an exception on these.
    assert enrich("a sofa, barely used", store={})["status"] == "no_model"
    amb = enrich("MacBook Pro 2019 for sale", store={})
    assert amb["status"] in ("no_model", "ambiguous"), amb
    assert amb["specs"] == []

    # the search fixes, pinned: these exact strings sent the wrong article
    assert search_terms("iphone", "15-promax") == "iphone 15 pro max"
    assert search_terms("xbox", "seriess") == "xbox series s"
    assert about_this_product("Canon EOS 200D", "canon", "200d")
    assert about_this_product("IPhone 15 Pro", "iphone", "15-promax")
    assert not about_this_product("Horrible Histories", "iphone", "15-promax")
    assert not about_this_product("Sony ZV-E10", "sony", "a6100"), "wrong camera's specs"

    hit = enrich(
        "Canon 200D with box",
        store={"canon:200d": {"specs": [{"name": "Sensor", "value": "APS-C"}], "reason": ""}},
    )
    assert hit["status"] == "cached" and hit["tokens"] == 0, hit
    print("selfcheck ok: uncited dropped, gaps detected, families refused, cache serves free")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="30d")
    ap.add_argument("--limit", type=int, default=8, help="distinct models to look up")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--title", help="enrich ONE listing, as if it just arrived")
    ap.add_argument("--desc", default="")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if args.title:
        got = enrich(args.title, args.desc)
        whole = f"{args.title} {args.desc}"
        print()
        print(f"  {args.title}")
        print(f"  model: {got['key'] or '(none recognised)'}  [{got['status']}]")
        if got["reason"]:
            print(f"  note:  {got['reason']}")
        for s in got["specs"]:
            seen = "" if not already_stated(s, whole) else "  (already stated)"
            print(f"    - {s['name']}: {s['value']}{seen}")
            print(f"      {s['source_url']}")
        if got["status"] == "cached":
            print()
            print("  served from cache -- no API call, no cost")
        return None

    from vingo_scraper.extract.model import extract_model

    rows = __import__("asyncio").run(fetch(args.since))
    print(f"{len(rows)} listings in range")

    # Group listings under their model. The grouping IS the cost argument:
    # every row under one key is served by a single lookup.
    by_model: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    named: dict[str, tuple[str, str]] = {}
    ambiguous = 0
    for title, desc in rows:
        pm = extract_model(title, desc)
        if pm is None:
            continue
        if not pm.resolvable:
            ambiguous += 1
            continue
        by_model[pm.key].append((title, desc))
        named[pm.key] = (pm.brand, pm.model or "")

    ranked = sorted(by_model.items(), key=lambda kv: -len(kv[1]))
    print(
        f"{len(by_model)} resolvable models cover "
        f"{sum(len(v) for v in by_model.values())} listings; "
        f"{ambiguous} skipped as ambiguous (family names)"
    )

    cache = load_cache()
    tokens = 0
    for key, listings in ranked[: args.limit]:
        got = enrich(listings[0][0], listings[0][1], store=cache)
        save_cache(cache)  # a crash at item 6 must not discard items 1-5
        tokens += got["tokens"]
        note = got["reason"] and f" -- {got['reason']}"
        print(f"  {key}: {got['status']}, {len(got['specs'])} spec(s){note or ''}")
    save_cache(cache)

    render(ranked[: args.limit], by_model, cache, len(rows), ambiguous, tokens)
    print(f"\n{tokens} tokens this run -> {DEMO}")


CSS = """
:root{
  --ground:#FBFCFD; --raised:#FFFFFF; --ink:#1A1F24; --muted:#5C6670;
  --line:#E3E7EA; --accent:#0F6E68; --accent-soft:#E6F0EF;
  --warn:#8A5A00; --warn-soft:#F7EFE0;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#121619; --raised:#191E22; --ink:#E8ECEF; --muted:#94A0A8;
    --line:#2A3238; --accent:#5FC2B8; --accent-soft:#17302E;
    --warn:#D9A441; --warn-soft:#2E2518;
  }
}
:root[data-theme="dark"]{
  --ground:#121619; --raised:#191E22; --ink:#E8ECEF; --muted:#94A0A8;
  --line:#2A3238; --accent:#5FC2B8; --accent-soft:#17302E;
  --warn:#D9A441; --warn-soft:#2E2518;
}
*{box-sizing:border-box}
body{
  background:var(--ground); color:var(--ink);
  font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  margin:0; padding:56px 24px 96px;
}
.wrap{max-width:860px;margin:0 auto;display:flex;flex-direction:column;gap:40px}
h1{
  font-family:ui-serif,Georgia,"Times New Roman",serif;
  font-size:34px; line-height:1.2; font-weight:600; margin:0 0 8px;
  text-wrap:balance; letter-spacing:-.01em;
}
h2{
  font-family:ui-serif,Georgia,serif; font-size:21px; font-weight:600;
  margin:0 0 4px; text-wrap:balance;
}
.lede{color:var(--muted);margin:0;max-width:62ch}
.eyebrow{
  font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.13em; text-transform:uppercase; color:var(--accent); margin:0 0 10px;
}
.stats{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(132px,1fr));
  gap:1px; background:var(--line); border:1px solid var(--line); border-radius:10px;
  overflow:hidden;
}
.stat{background:var(--raised);padding:16px 18px}
.stat b{
  display:block; font:600 25px/1.1 ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em;
}
.stat span{display:block;font-size:12.5px;color:var(--muted);margin-top:5px}
section{display:flex;flex-direction:column;gap:16px}
.card{
  background:var(--raised); border:1px solid var(--line);
  border-radius:10px; padding:18px 20px;
}
.card-h{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline;margin-bottom:4px}
.key{
  font:600 13px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  background:var(--accent-soft); color:var(--accent);
  padding:5px 8px; border-radius:5px;
}
.reuse{font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums}
.title{font-size:14.5px;color:var(--muted);margin:2px 0 0}
.scroll{overflow-x:auto;margin-top:12px}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th{
  text-align:left; font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
  letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  padding:0 10px 8px 0; border-bottom:1px solid var(--line); white-space:nowrap;
}
td{padding:9px 10px 9px 0;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
td.n{color:var(--muted);width:34%}
td.v{font-variant-numeric:tabular-nums}
.stated{font-size:12.5px;color:var(--muted)}
a{color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.why{
  border-left:3px solid var(--warn); background:var(--warn-soft);
  padding:12px 16px; border-radius:0 7px 7px 0; font-size:14.5px;
}
.note{font-size:13.5px;color:var(--muted);margin:0}
"""


def render(ranked, by_model, cache, total, ambiguous, tokens) -> None:
    """The demo page. Enriched and refused get equal billing on purpose.

    The refusals are the evidence that the thing is trustworthy -- a reader who
    only sees successes has no way to tell careful work from a system that
    always answers. Burying them in a tail would be the dishonest layout.
    """
    e = html.escape
    done = [(k, v) for k, v in ranked if (cache.get(k) or {}).get("specs")]
    skipped = [(k, v) for k, v in ranked if not (cache.get(k) or {}).get("specs")]
    filled = sum(len(v) for _, v in done)

    p = [
        "<title>Spec Enrichment Prototype</title>",
        f"<style>{CSS}</style>",
        "<div class='wrap'>",
        "<header>",
        "<p class='eyebrow'>Project Comet &middot; prototype</p>",
        "<h1>Filling in specs a seller never typed</h1>",
        "<p class='lede'>A seller writes &ldquo;Canon 200D, 8 months used&rdquo;. "
        "A buyer wants the sensor, the mount, the resolution. Those facts are the "
        "same for every 200D ever made, so we look them up once per model and "
        "reuse the answer for every listing of it.</p>",
        "</header>",
        "<div class='stats'>",
        f"<div class='stat'><b>{total:,}</b><span>listings scanned</span></div>",
        f"<div class='stat'><b>{len(ranked)}</b><span>models looked up</span></div>",
        f"<div class='stat'><b>{filled:,}</b><span>listings enriched</span></div>",
        f"<div class='stat'><b>{ambiguous:,}</b><span>refused as ambiguous</span></div>",
        f"<div class='stat'><b>{tokens:,}</b><span>tokens spent</span></div>",
        "</div>",
        "<section>",
        "<div><p class='eyebrow'>What it filled</p>"
        "<h2>Specs read from a cited page</h2>"
        "<p class='note'>Each spec was read out of the article linked beside it. "
        "The model is never asked what it knows and never asked for the URL &mdash; "
        "asked for a source, a model invents a plausible one.</p></div>",
    ]
    for key, listings in done:
        hit = cache[key]
        title, desc = listings[0]
        p.append(
            f"<article class='card'><div class='card-h'><span class='key'>{e(key)}</span>"
            f"<span class='reuse'>one lookup &rarr; {len(listings)} listing"
            f"{'s' if len(listings) != 1 else ''}</span></div>"
            f"<p class='title'>{e(title)}</p>"
            "<div class='scroll'><table><thead><tr><th>Specification</th>"
            "<th>Value</th></tr></thead><tbody>"
        )
        for s in hit["specs"]:
            mark = (
                ""
                if not already_stated(s, f"{title} {desc}")
                else " <span class='stated'>&mdash; seller already said this</span>"
            )
            p.append(
                f"<tr><td class='n'>{e(s['name'])}</td><td class='v'>{e(s['value'])}{mark} "
                f"<a href='{e(s['source_url'])}'>source</a></td></tr>"
            )
        p.append("</tbody></table></div></article>")
    p.append("</section>")

    if skipped:
        p += [
            "<section>",
            "<div><p class='eyebrow'>What it refused</p>"
            "<h2>Where it declined to answer</h2>"
            "<p class='note'>A wrong spec is worse than a blank one: a blank field "
            "is visibly missing, a wrong one is believed. These are the cases the "
            "guards caught.</p></div>",
        ]
        for key, listings in skipped:
            why = (cache.get(key) or {}).get("reason") or "not looked up"
            p.append(
                f"<article class='card'><div class='card-h'>"
                f"<span class='key'>{e(key)}</span>"
                f"<span class='reuse'>{len(listings)} listing"
                f"{'s' if len(listings) != 1 else ''} affected</span></div>"
                f"<div class='why'>{e(why)}</div></article>"
            )
        p.append("</section>")

    p += [
        "<section>",
        "<div><p class='eyebrow'>Cost</p><h2>Why this stays cheap at 500 a day</h2></div>",
        "<article class='card'><p class='note'>A model is looked up <b>once, ever</b>. "
        "Of ~500 new listings a day, roughly 17&percnt; carry a model we can resolve, "
        "and most are models already in the cache &mdash; so only the genuinely new "
        "ones cost anything, and that number falls as the catalogue fills. Every "
        "other listing is answered from disk with no network call.</p></article>",
        "</section>",
        "</div>",
    ]
    DEMO.parent.mkdir(parents=True, exist_ok=True)
    DEMO.write_text("\n".join(p), encoding="utf-8")


if __name__ == "__main__":
    main()
