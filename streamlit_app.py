"""Spec lookup: what the seller didn't tell you.

EXPERIMENTAL. Untracked, deployed separately, nothing here reaches the scraper
or the console.

Type a product name or paste a listing title. A Groq-hosted model is given
real internet access through TOOL CALLING -- it decides what to search, which
page to open, and when it has enough -- then returns the objective specs plus
how the product is regarded.

    streamlit run demo/streamlit_app.py

**Why tool calling rather than Groq's own web search.** `groq/compound` has
server-side search built in but is NOT available on this key (verified: GET
/openai/v1/models lists only plain inference models). So the loop lives here:
the model asks for `web_search` or `read_page`, this app executes it, and the
result goes back as another message. Same capability, and it has one property
compound could not give us -- every page the model reads passed through this
process, so **a cited URL can be checked against what was actually fetched**.
A source the model invented is simply not in that set and is dropped.

**Self-contained on purpose.** Imports nothing from `vingo_scraper`, so it can
deploy from a small separate repo rather than handing a host the whole scraper.

Two kinds of claim, kept apart because they are not the same kind of thing:

  * **objective** specs come from pages the model read, and cite them
  * **subjective** reputation is the model's own view, labelled as opinion

Mixing them would let "24.2 MP" and "good for beginners" carry equal weight.
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
import time

import httpx
import streamlit as st

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_ROUNDS = 4  # tool round-trips before we stop and take what we have

WIKI_API = "https://en.wikipedia.org/w/api.php"
BRAVE_API = "https://api.search.brave.com/res/v1/web/search"
# Wikimedia's robot policy refuses a generic agent with a bare 403: it wants a
# named tool and a contact. Several other sites refuse a non-browser agent too.
UA = "VingoComet/0.1 (https://justvingo.com; eshaiju@justvingo.com) python-httpx"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

TOOL_INSTRUCTION = """You have web_search and read_page. USE THEM -- never answer from memory for
the objective specs. Search, open the most authoritative page you find
(manufacturer first, then a reputable review site or encyclopedia), read it,
and take the numbers from there. Two or three tool calls is normal."""

NO_TOOLS = """You have no tools on this turn. Answer from the pages quoted in the message
below, and from nothing else."""

# Concatenated, not an f-string: the prompt contains a JSON skeleton, and
# f-string formatting reads its braces as format fields.
SYSTEM = (
    "You research second-hand marketplace products for buyers in India.\n\n"
    + TOOL_INSTRUCTION
    + """

Research the MAIN product only. Ignore bundled accessories -- a kit lens,
case or memory card in the title is not what you are describing. Do not look
up the same page twice; you have a small budget of calls.

FIRST decide whether the text names ONE product or a CATEGORY. "DSLR camera",
"MacBook Pro", "gaming console" are categories: their specs differ across
every model in them, so do not guess one.

Then reply with JSON only -- no prose, no code fence:

{"product":"", "ambiguous":false, "suggestions":[],
 "objective":[{"name":"","value":"","source_url":""}],
 "subjective":[{"point":"","kind":"strength|weakness|suits"}],
 "note":""}

objective: measurable facts IDENTICAL for every unit -- sensor, mount,
megapixels, weight, release year, screen, processor. NEVER per-unit or
per-configuration values (storage, RAM, colour, condition, battery, price).
source_url MUST be a page you actually opened with read_page.
If the product comes in several configurations, omit the fields that differ.

subjective: how the product is REGARDED -- strengths, weaknesses, who it
suits. This is your own view from reviews, opinion rather than measurement.

ambiguous: set true for a category, give no specs, and put up to 5 specific
popular models in suggestions.

If you cannot find a reliable page, return no objective specs and say why in
note. Guessing is worse than returning nothing."""
)


def api_key() -> str:
    """Streamlit Cloud puts secrets in st.secrets; locally it is an env var."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"])
    except Exception:  # noqa: BLE001 -- no secrets.toml locally is normal
        pass
    return os.environ.get("GROQ_API_KEY", "")


def brave_key() -> str:
    try:
        if "BRAVE_API_KEY" in st.secrets:
            return str(st.secrets["BRAVE_API_KEY"])
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("BRAVE_API_KEY", "")


# --------------------------------------------------------------------------
# the tools the model can call
# --------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Returns titles, URLs and snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Fetch a URL and return its text. Use this before quoting any spec.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


def web_search(query: str) -> list[dict]:
    """Brave when a key is set, Wikipedia otherwise.

    DuckDuckGo and the public SearXNG instances were both tried and are not
    usable unattended -- DDG serves a bot-check page and the SearXNG mirrors
    answered 429/403. Wikipedia's API has no key and no rate problem, so it is
    the floor; Brave's free tier (2k/month) is the upgrade that makes
    manufacturer pages reachable.
    """
    bk = brave_key()
    if bk:
        r = httpx.get(
            BRAVE_API,
            params={"q": query, "count": 5},
            headers={"X-Subscription-Token": bk, "Accept": "application/json"},
            timeout=25.0,
        )
        r.raise_for_status()
        return [
            {
                "title": w.get("title", ""),
                "url": w.get("url", ""),
                "snippet": w.get("description", ""),
            }
            for w in r.json().get("web", {}).get("results", [])[:5]
        ]

    r = httpx.get(
        WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 5,
            "format": "json",
        },
        headers={"User-Agent": UA},
        timeout=25.0,
    )
    r.raise_for_status()
    return [
        {
            "title": h["title"],
            "url": "https://en.wikipedia.org/wiki/" + h["title"].replace(" ", "_"),
            "snippet": re.sub("<[^>]+>", "", h.get("snippet", "")),
        }
        for h in r.json().get("query", {}).get("search", [])
    ]


def read_page(url: str) -> str:
    """Fetch a page as text. Wikipedia needs our named agent; most other sites
    refuse one and want a browser agent, so try both."""
    last = ""
    for agent in (UA if "wikipedia.org" in url else BROWSER_UA, UA):
        try:
            r = httpx.get(url, headers={"User-Agent": agent}, timeout=25.0, follow_redirects=True)
            r.raise_for_status()
            text = re.sub(r"(?is)<(script|style|nav|footer)[^>]*>.*?</\1>", " ", r.text)
            text = html_mod.unescape(re.sub(r"<[^>]+>", " ", text))
            text = re.sub(r"[ \t\xa0]+", " ", text)
            text = re.sub(r"\n\s*\n+", "\n", text).strip()
            # Enough for a spec table; the rest is navigation and reviews.
            # Kept small because the free tier caps TOKENS PER MINUTE at
            # 8k and page text accumulates in the conversation.
            return text[:2500] or "(page was empty)"
        except Exception as exc:  # noqa: BLE001 -- reported to the model, not raised
            last = f"{type(exc).__name__}: {exc}"
    return f"(could not read this page: {last})"


# --------------------------------------------------------------------------
# the agent loop
# --------------------------------------------------------------------------


def groq_call(messages: list[dict], key: str, *, tools: bool = True) -> dict:
    """One chat turn. Retries only a rate limit, honouring Retry-After."""
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        # Omitted only when the history is free of tool calls -- see the
        # wrap-up below.
        **({"tools": TOOLS, "tool_choice": "auto"} if tools else {}),
        # A lookup, not a composition. Creativity here is a fabricated number.
        "temperature": 0,
    }
    for attempt in range(1, 6):
        r = httpx.post(
            GROQ_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=120.0
        )
        if r.status_code != 429:
            break
        # The body states the wait when the header does not. A tokens-per-
        # minute limit clears on a rolling window, so waiting is the fix.
        wait = r.headers.get("retry-after")
        if wait is None:
            found = re.search(r"try again in ([\d.]+)s", r.text)
            wait = found.group(1) if found else 2**attempt
        time.sleep(min(float(wait) + 0.5, 30))
    if r.status_code >= 400:
        # Carry the API's own words. A bare "400 Bad Request" sends you
        # guessing at the payload; the body names the field.
        try:
            why = r.json()["error"]["message"]
        except Exception:  # noqa: BLE001
            why = r.text[:300]
        raise RuntimeError(f"Groq {r.status_code}: {why}")
    return r.json()["choices"][0]["message"]


def research(text: str, trace: list) -> dict:
    """Let the model search and read until it answers. Returns parsed JSON.

    `trace` is filled with what it did, so the page can show the working --
    which is the difference between "trust me" and evidence.
    """
    key = api_key()
    if not key:
        raise RuntimeError("No GROQ_API_KEY. Add it under Settings -> Secrets.")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]
    fetched: set[str] = set()
    pages: dict[str, str] = {}

    for rounds_left in range(MAX_ROUNDS, 0, -1):
        # Only the two most recent pages stay in full. Without this the
        # transcript grows past the 8k tokens-per-minute cap by round three
        # and every later call is rejected -- which reads as a broken app
        # rather than as a quota.
        bulky = [i for i, m in enumerate(messages) if m.get("role") == "tool"][:-1]
        for i in bulky:
            body = messages[i]["content"]
            if len(body) > 150:
                messages[i] = dict(messages[i], content=body[:150] + " …[trimmed]")
        searches = sum(1 for k, _, _ in trace if k == "search")
        if searches >= 2 and not pages:
            # It searched repeatedly and opened nothing, then ran out of turns
            # with no page to answer from. Snippets are not a source.
            messages.append(
                {
                    "role": "user",
                    "content": "Stop searching. Open the most relevant URL you have "
                    "already seen with read_page.",
                }
            )
        if rounds_left == 2:
            # Tell it to wrap up while it still has a turn to do so. Cutting
            # the tools off instead fails: once the history holds tool calls,
            # Groq rejects the request with "Tool choice is none, but model
            # called a tool" -- an opaque 400 that looks like a payload bug.
            messages.append(
                {
                    "role": "user",
                    "content": "No more research steps. Reply now with the JSON, "
                    "using only what you have already read.",
                }
            )
        msg = groq_call(messages, key)
        calls = msg.get("tool_calls") or []
        if not calls:
            raw = re.sub(r"^```(?:json)?|```$", "", (msg.get("content") or "").strip(), flags=re.M)
            try:
                out = json.loads(raw.strip())
            except json.JSONDecodeError:
                return {"ambiguous": False, "objective": [], "subjective": [], "note": raw[:300]}
            # THE guard: a source the model did not actually open is dropped.
            # Presence of a URL is not evidence it was read.
            for spec in out.get("objective") or []:
                if spec.get("source_url") not in fetched:
                    spec["source_url"] = ""
            return out

        # content must be null, not "", on a tool-call turn.
        messages.append(
            {"role": "assistant", "content": msg.get("content") or None, "tool_calls": calls}
        )
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if name == "web_search":
                q = args.get("query", "")
                hits = web_search(q)
                trace.append(("search", q, f"{len(hits)} results"))
                result = json.dumps(hits)
            elif name == "read_page":
                url = args.get("url", "")
                if url in pages:
                    # It re-read the same article and ran out of rounds doing
                    # it. Serving the copy keeps the budget for new ground.
                    body, ok = pages[url], True
                    trace.append(("read", url, "already read, served from memory"))
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": body})
                    continue
                body = read_page(url)
                ok = not body.startswith("(could not read")
                if ok:
                    pages[url] = body
                if ok:
                    fetched.add(url)
                trace.append(("read", url, f"{len(body)} chars" if ok else "failed"))
                result = body
            else:
                result = "(unknown tool)"
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    # Out of research steps. Rather than throw away pages already fetched,
    # ask once more with a CLEAN history: no tool calls in it, so the tools
    # can be withheld without Groq rejecting the request, and the model has
    # no option but to answer.
    if pages:
        digest = "\n\n".join(f"PAGE {u}\n{body[:1400]}" for u, body in list(pages.items())[:2])
        wrap_up = (
            f"{text}\n\nPages already read:\n{digest}\n\n"
            "Reply with the JSON now, using only these pages."
        )
        # NOT the research prompt: that one says "you have web_search and
        # read_page, USE THEM", so the model calls a tool that is no longer
        # offered and Groq rejects the request with "Tool choice is none, but
        # model called a tool".
        msg = groq_call(
            [
                {"role": "system", "content": SYSTEM.replace(TOOL_INSTRUCTION, NO_TOOLS)},
                {"role": "user", "content": wrap_up},
            ],
            key,
            tools=False,
        )
        raw = re.sub(r"^```(?:json)?|```$", "", (msg.get("content") or "").strip(), flags=re.M)
        try:
            out = json.loads(raw.strip())
        except json.JSONDecodeError:
            out = None
        if out:
            for spec in out.get("objective") or []:
                if spec.get("source_url") not in fetched:
                    spec["source_url"] = ""
            trace.append(("wrap up", "answered from pages already read", "no more searching"))
            return out

    return {
        "ambiguous": False,
        "objective": [],
        "subjective": [],
        "note": "Could not settle on an answer within the research budget.",
    }


@st.cache_data(show_spinner=False)
def look_up(text: str) -> tuple[dict, list]:
    """Cached: a product is researched once, and every later listing of it is
    answered with no calls at all. That is the cost argument."""
    trace: list = []
    return research(text, trace), trace


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Spec Lookup", page_icon="🔎", layout="centered")

st.title("What the seller didn't tell you")
st.caption(
    "Paste a listing title, or just name a product. The model searches the web "
    "and reads the pages itself — objective specs cite the page they came "
    "from; reputation is labelled as opinion."
)

if "q" not in st.session_state:
    st.session_state.q = ""

st.write("**Try:**")
cols = st.columns(3)
for col, (label, value) in zip(
    cols,
    [
        ("DSLR camera", "DSLR camera"),
        ("Canon EOS 200D", "Canon EOS 200D"),
        ("a real listing", "Nikon D5200 with 18-55mm lens, 2 years old, good condition"),
    ],
    strict=True,
):
    if col.button(label, use_container_width=True):
        st.session_state.q = value

query = st.text_input(
    "Product or listing title",
    value=st.session_state.q,
    placeholder="Canon EOS 200D — or paste a whole listing title",
)

if query:
    try:
        with st.spinner("Searching and reading pages…"):
            r, trace = look_up(query)
    except Exception as exc:  # noqa: BLE001 -- surfaced, never swallowed
        st.error(f"That lookup failed: {exc}")
        st.stop()

    if r.get("ambiguous"):
        st.warning(
            "That names a category, not one product — specs differ across every "
            "model in it. Guessing one would fill the listing with facts about "
            "a different product."
        )
        st.write("**Did you mean:**")
        for i, s in enumerate(r.get("suggestions") or []):
            if st.button(s, key=f"s{i}"):
                st.session_state.q = s
                st.rerun()
    else:
        if r.get("product"):
            st.subheader(r["product"])

        objective = r.get("objective") or []
        if objective:
            st.markdown("#### Objective specifications")
            low = query.lower()
            st.table(
                {
                    "Specification": [o.get("name", "") for o in objective],
                    "Value": [o.get("value", "") for o in objective],
                    "In the listing?": [
                        "already stated"
                        if any(
                            n in low for n in re.findall(r"\d+(?:\.\d+)?", str(o.get("value", "")))
                        )
                        else "filled in"
                        for o in objective
                    ],
                    "Source": [o.get("source_url") or "— not verified" for o in objective],
                }
            )
            st.caption(
                "A source is shown only if the model actually opened that page "
                "during this lookup. Anything it named without reading is "
                "marked unverified rather than linked."
            )
        elif r.get("note"):
            st.info(r["note"])

        subjective = r.get("subjective") or []
        if subjective:
            st.markdown("#### How it's regarded")
            st.caption(
                "Reputation and review consensus — opinion, not measurement, "
                "and kept separate from the facts above for that reason."
            )
            for s in subjective:
                st.markdown(f"**{str(s.get('kind', 'note')).title()}** · {s.get('point', '')}")

    if trace:
        with st.expander(f"What it did — {len(trace)} steps"):
            for kind, what, outcome in trace:
                st.markdown(
                    f"**{kind}** · `{what}` → {outcome}"
                    if kind == "search"
                    else f"**{kind}** · [{what}]({what}) → {outcome}"
                )

SEARCH_LABEL = (
    "Brave" if brave_key() else "Wikipedia (add a free BRAVE_API_KEY for broader results)"
)

st.divider()
st.caption(
    f"Model: `{GROQ_MODEL}` on Groq, with web access through tool calling — it "
    "chooses what to search and which pages to open, and this app executes "
    f"those calls. Search is {SEARCH_LABEL}. "
    "One lookup per product, cached: at 500 new listings a day only genuinely "
    "new products cost anything."
)
