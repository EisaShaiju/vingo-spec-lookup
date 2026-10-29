# Spec Lookup — experimental

Type a product name or paste a listing title; a Groq-hosted model searches the
web, reads the pages itself, and returns the objective specs plus how the
product is regarded.

**Experimental and untracked.** Nothing here imports from or touches the
scraper, the console or the database.

## Run it locally

```bash
pip install -r demo/requirements.txt
export GROQ_API_KEY=gsk_...          # PowerShell: $env:GROQ_API_KEY="gsk_..."
streamlit run demo/streamlit_app.py
```

## Deploy so the team can test it

Streamlit Community Cloud is free and needs no server.

1. Push **`streamlit_app.py`** and **`requirements.txt`** to a small repo of
   their own. Do not deploy from the scraper repo — the host would get a copy
   of everything in it, and this app needs none of it.
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the
   repo → main file `streamlit_app.py`.
3. **Settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
   Secrets live in Streamlit's own store, never in the repo.
4. Deploy. You get a public `*.streamlit.app` URL.

Anyone with the link can use it, and every lookup spends your Groq quota — so
treat the URL as semi-private until you are happy with the spend.

## Optional: better search

With no `BRAVE_API_KEY` the app searches **Wikipedia**, which covers cameras
and consoles well and misses plenty else — `Canon EOS 1500D` returns nothing,
because Wikipedia files it under a different name.

Add a free [Brave Search](https://brave.com/search/api/) key (2,000 queries a
month) as a second secret and the model can reach manufacturer spec pages and
review sites instead:

```toml
BRAVE_API_KEY = "BSA..."
```

DuckDuckGo and the public SearXNG instances were both tried and are not usable
unattended — DDG serves a bot-check page, the SearXNG mirrors answered 429/403.

## What it does, and what it refuses

- **One lookup per product**, cached. At 500 new listings a day only genuinely
  new products cost anything.
- **A category is refused, not guessed.** "DSLR camera" returns suggestions
  rather than one model's specs.
- **A cited source is one the model actually opened.** Every page goes through
  this app, so a URL the model invented is not in the fetched set and the
  citation is dropped — it shows as *not verified* rather than as a link.
- **Objective and subjective are kept apart**, because "24.2 MP" and "good for
  beginners" are not the same kind of claim.

## Known limits

- The free Groq tier allows **8,000 tokens per minute**. The app trims old page
  text out of the transcript and waits out a rate limit, but a cold lookup can
  take 20–30 seconds.
- Research is capped at 4 tool round-trips; if it has read a page by then it
  answers from it, otherwise it returns nothing rather than guessing.
