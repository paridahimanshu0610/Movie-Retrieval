"""
intent_classifier.py — Classify query intent using TAMU AI chat (Claude Sonnet 4.5).

Recognises 13 query types and returns a QueryPlan dict used by the retriever:
  query_type     — one of the 13 types below
  intents        — which indices to search (scene/dialogue/character/plot/full)
  filters        — genre, year_min, year_max extracted from the query
  exclude_titles — movie titles to remove from results (negation queries)
  reference_title— movie the user cited for similarity/comparative queries
  rewrite        — improved search text when the original query is indirect

Falls back to a safe default plan on any API or parse error.
"""

import json
import logging
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

TAMU_AI_ENDPOINT = "https://chat-api.tamu.ai/api/v1/chat/completions"
TAMU_AI_MODEL    = "protected.Claude Sonnet 4.5"
VALID_INTENTS    = {"scene", "dialogue", "character", "plot", "full"}
VALID_TYPES      = {
    "dialogue", "simple_scene", "complex_scene", "detailed_scene",
    "event_journey", "plot_level", "similarity", "thematic",
    "thematic_scene", "filtered", "multi_criteria", "comparative", "negation",
}

_DEFAULT_PLAN = {
    "query_type":      "plot_level",
    "intents":         ["full"],
    "filters":         {"genre": None, "year_min": None, "year_max": None},
    "exclude_titles":  [],
    "reference_title": None,
    "rewrite":         None,
}

_SYSTEM_PROMPT = """\
You are a query planner for a movie screenplay retrieval system.

── QUERY TYPES ──────────────────────────────────────────────────────────────
dialogue       A remembered quote or line of speech
               e.g. "Someone says I know kung fu"
simple_scene   Brief visual/location description
               e.g. "Rooftop fight"
complex_scene  Multi-element scene with mood or context
               e.g. "Tense argument in the rain about a plan"
detailed_scene Very specific scene with character/prop details
               e.g. "Man in suit offers red pill and blue pill explaining a choice"
event_journey  A sequence or arc spanning multiple scenes
               e.g. "Training montage from beginner to master"
plot_level     Plot summary or story description (no named real person)
               e.g. "Hacker discovers reality is fake"
similarity     "Movies like X" — user names a reference film
               e.g. "Movies like The Matrix"
thematic       Abstract theme or concept, no specific scene
               e.g. "Movies about redemption"
thematic_scene Theme combined with a specific scene type
               e.g. "Scene about betrayal at a dinner table"
filtered       Content query with explicit metadata constraints (genre, era, decade, year)
               e.g. "90s action movie with a helicopter chase"
multi_criteria Multiple independent scene/event requirements joined by AND/OR
               e.g. "Car chase AND rooftop fight"
comparative    Like a reference film but with a stated contrast
               e.g. "Like Inception but less confusing"
negation       Explicitly excludes a known title
               e.g. "Heist movie but NOT Ocean's Eleven"

── SEARCH INDICES ───────────────────────────────────────────────────────────
scene     Scene headings + action lines  → visual, location, physical event queries
dialogue  Character speech               → quotes, remembered lines, conversations
character Dialogue grouped by character  → queries about what a fictional character says/does
plot      TMDB overview (one per movie)  → plot summaries, genre/theme, actor/director names
full      All fields combined            → broad or ambiguous queries

── INTENT MAPPING ───────────────────────────────────────────────────────────
dialogue       → ["dialogue"]
simple_scene   → ["scene"]
complex_scene  → ["scene", "full"]
detailed_scene → ["scene", "dialogue"]
event_journey  → ["scene", "full"]
plot_level     → ["plot"]
similarity     → ["plot", "full"]
thematic       → ["plot"]
thematic_scene → ["plot", "scene"]
filtered       → choose based on the non-filter content; always pair with "plot" if genre-only
multi_criteria → ["scene", "full"]
comparative    → ["plot", "full"]
negation       → choose based on the non-negation content

── SPECIAL RULES ────────────────────────────────────────────────────────────
• If the query mentions a real actor or director name → always include "plot"
• rewrite field:
    - similarity:    describe the reference film's themes/tone without naming it
    - comparative:   describe the desired characteristics without the reference name
    - negation:      the query content without the negation clause
    - filtered:      the query content without the metadata constraints
    - all others:    null (use the original query as-is)
• reference_title: the exact movie title named for similarity/comparative queries; null otherwise
• exclude_titles:  list of movie titles to exclude (negation queries); empty otherwise
• filters.genre:   genre string as named in TMDB (e.g. "Action", "Comedy"); null if not stated
• filters.year_min / year_max: integer years; convert "90s" → year_min=1990, year_max=1999

── OUTPUT FORMAT ────────────────────────────────────────────────────────────
Return ONLY a raw JSON object — no markdown, no code fences, no explanation.

{
  "query_type":      "<type>",
  "intents":         ["<intent>", ...],
  "filters":         {"genre": <str|null>, "year_min": <int|null>, "year_max": <int|null>},
  "exclude_titles":  ["<title>", ...],
  "reference_title": <str|null>,
  "rewrite":         <str|null>
}\
"""

_EXAMPLE_PAIRS = [
    (
        'Someone says "I know kung fu"',
        '{"query_type":"dialogue","intents":["dialogue"],"filters":{"genre":null,"year_min":null,"year_max":null},"exclude_titles":[],"reference_title":null,"rewrite":null}',
    ),
    (
        "Movies like The Matrix",
        '{"query_type":"similarity","intents":["plot","full"],"filters":{"genre":null,"year_min":null,"year_max":null},"exclude_titles":[],"reference_title":"The Matrix","rewrite":"science fiction dystopia simulated reality rebels artificial intelligence awakening"}',
    ),
    (
        "90s action movie with a helicopter chase",
        '{"query_type":"filtered","intents":["scene","plot"],"filters":{"genre":"Action","year_min":1990,"year_max":1999},"exclude_titles":[],"reference_title":null,"rewrite":"helicopter chase action"}',
    ),
    (
        "Heist movie but NOT Ocean's Eleven",
        '{"query_type":"negation","intents":["plot","full"],"filters":{"genre":null,"year_min":null,"year_max":null},"exclude_titles":["Ocean\'s Eleven"],"reference_title":null,"rewrite":"heist movie"}',
    ),
]


def _load_api_key(env_path: Path | None) -> str | None:
    if env_path and env_path.exists():
        load_dotenv(env_path, override=False)
    return os.environ.get("TAMU_AI_CHAT_API_KEY")


def _strip_fences(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    return text


def _validate_plan(raw: dict) -> dict:
    """Coerce and validate a raw LLM-returned dict into a well-formed QueryPlan."""
    plan = dict(_DEFAULT_PLAN)
    plan["filters"] = dict(_DEFAULT_PLAN["filters"])

    if raw.get("query_type") in VALID_TYPES:
        plan["query_type"] = raw["query_type"]

    intents = [i for i in (raw.get("intents") or []) if i in VALID_INTENTS]
    plan["intents"] = intents if intents else ["full"]

    f = raw.get("filters") or {}
    plan["filters"]["genre"]    = f.get("genre") or None
    plan["filters"]["year_min"] = int(f["year_min"]) if f.get("year_min") else None
    plan["filters"]["year_max"] = int(f["year_max"]) if f.get("year_max") else None

    plan["exclude_titles"]  = [str(t) for t in (raw.get("exclude_titles") or [])]
    plan["reference_title"] = raw.get("reference_title") or None
    plan["rewrite"]         = raw.get("rewrite") or None

    return plan


def classify_query(query: str, env_path: Path | None = None) -> dict:
    """
    Classify a query and return a full QueryPlan dict.

    Args:
        query:    Natural-language query.
        env_path: Path to .env file with TAMU_AI_CHAT_API_KEY.

    Returns:
        QueryPlan dict (always valid; falls back to _DEFAULT_PLAN on error).
    """
    api_key = _load_api_key(env_path)
    if not api_key:
        logger.warning("TAMU_AI_CHAT_API_KEY not set — using default plan")
        return dict(_DEFAULT_PLAN)

    # Build few-shot messages
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for user_ex, assistant_ex in _EXAMPLE_PAIRS:
        messages.append({"role": "user",      "content": user_ex})
        messages.append({"role": "assistant", "content": assistant_ex})
    messages.append({"role": "user", "content": query})

    try:
        response = requests.post(
            TAMU_AI_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model":       TAMU_AI_MODEL,
                "stream":      False,
                "messages":    messages,
                "max_tokens":  256,
                "temperature": 0,
            },
            timeout=20,
        )
        response.raise_for_status()
        content = _strip_fences(
            response.json()["choices"][0]["message"]["content"].strip()
        )
        raw = json.loads(content)
        plan = _validate_plan(raw)
        plan["_raw_response"] = content  # pass through for display in main.py
        print(f"[intent] type={plan['query_type']}  intents={plan['intents']}"
              + (f"  genre={plan['filters']['genre']}" if plan['filters'].get('genre') else "")
              + (f"  years={plan['filters']['year_min']}-{plan['filters']['year_max']}" if plan['filters'].get('year_min') or plan['filters'].get('year_max') else "")
              + (f"  ref={plan['reference_title']!r}" if plan.get('reference_title') else "")
              + (f"  exclude={plan['exclude_titles']}" if plan.get('exclude_titles') else "")
              + (f"  rewrite={plan['rewrite']!r}" if plan.get('rewrite') else ""))
        return plan

    except requests.RequestException as exc:
        logger.warning("TAMU AI request failed: %s — using default plan", exc)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse QueryPlan: %s — using default plan", exc)

    return dict(_DEFAULT_PLAN)


def classify_intent(query: str, env_path: Path | None = None) -> list[str]:
    """Backward-compatible wrapper — returns just the intents list."""
    return classify_query(query, env_path=env_path)["intents"]


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.WARNING)

    ap = argparse.ArgumentParser(description="Classify query via TAMU AI.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--env",   default="../.env")
    args = ap.parse_args()

    plan = classify_query(args.query, env_path=Path(args.env))
    print(f"Query      : {args.query}")
    print(f"Type       : {plan['query_type']}")
    print(f"Intents    : {plan['intents']}")
    print(f"Filters    : {plan['filters']}")
    print(f"Ref title  : {plan['reference_title']}")
    print(f"Exclude    : {plan['exclude_titles']}")
    print(f"Rewrite    : {plan['rewrite']}")
