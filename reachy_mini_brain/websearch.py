"""Web search (Dell-side, always available - no MacBook needed).

Keyless DuckDuckGo by default (the `ddgs` library); if REACHY_BRAVE_KEY is set,
uses the Brave Search API instead for higher quality/quota. Returns a short
plain-text summary of the top results for the brain to speak/summarize.
Mirrors weather.py: a provider class with graceful try/except fallback.
"""

import logging

import requests

from reachy_mini_brain import config

logger = logging.getLogger(__name__)


class WebSearch:
    def __init__(self) -> None:
        self.brave_key = config.BRAVE_KEY
        self.max_results = config.SEARCH_MAX_RESULTS

    def search(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "What should I search for?"
        try:
            results = self._brave(query) if self.brave_key else self._ddg(query)
        except Exception:
            logger.exception("Web search failed for %r", query)
            return "I couldn't run that search right now."
        if not results:
            return f"I didn't find anything useful for '{query}'."
        # Compact, speakable summary; the brain rephrases in the user's language.
        lines = [f"Top results for '{query}':"]
        for i, (title, snippet) in enumerate(results, 1):
            snippet = (snippet or "").strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            lines.append(f"{i}. {title}. {snippet}")
        return " ".join(lines)

    def _ddg(self, query: str) -> list[tuple[str, str]]:
        from ddgs import DDGS

        out = []
        for r in DDGS().text(query, max_results=self.max_results):
            out.append((r.get("title", ""), r.get("body", "")))
        return out

    def _brave(self, query: str) -> list[tuple[str, str]]:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.brave_key, "Accept": "application/json"},
            params={"q": query, "count": self.max_results},
            timeout=8,
        )
        resp.raise_for_status()
        web = (resp.json().get("web") or {}).get("results") or []
        return [(r.get("title", ""), r.get("description", "")) for r in web[: self.max_results]]
