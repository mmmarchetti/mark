"""Wikipedia quick-facts lookup (keyless REST summary API)."""

import logging

import requests

logger = logging.getLogger(__name__)

# Wikipedia's APIs require a descriptive User-Agent or they return 403/HTML.
_HEADERS = {"User-Agent": "ReachyMiniMark/1.0 (personal robot assistant)"}


class Wiki:
    def lookup(self, query: str, lang: str = "en") -> str:
        query = (query or "").strip()
        if not query:
            return "What should I look up?"
        wiki_lang = "pt" if lang == "pt" else "en"
        try:
            # Search for the best-matching page title first.
            s = requests.get(
                f"https://{wiki_lang}.wikipedia.org/w/rest.php/v1/search/title",
                params={"q": query, "limit": 1}, headers=_HEADERS, timeout=6,
            ).json()
            pages = s.get("pages") or []
            if not pages:
                return f"I couldn't find a Wikipedia article about {query}."
            title = pages[0]["title"]
            r = requests.get(
                f"https://{wiki_lang}.wikipedia.org/api/rest_v1/page/summary/{title}",
                headers=_HEADERS, timeout=6,
            ).json()
            extract = (r.get("extract") or "").strip()
            return extract or f"I found {title} but no summary."
        except Exception:
            logger.exception("wiki lookup failed")
            return "I couldn't reach Wikipedia right now."
