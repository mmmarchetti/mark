"""Headlines via keyless RSS feeds (feedparser). Topic -> feed map, with
Portuguese (G1) and English (BBC/Reuters-style) sources.
"""

import logging

logger = logging.getLogger(__name__)

_FEEDS = {
    "pt": {
        "top": "https://g1.globo.com/rss/g1/",
        "tech": "https://g1.globo.com/rss/g1/tecnologia/",
        "world": "https://g1.globo.com/rss/g1/mundo/",
        "business": "https://g1.globo.com/rss/g1/economia/",
    },
    "en": {
        "top": "http://feeds.bbci.co.uk/news/rss.xml",
        "tech": "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "world": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "business": "http://feeds.bbci.co.uk/news/business/rss.xml",
    },
}


class News:
    def headlines(self, topic: str = "top", language: str = "pt", n: int = 4) -> str:
        import feedparser

        lang = "pt" if language == "pt" else "en"
        url = _FEEDS[lang].get((topic or "top").lower(), _FEEDS[lang]["top"])
        try:
            feed = feedparser.parse(url)
            items = feed.entries[:n]
            if not items:
                return "I couldn't get the news right now."
            titles = [e.get("title", "").strip() for e in items if e.get("title")]
            return "Top headlines: " + " ... ".join(titles) + "."
        except Exception:
            logger.exception("news fetch failed")
            return "I couldn't reach the news right now."
