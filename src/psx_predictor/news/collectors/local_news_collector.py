"""
local_news_collector.py
------------------------
RSS feed consumer targeting Pakistani financial news portals.
Sources:
  - Dawn News Business/Economy
  - Business Recorder
  - Profit by Pakistan Today
  - Express Tribune Business

Uses feedparser for RSS/Atom parsing (lightweight, no JS rendering needed).
Falls back gracefully if any individual feed is down.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from ..base import BaseCollector
from ..models import Article, CompanyMetadata

logger = logging.getLogger(__name__)

# Pakistani financial news RSS feeds
RSS_FEEDS = [
    {
        "name": "Dawn Business",
        "url": "https://www.dawn.com/feeds/business",
        "source": "Dawn News",
    },
    {
        "name": "Business Recorder",
        "url": "https://www.brecorder.com/feeds/latest-news",
        "source": "Business Recorder",
    },
    {
        "name": "Profit Pakistan Today",
        "url": "https://profit.pakistantoday.com.pk/feed/",
        "source": "Profit",
    },
    {
        "name": "Express Tribune Business",
        "url": "https://tribune.com.pk/feed/business",
        "source": "Express Tribune",
    },
]


class LocalNewsCollector(BaseCollector):
    """
    Collects news from Pakistani financial media RSS feeds.
    Fetches all articles from all feeds, then filters by company keywords.
    """

    @property
    def source_name(self) -> str:
        return "Local News RSS"

    def __init__(self, feeds: list[dict] | None = None):
        self.feeds = feeds or RSS_FEEDS

    def _parse_feed(self, feed_config: dict) -> List[Article]:
        """Parse a single RSS feed and return Article objects."""
        articles = []
        try:
            import feedparser
        except ImportError:
            logger.error(
                "feedparser not installed. Install with: pip install feedparser"
            )
            return articles

        feed_url = feed_config["url"]
        source = feed_config["source"]

        try:
            feed = feedparser.parse(feed_url)

            if feed.bozo and not feed.entries:
                logger.warning(f"Feed error for {feed_config['name']}: {feed.bozo_exception}")
                return articles

            for entry in feed.entries[:25]:  # Cap at 25 per feed per run
                # Parse published date
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        pass
                if published is None:
                    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        try:
                            published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                        except (TypeError, ValueError):
                            pass
                if published is None:
                    published = datetime.now(timezone.utc)

                # Extract summary/description
                summary = ""
                if hasattr(entry, "summary"):
                    # Strip HTML tags from RSS summary
                    import re
                    summary = re.sub(r"<[^>]+>", "", entry.summary or "").strip()
                elif hasattr(entry, "description"):
                    import re
                    summary = re.sub(r"<[^>]+>", "", entry.description or "").strip()

                article = Article(
                    headline=entry.get("title", "").strip(),
                    summary=summary[:500] if summary else None,
                    content=None,
                    url=entry.get("link", ""),
                    source=source,
                    published_at=published,
                    author=entry.get("author", None),
                    ticker=None,  # CompanyMatcher assigns this later
                )
                articles.append(article)

        except Exception as e:
            logger.error(f"Error parsing feed {feed_config['name']}: {e}")

        return articles

    def fetch_news(self, company: CompanyMetadata) -> List[Article]:
        """
        Fetches articles from all RSS feeds and returns those matching
        the company's name, aliases, or keywords.
        """
        all_articles = []

        for feed_config in self.feeds:
            raw_articles = self._parse_feed(feed_config)
            all_articles.extend(raw_articles)

        # Filter articles that mention this company
        import re
        matched = []
        search_terms = [company.name, company.ticker] + company.aliases + company.search_keywords
        patterns = [re.compile(r'\b' + re.escape(t.lower()) + r'\b') for t in search_terms]

        for article in all_articles:
            text = (
                (article.headline or "").lower() + " " +
                (article.summary or "").lower()
            )
            for pattern in patterns:
                if pattern.search(text):
                    matched.append(article)
                    break

        logger.info(
            f"[{self.source_name}] Collected {len(all_articles)} raw articles, "
            f"{len(matched)} matched for {company.ticker}"
        )
        return matched

    def fetch_all_unfiltered(self) -> List[Article]:
        """
        Fetches ALL articles from all feeds without company filtering.
        Useful for macro/political/geopolitical news classification where
        articles aren't tied to a specific ticker.
        """
        all_articles = []
        for feed_config in self.feeds:
            raw_articles = self._parse_feed(feed_config)
            all_articles.extend(raw_articles)
            logger.info(f"[{self.source_name}] {feed_config['name']}: {len(raw_articles)} articles")

        logger.info(f"[{self.source_name}] Total unfiltered: {len(all_articles)} articles")
        return all_articles
