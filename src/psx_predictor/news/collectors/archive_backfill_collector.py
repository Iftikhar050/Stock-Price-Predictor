"""
archive_backfill_collector.py
----------------------------------
Historical backfill collector for news archives.

Scrapes date-indexed archives of Pakistani financial portals:
- Dawn (dawn.com) -> source="dawn_archive_backfill"
- Business Recorder (brecorder.com) -> source="brecorder_archive_backfill"
- Profit (profit.pakistantoday.com.pk) -> source="profit_archive_backfill"

Tags all extracted articles with source="*_archive_backfill" so downstream feature-quality
checks can distinguish historical coverage from live coverage via sentiment_coverage_era.

NOTE: this used to also return ~40 hand-written "historical" headlines dated
2018-2026, each given a fabricated URL built from a Python hash() of the
headline text (i.e. a URL that does not resolve to any real article). That
literal dataset has been removed rather than kept as a labeled synthetic
source: real historical news depth before ~2023 doesn't exist anywhere in
this pipeline (corporate_announcements_pucars/corporate_events are genuinely
only available from 2023-06 onward for the same reason - PSX's own
disclosure feed doesn't go back further), and there's no way to honestly
backfill it. This collector now only returns what _scrape_live_archives can
actually find on today's category pages.
"""

import os
import sys
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.psx_predictor.news.base import BaseCollector
from src.psx_predictor.news.models import Article, CompanyMetadata

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

class ArchiveBackfillCollector(BaseCollector):
    """
    Historical backfill collector that scrapes and formats date-indexed historical news
    archives from Dawn, Business Recorder, and Profit.
    """

    @property
    def source_name(self) -> str:
        return "Archive Backfill Collector"

    def fetch_news(self, company: CompanyMetadata) -> List[Article]:
        """
        Fetches backfill news articles for a given company by scraping today's
        archive/category pages (see module docstring - there is no fabricated
        historical dataset here anymore).
        """
        ticker = company.ticker.upper()
        articles = self._scrape_live_archives(company)
        logger.info(f"[{self.source_name}] Collected {len(articles)} backfill articles for {ticker}")
        return articles

    def fetch_macro_news(self) -> List[Article]:
        """
        Fetches unfiltered macro news articles tagged with source="*_archive_backfill".

        There is no fabricated historical dataset to fall back to (see module
        docstring) - this currently returns an empty list, since
        _scrape_live_archives is company-keyword-filtered and has no
        macro-wide equivalent. Left as a stub rather than removed so callers
        (aggregator.py) keep a stable interface.
        """
        return []

    def _scrape_live_archives(self, company: CompanyMetadata) -> List[Article]:
        """
        Scrapes archive endpoints of Dawn / Business Recorder / Profit if accessible.
        """
        articles = []
        # Attempt to scrape Dawn Business Archive RSS/Search if reachable
        urls = [
            ("https://www.dawn.com/authors/1/dawn-business", "dawn"),
            ("https://www.brecorder.com/markets", "brecorder"),
            ("https://profit.pakistantoday.com.pk/category/banking-finance/", "profit"),
        ]

        for url, portal in urls:
            try:
                resp = requests.get(url, headers=HEADERS, timeout=8)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    # Extract headline tags (h2, h3, a)
                    headings = soup.find_all(["h2", "h3", "a"], limit=30)
                    for h in headings:
                        text = h.get_text(strip=True)
                        if len(text) > 25 and any(kw in text.lower() for kw in [company.name.lower(), company.ticker.lower()]):
                            source_tag = f"{portal}_archive_backfill"
                            art = Article(
                                headline=text,
                                summary=f"Live scraped archive article from {portal.capitalize()}",
                                content=None,
                                url=h.get("href", url),
                                source=source_tag,
                                published_at=datetime.now(timezone.utc),
                                author="Archive Scraper",
                                ticker=company.ticker,
                            )
                            articles.append(art)
            except Exception as e:
                logger.debug(f"Live archive scrape for {portal} skipped: {e}")

        return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = ArchiveBackfillCollector()
    dummy_co = CompanyMetadata(ticker="PSO", name="Pakistan State Oil", aliases=["PSO"])
    arts = collector.fetch_news(dummy_co)
    print(f"Total backfill articles fetched for PSO: {len(arts)}")
    for a in arts[:3]:
        print(f" - [{a.source}] {a.published_at.strftime('%Y-%m-%d')}: {a.headline}")
