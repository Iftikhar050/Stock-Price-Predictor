import logging
import time
from typing import List
from datetime import datetime, timezone, date, timedelta
import dateparser

from gnews import GNews
from ..base import BaseCollector
from ..models import Article, CompanyMetadata
from ..company_matcher import REQUIRES_PAKISTAN_ANCHOR

logger = logging.getLogger(__name__)

class GoogleNewsCollector(BaseCollector):
    """
    Collects news using Google News RSS feeds via the gnews library.
    """

    @property
    def source_name(self) -> str:
        return "Google News"

    def __init__(self, period: str = '2d'):
        # We focus on recent news for daily runs (e.g., last 2 days to overlap and avoid missing anything)
        self.gnews = GNews(language='en', country='PK', period=period, max_results=10)

    def _parse_item(self, item: dict, default_ticker: str | None) -> Article:
        dt = dateparser.parse(item.get('published date', ''))
        if dt:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        return Article(
            headline=item.get('title', ''),
            summary=item.get('description', ''),
            content=None,
            url=item.get('url', ''),
            source=item.get('publisher', {}).get('title', self.source_name),
            published_at=dt,
            author=None,
            ticker=default_ticker,
        )

    def fetch_news_range(
        self,
        company: CompanyMetadata,
        start_date: date,
        end_date: date,
        chunk_days: int = 30,
        sleep_between_calls: float = 1.5,
        include_alias: bool = True,
    ) -> List[Article]:
        """
        Backfills historical news for a company using Google News' own date-range
        search (gnews' start_date/end_date, which map to Google's before:/after:
        search operators). Unlike RSS feeds - which only ever carry the current
        handful of items and have no archive - Google News' search index itself
        goes back years, so this is the one collector that can genuinely add
        historical depth rather than just near-real-time coverage. Confirmed via
        manual test: querying a 2024 date range returns real 2024 headlines.

        Chunked into `chunk_days`-day windows because gnews caps each call's
        results (max_results on the client), so a single multi-year query would
        silently truncate to the newest items in that huge range instead of
        covering it evenly.
        """
        articles: List[Article] = []
        # Companies whose name/ticker is a generic collision risk (e.g. "Abbott")
        # get buried by the unrelated global entity even in the raw search
        # results - confirmed via backfill: a bare "Abbott" query returned zero
        # genuine Pakistan-specific hits in 175 raw results, all Texas governor/
        # global Abbott Labs/TV show content. Qualifying the query itself with
        # "Pakistan" surfaces the real local coverage instead of relying purely
        # on post-hoc filtering of an already-dominated result set.
        name_query = f"{company.name} Pakistan" if company.ticker in REQUIRES_PAKISTAN_ANCHOR else company.name
        queries = [name_query]
        if include_alias and company.aliases:
            queries.append(company.aliases[0])

        cursor = start_date
        while cursor < end_date:
            chunk_end = min(cursor + timedelta(days=chunk_days), end_date)
            g = GNews(
                language='en', country='PK', max_results=25,
                start_date=(cursor.year, cursor.month, cursor.day),
                end_date=(chunk_end.year, chunk_end.month, chunk_end.day),
            )
            for query in queries:
                try:
                    results = g.get_news(query)
                    for item in results:
                        articles.append(self._parse_item(item, None))
                except Exception as e:
                    logger.error(
                        f"[{self.source_name}] Historical fetch failed for {query} "
                        f"({cursor}..{chunk_end}): {e}"
                    )
                time.sleep(sleep_between_calls)
            cursor = chunk_end

        return articles

    def fetch_news(self, company: CompanyMetadata) -> List[Article]:
        articles = []
        
        # We query the primary name and alias
        queries = [company.name]
        if company.aliases:
            queries.append(company.aliases[0])
            
        for query in queries:
            logger.info(f"[{self.source_name}] Querying for: {query}")
            try:
                results = self.gnews.get_news(query)
                for item in results:
                    # gnews returns dicts with 'title', 'description', 'published date', 'url', 'publisher'
                    
                    # Parse date string to UTC datetime
                    dt = dateparser.parse(item.get('published date', ''))
                    if dt:
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = datetime.now(timezone.utc)
                        
                    article = Article(
                        headline=item.get('title', ''),
                        summary=item.get('description', ''),
                        content=None, # Google News only gives snippets
                        url=item.get('url', ''),
                        source=item.get('publisher', {}).get('title', self.source_name),
                        published_at=dt,
                        author=None,
                        ticker=None # Matcher assigns this later
                    )
                    articles.append(article)
            except Exception as e:
                logger.error(f"[{self.source_name}] Failed to fetch news for {query}: {e}")
                
        return articles

    def fetch_macro_news(self) -> List[Article]:
        """Fetch general macroeconomic and political news for Pakistan."""
        articles = []
        queries = ["Pakistan economy", "State Bank of Pakistan", "Pakistan politics", "Pakistan government"]
        
        for query in queries:
            logger.info(f"[{self.source_name}] Querying MACRO news for: {query}")
            try:
                results = self.gnews.get_news(query)
                for item in results:
                    dt = dateparser.parse(item.get('published date', ''))
                    if dt:
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = datetime.now(timezone.utc)
                        
                    article = Article(
                        headline=item.get('title', ''),
                        summary=item.get('description', ''),
                        content=None,
                        url=item.get('url', ''),
                        source=item.get('publisher', {}).get('title', self.source_name),
                        published_at=dt,
                        author=None,
                        ticker="MACRO"
                    )
                    articles.append(article)
            except Exception as e:
                logger.error(f"[{self.source_name}] Failed to fetch MACRO news for {query}: {e}")
                
        return articles
