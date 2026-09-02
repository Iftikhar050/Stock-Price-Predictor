import logging
from typing import List
from datetime import datetime, timezone
import dateparser

from gnews import GNews
from ..base import BaseCollector
from ..models import Article, CompanyMetadata

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
