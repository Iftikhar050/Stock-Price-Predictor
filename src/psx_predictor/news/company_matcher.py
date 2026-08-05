import re
from typing import List
from .models import Article, CompanyMetadata
from .config import COMPANIES

class CompanyMatcher:
    """
    Intelligently matches a raw article to one or more stock tickers based on
    company names, aliases, and search keywords.
    """
    
    @staticmethod
    def match_article(article: Article) -> List[Article]:
        """
        Takes a raw article and returns a list of Article objects, one for each
        matched ticker. If an article mentions both PSO and OGDC, this returns
        two Article instances, identical except for the `ticker` field.
        """
        matched_tickers = set()
        
        # We search in both headline and summary/content for maximum recall.
        text_to_search = (
            article.headline + " " + 
            (article.summary or "") + " " + 
            (article.content or "")
        ).lower()
        
        for ticker, metadata in COMPANIES.items():
            # Check if any keyword, alias, or the ticker itself is present
            search_terms = [metadata.name, metadata.ticker] + metadata.aliases + metadata.search_keywords
            
            for term in search_terms:
                # Use regex with word boundaries to avoid partial matches
                # e.g., we don't want "FFC" to match "OFFCUT"
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                if re.search(pattern, text_to_search):
                    matched_tickers.add(ticker)
                    break # Move to next company once this one is matched
                    
        # Create separate article instances for each matched ticker
        matched_articles = []
        for ticker in matched_tickers:
            # We must create a new Pydantic model instance with the assigned ticker
            new_article = article.model_copy()
            new_article.ticker = ticker
            matched_articles.append(new_article)
            
        return matched_articles
