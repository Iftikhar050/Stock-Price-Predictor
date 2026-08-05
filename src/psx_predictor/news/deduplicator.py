from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
import hashlib
from typing import List
from .models import Article

class Deduplicator:
    """
    Handles removing duplicate articles based on URL normalization and headline hashing.
    """
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """
        Strips tracking parameters (e.g., utm_source) to normalize URLs.
        """
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        
        # Remove common tracking parameters
        tracking_params = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 'fbclid', 'gclid'}
        filtered_qs = {k: v for k, v in qs.items() if k not in tracking_params}
        
        new_query = urlencode(filtered_qs, doseq=True)
        return urlunparse(parsed._replace(query=new_query))
        
    @staticmethod
    def hash_headline(headline: str) -> str:
        """
        Creates a basic hash of the headline. 
        For more advanced similarity, fuzzy matching could be added here.
        """
        cleaned = "".join(e for e in headline.lower() if e.isalnum())
        return hashlib.md5(cleaned.encode()).hexdigest()
        
    @classmethod
    def filter_duplicates(cls, articles: List[Article], existing_urls: set, existing_headline_hashes: set) -> List[Article]:
        """
        Filters a list of newly scraped articles against already known ones.
        """
        unique_articles = []
        for article in articles:
            norm_url = cls.normalize_url(article.url)
            h_hash = cls.hash_headline(article.headline)
            
            if norm_url in existing_urls or h_hash in existing_headline_hashes:
                continue
                
            existing_urls.add(norm_url)
            existing_headline_hashes.add(h_hash)
            
            # Update article with normalized URL before returning
            article.url = norm_url
            unique_articles.append(article)
            
        return unique_articles
