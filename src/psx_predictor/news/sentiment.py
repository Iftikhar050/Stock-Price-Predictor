import logging
from abc import ABC, abstractmethod
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .models import Article

logger = logging.getLogger(__name__)

class SentimentEngine(ABC):
    @abstractmethod
    def analyze(self, articles: List[Article]) -> List[Article]:
        pass

class VaderSentimentEngine(SentimentEngine):
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer.
    Great for social media and short headlines. Swappable later for FinBERT.
    """
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize VADER: {e}")
            self.analyzer = None

    def analyze(self, articles: List[Article]) -> List[Article]:
        if not self.analyzer:
            logger.error("VADER analyzer is not initialized. Returning articles without sentiment.")
            return articles
            
        for article in articles:
            # Analyze headline. Summary can be appended if available, but headline is strongest signal.
            text_to_analyze = article.headline
            if article.summary:
                text_to_analyze += " " + article.summary
                
            try:
                scores = self.analyzer.polarity_scores(text_to_analyze)
                # 'compound' is a normalized, weighted composite score [-1.0, 1.0]
                article.sentiment_score = scores.get('compound', 0.0)
            except Exception as e:
                logger.error(f"Error calculating sentiment for article '{article.headline}': {e}")
                article.sentiment_score = 0.0
                
        return articles
