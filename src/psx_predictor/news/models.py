from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

class Article(BaseModel):
    """Standardized representation of a financial news article."""
    headline: str = Field(..., description="The title of the news article")
    summary: Optional[str] = Field(None, description="A brief snippet or summary of the article")
    content: Optional[str] = Field(None, description="The full text content of the article, if available")
    url: str = Field(..., description="The original URL of the article")
    source: str = Field(..., description="The publisher or source (e.g., 'Google News', 'Business Recorder')")
    published_at: datetime = Field(..., description="The timestamp when the article was published in UTC")
    author: Optional[str] = Field(None, description="The author of the article")
    ticker: Optional[str] = Field(None, description="The stock ticker this article relates to (can be populated later)")
    sentiment_score: Optional[float] = Field(None, description="The compound sentiment score (-1.0 to 1.0)")

    # Topic classification (populated by macro_event_classifier)
    topic_category: Optional[str] = Field(None, description="Canonical topic: CORPORATE, POLITICAL, GEOPOLITICAL, MACRO_ECONOMIC, SECTOR_SPECIFIC, REGULATORY")

    # FinBERT probability triplet (populated by HybridSentimentEngine)
    finbert_pos: Optional[float] = Field(None, description="FinBERT positive probability [0-1]")
    finbert_neg: Optional[float] = Field(None, description="FinBERT negative probability [0-1]")
    finbert_neu: Optional[float] = Field(None, description="FinBERT neutral probability [0-1]")

class CompanyMetadata(BaseModel):
    """Metadata mapping for a single stock ticker to assist with news matching."""
    ticker: str = Field(..., description="The stock symbol (e.g., 'OGDC')")
    name: str = Field(..., description="The full company name")
    aliases: List[str] = Field(default_factory=list, description="Other names the company is known by")
    search_keywords: List[str] = Field(default_factory=list, description="Keywords used to query news APIs or scrapers")
