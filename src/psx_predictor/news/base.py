from abc import ABC, abstractmethod
from typing import List
from .models import Article, CompanyMetadata

class BaseCollector(ABC):
    """
    Abstract base class for all news collectors.
    Every news source (Google News, Business Recorder, etc.) must implement this interface.
    """
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """The name of the news source."""
        pass
        
    @abstractmethod
    def fetch_news(self, company: CompanyMetadata) -> List[Article]:
        """
        Fetches news articles for a given company.
        
        Args:
            company: The CompanyMetadata object containing search keywords.
            
        Returns:
            A list of Article objects. Should never crash. Return empty list on failure.
        """
        pass
