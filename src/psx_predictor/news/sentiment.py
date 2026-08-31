"""
sentiment.py
-------------
Hybrid Financial Sentiment Engine: FinBERT (primary) + VADER (fallback).

FinBERT (ProsusAI/finbert) is a BERT model fine-tuned on financial text.
It returns probability triplets (positive, negative, neutral) and a
compound score mapped to [-1.0, +1.0].

If PyTorch/transformers are not available (e.g. lightweight deployment),
falls back to VADER with a custom financial lexicon extension.

Closes PDF Groups 26 (Investor Sentiment) and 40 (Market Psychology).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .models import Article

logger = logging.getLogger(__name__)


class SentimentEngine(ABC):
    """Abstract base for all sentiment engines."""

    @abstractmethod
    def analyze(self, articles: List[Article]) -> List[Article]:
        """Score articles in-place and return them."""
        pass

    @abstractmethod
    def score_text(self, text: str) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
        """
        Returns (compound_score, positive_prob, negative_prob, neutral_prob).
        For engines that don't produce the triplet, return (score, None, None, None).
        """
        pass


class FinBERTSentimentEngine(SentimentEngine):
    """
    FinBERT (ProsusAI/finbert) sentiment analyzer.
    Returns calibrated positive/negative/neutral probabilities.
    Compound score = P(positive) - P(negative), range [-1.0, +1.0].
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu"):
        self._available = False
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(f"Loading FinBERT model: {model_name} on {device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            self.device = device
            self._available = True
            logger.info("FinBERT loaded successfully.")
        except ImportError:
            logger.warning(
                "transformers/torch not installed. FinBERT unavailable. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    def score_text(self, text: str) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
        if not self._available:
            return (0.0, None, None, None)

        import torch

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        # FinBERT label order: positive=0, negative=1, neutral=2
        pos = float(probs[0])
        neg = float(probs[1])
        neu = float(probs[2])
        compound = pos - neg  # [-1.0, +1.0]

        return (compound, pos, neg, neu)

    def analyze(self, articles: List[Article]) -> List[Article]:
        if not self._available:
            logger.warning("FinBERT not available. Returning articles without sentiment.")
            return articles

        for article in articles:
            text = article.headline
            if article.summary:
                text += " " + article.summary

            try:
                compound, pos, neg, neu = self.score_text(text)
                article.sentiment_score = compound
            except Exception as e:
                logger.error(f"FinBERT error on '{article.headline[:60]}': {e}")
                article.sentiment_score = 0.0

        return articles


class VaderSentimentEngine(SentimentEngine):
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analyzer.
    Extended with custom financial lexicon for PSX-specific terminology.
    Great for social media and short headlines. Swappable later for FinBERT.
    """

    # Custom financial lexicon additions for VADER
    # (word -> sentiment intensity, same scale as VADER's existing lexicon)
    FINANCIAL_LEXICON = {
        # Positive financial signals
        "dividend": 1.5,
        "bonus": 1.2,
        "profit": 1.8,
        "earnings beat": 2.5,
        "upgrade": 2.0,
        "outperform": 2.0,
        "bullish": 2.0,
        "rally": 1.8,
        "expansion": 1.5,
        "record high": 2.5,
        "strong results": 2.0,
        # Negative financial signals
        "loss": -1.8,
        "downgrade": -2.0,
        "bearish": -2.0,
        "circular debt": -2.5,
        "default": -3.0,
        "shutdown": -2.0,
        "layoff": -2.0,
        "plunge": -2.5,
        "crash": -3.0,
        "selloff": -2.0,
        "sell-off": -2.0,
        "scandal": -2.5,
        "litigation": -1.5,
        "devaluation": -2.0,
        "inflation": -0.5,
        "deficit": -1.0,
        "sanctions": -2.0,
    }

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.analyzer = SentimentIntensityAnalyzer()
            # Extend VADER's lexicon with financial terms
            self.analyzer.lexicon.update(self.FINANCIAL_LEXICON)
            logger.info("VADER initialized with extended financial lexicon.")
        except ImportError:
            logger.error("vaderSentiment not installed. pip install vaderSentiment")
            self.analyzer = None
        except Exception as e:
            logger.error(f"Failed to initialize VADER: {e}")
            self.analyzer = None

    def score_text(self, text: str) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
        if not self.analyzer:
            return (0.0, None, None, None)

        scores = self.analyzer.polarity_scores(text)
        return (scores.get("compound", 0.0), None, None, None)

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
                article.sentiment_score = scores.get("compound", 0.0)
            except Exception as e:
                logger.error(f"Error calculating sentiment for article '{article.headline}': {e}")
                article.sentiment_score = 0.0

        return articles


class HybridSentimentEngine(SentimentEngine):
    """
    Production sentiment engine: tries FinBERT first, falls back to VADER.
    Also provides batch scoring with FinBERT probability triplets for
    downstream weighting (finbert_pos, finbert_neg, finbert_neu).
    """

    def __init__(self, prefer_finbert: bool = True):
        self.finbert = FinBERTSentimentEngine() if prefer_finbert else None
        self.vader = VaderSentimentEngine()

        if self.finbert and self.finbert.is_available:
            self._primary = self.finbert
            self._engine_name = "FinBERT"
        else:
            self._primary = self.vader
            self._engine_name = "VADER"
            if prefer_finbert:
                logger.warning("FinBERT unavailable; falling back to VADER.")

        logger.info(f"HybridSentimentEngine using: {self._engine_name}")

    def score_text(self, text: str) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
        return self._primary.score_text(text)

    def analyze(self, articles: List[Article]) -> List[Article]:
        """
        Score all articles. If FinBERT is available, also populates the
        finbert probability fields on each article's model_dump output.
        """
        return self._primary.analyze(articles)

    def analyze_with_probabilities(self, articles: List[Article]) -> List[dict]:
        """
        Returns article dicts with additional finbert_pos/neg/neu fields.
        For downstream DB insertion into stock_news with the new columns.
        """
        results = []
        for article in articles:
            text = article.headline
            if article.summary:
                text += " " + article.summary

            try:
                compound, pos, neg, neu = self.score_text(text)
                article.sentiment_score = compound
            except Exception as e:
                logger.error(f"Sentiment error: {e}")
                compound, pos, neg, neu = 0.0, None, None, None
                article.sentiment_score = 0.0

            d = article.model_dump()
            d["finbert_pos"] = pos
            d["finbert_neg"] = neg
            d["finbert_neu"] = neu
            results.append(d)

        return results
