from datetime import datetime, timezone

import pytest

from src.psx_predictor.news.models import Article
from src.psx_predictor.news.sentiment import (
    FinBERTSentimentEngine,
    HybridSentimentEngine,
    VaderSentimentEngine,
)


def _article(headline, summary=None):
    return Article(
        headline=headline, summary=summary,
        url="https://example.com/a", source="Test", published_at=datetime.now(timezone.utc),
    )


class TestVaderSentimentEngine:
    """Uses the real vaderSentiment package (installed, fast, deterministic) -
    no mocking needed, and it exercises the actual custom financial lexicon
    extension this class adds."""

    def test_positive_headline_scores_positive(self):
        engine = VaderSentimentEngine()
        score, pos, neg, neu = engine.score_text("Company reports record profit and strong growth")
        assert score > 0
        assert pos is None and neg is None and neu is None  # VADER has no triplet

    def test_negative_headline_scores_negative(self):
        engine = VaderSentimentEngine()
        score, _, _, _ = engine.score_text("Company stock crashes amid fraud scandal")
        assert score < 0

    def test_custom_financial_lexicon_is_applied(self):
        """"crash" isn't in VADER's base lexicon with this intensity - this
        confirms FINANCIAL_LEXICON actually got merged in, not just that some
        generic negative word was recognized."""
        engine = VaderSentimentEngine()
        assert engine.analyzer.lexicon["crash"] == -3.0
        assert engine.analyzer.lexicon["dividend"] == 1.5

    def test_analyze_sets_sentiment_score_on_each_article(self):
        engine = VaderSentimentEngine()
        articles = [_article("Company posts record profit"), _article("Company faces major scandal")]
        result = engine.analyze(articles)
        assert result[0].sentiment_score > 0
        assert result[1].sentiment_score < 0

    def test_analyze_combines_headline_and_summary(self):
        engine = VaderSentimentEngine()
        headline_only = engine.analyze([_article("Company update")])[0].sentiment_score
        with_bad_summary = engine.analyze(
            [_article("Company update", summary="massive fraud scandal and crash")]
        )[0].sentiment_score
        assert with_bad_summary < headline_only

    def test_gracefully_handles_unavailable_analyzer(self):
        engine = VaderSentimentEngine()
        engine.analyzer = None
        score, pos, neg, neu = engine.score_text("irrelevant")
        assert (score, pos, neg, neu) == (0.0, None, None, None)
        articles = engine.analyze([_article("Company update")])
        assert articles[0].sentiment_score is None


class TestFinBERTSentimentEngine:
    def test_unavailable_when_transformers_not_installed(self, mocker):
        mocker.patch.dict("sys.modules", {"transformers": None})
        engine = FinBERTSentimentEngine()
        assert engine.is_available is False
        assert engine.score_text("anything") == (0.0, None, None, None)

    def test_analyze_returns_articles_unscored_when_unavailable(self, mocker):
        mocker.patch.dict("sys.modules", {"transformers": None})
        engine = FinBERTSentimentEngine()
        articles = [_article("Company update")]
        result = engine.analyze(articles)
        assert result[0].sentiment_score is None

    def test_score_text_maps_softmax_probs_to_compound(self, mocker):
        """FinBERT label order is positive=0, negative=1, neutral=2;
        compound = P(positive) - P(negative) - verify that mapping directly
        rather than trusting a real model load (slow, non-deterministic)."""
        import torch

        engine = FinBERTSentimentEngine.__new__(FinBERTSentimentEngine)
        engine._available = True
        engine.device = "cpu"
        fake_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        engine.tokenizer = mocker.Mock(return_value=fake_inputs)

        mock_logits = torch.tensor([[2.0, 0.5, 0.1]])
        engine.model = mocker.Mock(return_value=mocker.Mock(logits=mock_logits))

        compound, pos, neg, neu = engine.score_text("Company reports record profit")
        probs = torch.nn.functional.softmax(mock_logits, dim=-1)[0]
        assert pos == pytest.approx(float(probs[0]))
        assert neg == pytest.approx(float(probs[1]))
        assert compound == pytest.approx(pos - neg)


class TestHybridSentimentEngine:
    def test_falls_back_to_vader_when_finbert_unavailable(self, mocker):
        mocker.patch.object(FinBERTSentimentEngine, "__init__", lambda self, *a, **kw: setattr(self, "_available", False))
        engine = HybridSentimentEngine(prefer_finbert=True)
        assert engine._engine_name == "VADER"
        assert engine._primary is engine.vader

    def test_uses_finbert_when_available(self, mocker):
        mocker.patch.object(
            FinBERTSentimentEngine, "__init__",
            lambda self, *a, **kw: setattr(self, "_available", True),
        )
        engine = HybridSentimentEngine(prefer_finbert=True)
        assert engine._engine_name == "FinBERT"
        assert engine._primary is engine.finbert

    def test_prefer_finbert_false_skips_finbert_entirely(self):
        engine = HybridSentimentEngine(prefer_finbert=False)
        assert engine.finbert is None
        assert engine._engine_name == "VADER"

    def test_analyze_with_probabilities_includes_finbert_fields(self, mocker):
        mocker.patch.object(FinBERTSentimentEngine, "__init__", lambda self, *a, **kw: setattr(self, "_available", False))
        engine = HybridSentimentEngine(prefer_finbert=True)  # falls back to real VADER
        results = engine.analyze_with_probabilities([_article("Company reports record profit")])
        assert "finbert_pos" in results[0]
        assert "finbert_neg" in results[0]
        assert "finbert_neu" in results[0]
        assert results[0]["sentiment_score"] > 0
