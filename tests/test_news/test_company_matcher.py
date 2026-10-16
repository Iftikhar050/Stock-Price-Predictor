from datetime import datetime, timezone

import pytest

from src.psx_predictor.news.company_matcher import CompanyMatcher
from src.psx_predictor.news.models import Article, CompanyMetadata

# A small, controlled company universe instead of the real COMPANIES dict
# (config.py) - keeps these tests independent of that list's real contents
# and immune to it growing/changing over time.
FAKE_COMPANIES = {
    "HBL": CompanyMetadata(
        ticker="HBL", name="Habib Bank Limited",
        aliases=["Habib Bank"], search_keywords=["HBL earnings"],
    ),
    "NBP": CompanyMetadata(
        ticker="NBP", name="National Bank of Pakistan",
        aliases=["National Bank"], search_keywords=["NBP earnings"],
    ),
    "ABOT": CompanyMetadata(
        ticker="ABOT", name="Abbott Laboratories Pakistan",
        aliases=["Abbott"], search_keywords=["Abbott Pakistan"],
    ),
}


def _article(headline, summary=None):
    return Article(
        headline=headline,
        summary=summary,
        url="https://example.com/a",
        source="Test Source",
        published_at=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def _patch_companies(mocker):
    mocker.patch("src.psx_predictor.news.company_matcher.COMPANIES", FAKE_COMPANIES)


def test_trusted_name_matches_without_corroboration():
    """A company's full official name is trusted outright - no finance jargon
    needed, matching the module's own "FFC expects positive urea market"
    style real-world example."""
    articles = CompanyMatcher.match_article(_article("Habib Bank Limited opens new branch in Multan"))
    assert {a.ticker for a in articles} == {"HBL"}


def test_trusted_ticker_matches_without_corroboration():
    articles = CompanyMatcher.match_article(_article("NBP posts strong quarterly performance"))
    assert {a.ticker for a in articles} == {"NBP"}


def test_generic_alias_without_corroboration_is_rejected():
    """"National Bank" alone (no finance/Pakistan context) is exactly the
    "National Bank Open" tennis-tournament collision this module documents."""
    articles = CompanyMatcher.match_article(_article("National Bank Open quarterfinal results announced"))
    assert articles == []


def test_generic_alias_with_finance_context_is_accepted():
    articles = CompanyMatcher.match_article(_article("National Bank reports record profit this quarter"))
    assert {a.ticker for a in articles} == {"NBP"}


def test_generic_alias_with_pakistan_anchor_is_accepted():
    articles = CompanyMatcher.match_article(_article("National Bank expands branches across Karachi"))
    assert {a.ticker for a in articles} == {"NBP"}


def test_sports_denylist_rejects_even_trusted_term_match():
    """HBL title-sponsors the Pakistan Super League - PSL cricket coverage
    must never count as bank news, even though "HBL" itself is trusted."""
    articles = CompanyMatcher.match_article(
        _article("HBL PSL player draft: franchise cricket retentions announced")
    )
    assert articles == []


def test_non_financial_denylist_blocks_unrelated_sport_mention():
    articles = CompanyMatcher.match_article(_article("Habib Bank Limited stadium hosts cricket tournament final"))
    assert articles == []


def test_requires_pakistan_anchor_ticker_rejects_without_anchor():
    """ABOT collides with the NYSE-listed Abbott Labs, which has its own
    genuine finance coverage - generic finance-context corroboration isn't
    enough, a real Pakistan/PSX anchor is required."""
    articles = CompanyMatcher.match_article(
        _article("Abbott Laboratories stock price moves on strong earnings report")
    )
    assert articles == []


def test_requires_pakistan_anchor_ticker_accepts_with_anchor():
    articles = CompanyMatcher.match_article(
        _article("Abbott Pakistan announces new manufacturing plant in Karachi")
    )
    assert {a.ticker for a in articles} == {"ABOT"}


def test_cross_entity_denylist_blocks_known_namesake():
    """HBL Engineering (India) is an unrelated company whose own coverage
    contains genuine finance words, so context-keyword corroboration alone
    wouldn't catch this - the explicit cross-entity denylist is required."""
    articles = CompanyMatcher.match_article(
        _article("HBL Engineering shares rise after strong quarterly results", summary="HBL Power Systems reports growth")
    )
    assert articles == []


def test_multi_ticker_article_matches_all_mentioned_companies():
    articles = CompanyMatcher.match_article(
        _article("Habib Bank Limited and National Bank of Pakistan both report record profits")
    )
    assert {a.ticker for a in articles} == {"HBL", "NBP"}


def test_no_match_returns_empty_list():
    articles = CompanyMatcher.match_article(_article("Local weather forecast predicts rain this weekend"))
    assert articles == []


def test_search_is_case_insensitive():
    articles = CompanyMatcher.match_article(_article("HABIB BANK LIMITED posts quarterly results"))
    assert {a.ticker for a in articles} == {"HBL"}
