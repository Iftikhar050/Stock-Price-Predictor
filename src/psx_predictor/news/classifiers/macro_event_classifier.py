"""
macro_event_classifier.py
--------------------------
Rule-based tagger for Political / Geopolitical / Regulatory topics.
Runs on every article already fetched by the existing news collectors
(Google News, Local News, Alpha Vantage) — no new source needed, this
just adds a classification pass that the current pipeline never applied.

Closes PDF Groups 19 (Political Factors), 20 (Geopolitical Factors),
and 31 (Regulatory Factors).

Design choice: deliberate keyword/regex approach rather than ML classifier.
Advantages: fully auditable (you can see exactly why a day got tagged),
zero training data needed, fast inference, no model download.
Easily extensible — just add patterns to the dicts below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MacroTags:
    """Result of classifying a single article's text."""
    political: bool = False
    geopolitical: bool = False
    regulatory: bool = False
    india_pakistan_tension: bool = False
    middle_east_conflict: bool = False
    election_related: bool = False
    # Canonical topic category for the stock_news.topic_category column
    topic_category: Optional[str] = None

    def __post_init__(self):
        # Derive the canonical topic_category from individual flags
        # Priority: POLITICAL > GEOPOLITICAL > REGULATORY > None
        if self.political or self.election_related:
            self.topic_category = "POLITICAL"
        elif self.geopolitical or self.india_pakistan_tension or self.middle_east_conflict:
            self.topic_category = "GEOPOLITICAL"
        elif self.regulatory:
            self.topic_category = "REGULATORY"


# ── Pattern dictionaries ────────────────────────────────────────────────────
# Each key maps to a compiled regex that fires on headline + body text.
# Patterns are case-insensitive and use word boundaries where appropriate.

_PATTERNS: dict[str, re.Pattern] = {
    "political": re.compile(
        r"\b("
        r"no-?confidence|coalition\s+government|cabinet\s+reshuffle|"
        r"prime\s+minister|chief\s+minister|opposition\s+party|"
        r"national\s+assembly|senate\s+session|parliament|"
        r"PTI|PML-?N|PPP|JUI-?F|MQM|martial\s+law|"
        r"speaker\s+of\s+the|prorogation|dissolution\s+of\s+assembly|"
        r"supreme\s+court\s+verdict|chief\s+justice|"
        r"army\s+chief|coas|establishment|dharna|long\s+march"
        r")\b", re.IGNORECASE
    ),
    "geopolitical": re.compile(
        r"\b("
        r"border\s+tension|LoC|line\s+of\s+control|"
        r"sanctions|ceasefire|troops|missile|airstrike|"
        r"trade\s+war|tariff\s+war|CPEC|belt\s+and\s+road|"
        r"NATO|UN\s+security\s+council|nuclear|proliferation"
        r")\b", re.IGNORECASE
    ),
    "regulatory": re.compile(
        r"\b("
        r"SECP|FBR\s+notification|SRO\s+\d+|"
        r"NEPRA\s+tariff|OGRA|PEMRA|"
        r"competition\s+commission|anti-?trust|"
        r"tax\s+amnesty|super\s+tax|windfall\s+tax|"
        r"import\s+duty|customs\s+duty|regulatory\s+duty|"
        r"advance\s+tax|capital\s+gains\s+tax|CGT"
        r")\b", re.IGNORECASE
    ),
    "india_pakistan_tension": re.compile(
        r"\b("
        r"india.{0,25}pakistan|pakistan.{0,25}india|"
        r"loc\b|line\s+of\s+control|kashmir|pulwama|balakot|"
        r"surgical\s+strike|cross.?border\s+shelling|"
        r"bilateral\s+relations|diplomatic\s+tensions?"
        r")\b", re.IGNORECASE
    ),
    "middle_east_conflict": re.compile(
        r"\b("
        r"gaza|israel|hamas|hezbollah|iran.{0,15}strike|"
        r"houthi|red\s+sea|strait\s+of\s+hormuz|"
        r"yemen|saudi.{0,10}iran|syria\s+conflict|"
        r"oil\s+embargo|opec\s+cut|opec\+?"
        r")\b", re.IGNORECASE
    ),
    "election_related": re.compile(
        r"\b("
        r"general\s+election|by-?election|ECP|election\s+commission|"
        r"caretaker\s+government|interim\s+PM|"
        r"voter\s+turnout|rigging|election\s+tribunal|"
        r"delimitation|election\s+schedule"
        r")\b", re.IGNORECASE
    ),
}

# ── Corporate topic patterns (for articles not caught by macro classifiers) ──
_CORPORATE_PATTERN = re.compile(
    r"\b("
    r"earnings|quarterly\s+result|financial\s+result|profit|loss|"
    r"dividend|bonus\s+share|right\s+issue|merger|acquisition|"
    r"IPO|demerger|buyback|AGM|EGM|board\s+meeting|"
    r"CEO|CFO|managing\s+director|resignation|appointment"
    r")\b", re.IGNORECASE
)

_MACRO_ECONOMIC_PATTERN = re.compile(
    r"\b("
    r"GDP|inflation|CPI|monetary\s+policy|interest\s+rate|"
    r"SBP\s+rate|discount\s+rate|MPC\s+decision|"
    r"fiscal\s+deficit|trade\s+deficit|current\s+account|"
    r"IMF\s+program|IMF\s+tranche|debt\s+restructuring|"
    r"remittances|foreign\s+reserves|balance\s+of\s+payment|"
    r"budget\s+20\d{2}|mini\s+budget|supplementary\s+budget"
    r")\b", re.IGNORECASE
)

_SECTOR_SPECIFIC_PATTERN = re.compile(
    r"\b("
    r"oil\s+marketing|petroleum\s+levy|OMC|"
    r"circular\s+debt|power\s+tariff|gas\s+price|"
    r"cement\s+dispatch|auto\s+sales|"
    r"banking\s+spread|ADR|CAR\s+ratio|"
    r"fertilizer\s+off-?take|urea\s+price|DAP\s+price"
    r")\b", re.IGNORECASE
)


def classify(headline: str, body: str = "") -> MacroTags:
    """
    Classify an article's combined text into macro-event tags.

    Args:
        headline: Article headline.
        body: Article body/summary text (optional).

    Returns:
        MacroTags dataclass with boolean flags for each topic category
        and a derived `topic_category` string.
    """
    text = f"{headline} {body}"
    flags = {k: bool(p.search(text)) for k, p in _PATTERNS.items()}
    tags = MacroTags(**flags)

    # If no macro tag fired, try corporate/macro-economic/sector-specific
    if tags.topic_category is None:
        if _CORPORATE_PATTERN.search(text):
            tags.topic_category = "CORPORATE"
        elif _MACRO_ECONOMIC_PATTERN.search(text):
            tags.topic_category = "MACRO_ECONOMIC"
        elif _SECTOR_SPECIFIC_PATTERN.search(text):
            tags.topic_category = "SECTOR_SPECIFIC"

    return tags


def classify_batch(articles: list[dict]) -> list[dict]:
    """
    Classify a batch of article dicts in-place, adding 'topic_category'
    and individual flag fields.

    Each dict is expected to have at least a 'headline' key.
    Optional: 'summary' or 'body' key for additional text.

    Returns the same list with added fields.
    """
    for article in articles:
        headline = article.get("headline", "")
        body = article.get("summary", "") or article.get("body", "") or ""
        tags = classify(headline, body)

        article["topic_category"] = tags.topic_category
        article["is_political"] = tags.political
        article["is_geopolitical"] = tags.geopolitical
        article["is_regulatory"] = tags.regulatory
        article["is_india_pakistan_tension"] = tags.india_pakistan_tension
        article["is_middle_east_conflict"] = tags.middle_east_conflict
        article["is_election_related"] = tags.election_related

    return articles
