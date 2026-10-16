import re
from typing import List
from .models import Article, CompanyMetadata
from .config import COMPANIES

# Short/generic aliases (e.g. NBP's "National Bank", as opposed to its full
# name "National Bank of Pakistan" or its ticker) collide with unrelated
# content far more often than they hit real PSX news. Confirmed via DB audit:
# "National Bank" matching the "National Bank Open" tennis tournament
# (Cricinfo/WTA Tennis/Sky Sports/ATP Tour articles), "Faysal Bank" matching
# the "Faysal Bank T-20 Cup" domestic cricket tournament, and a short DHPL
# alias matching Pinkbike mountain-biking gear reviews.
#
# A positive "must contain finance jargon" requirement was tried and rejected -
# it's too strict and drops real company news that doesn't happen to use
# generic finance words (e.g. "FFC expects positive urea market, sees higher
# DAP prices ahead", "NBP expands digital banking footprint across nationwide
# branches" - both genuine, neither contains a word like "profit" or "shares").
#
# Instead: matches via the company's full official name, its ticker symbol, or
# a search_keyword (which are already written as specific phrases like "NBP
# earnings") are trusted outright, modulo the sports/entertainment denylist
# below. Only matches via a shorter/generic alias get the extra scrutiny of
# also requiring finance context or a Pakistan-specific anchor, since that's
# where the real collisions were found.
NON_FINANCIAL_DENYLIST = [
    'tennis', 'cricket', 'football', 'wta', 'atp', 'tournament', 'wicket', 'innings',
    'goal', 'goals', 'league', 'racquet', 'racket', 'grand slam', 'world cup',
    'semifinal', 'semi-final', 'quarterfinal', 'quarter-final', 'wild card',
    'seeded', 'ranking points', 'match point', 'olympics', 'olympic',
    'world series', 'batsman', 'bowler', 'bowling', 'pitch invasion', 'stadium',
    'referee', 'athlete', 'athletics', 'mountain bike', 'mountain biking',
    'downhill', 'bike check', 'cyclist', 'basketball', 'hockey match',
    'golf tournament', 'boxing match', 'wrestling',
    # HBL (Habib Bank) title-sponsors the Pakistan Super League - "HBL PSL" news
    # is cricket, not bank news. Confirmed via a 2025 historical Google News
    # backfill test where the large majority of "HBL" hits were PSL draft/squad
    # coverage, not the bank. "PSL"/"super league" aren't covered by the generic
    # 'league' denylist entry since neither spells out that word.
    'psl', 'pakistan super league', 'player draft', 'retentions', 'team squad',
    'playing xi', 'franchise cricket',
]

# Specific cross-entity name collisions found during the historical backfill
# test, where the colliding entity's own coverage is genuine finance news (so
# the generic FINANCE_CONTEXT_KEYWORDS corroboration check doesn't help - the
# article has real finance words, just about a different, unrelated company).
# "HBL Engineering" (formerly HBL Power Systems) is an India-listed company
# with no relation to Pakistan's Habib Bank Limited (ticker: HBL).
CROSS_ENTITY_DENYLIST = [
    'hbl engineering', 'hbl power systems', 'hbl power',
]
_CROSS_ENTITY_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CROSS_ENTITY_DENYLIST) + r')\b')
_DENY_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(k) for k in NON_FINANCIAL_DENYLIST) + r')\b')

FINANCE_CONTEXT_KEYWORDS = [
    'stock', 'stocks', 'share', 'shares', 'shareholder', 'psx', 'kse', 'kse-100',
    'profit', 'profits', 'earning', 'earnings', 'revenue', 'revenues', 'dividend',
    'dividends', 'investor', 'investors', 'market cap', 'market capitalization',
    'listed company', 'trading session', 'quarterly results', 'quarter results',
    'fiscal year', 'annual report', 'board of directors', 'agm', 'ceo', 'cfo',
    'net income', 'net loss', 'financial results', 'balance sheet', 'ipo',
    'merger', 'acquisition', 'stock exchange', 'bourse', 'equity market',
    'brokerage', 'analyst', 'valuation', 'market value', 'rupee', 'rupees', 'pkr',
    'sbp', 'state bank', 'interest rate', 'policy rate', 'inflation', 'economy',
    'economic', 'gdp', 'export', 'import', 'production', 'output', 'plant',
    'factory', 'subsidiary', 'corporate', 'loan', 'credit rating', 'bonds', 'bond',
    'rating', 'market', 'prices', 'price', 'branch', 'branches', 'digital banking',
    'foreign exchange', 'currency', 'forex', 'app', 'launch', 'pakistan', 'psx-listed',
]
_CONTEXT_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(k) for k in FINANCE_CONTEXT_KEYWORDS) + r')\b')

PAKISTAN_ANCHOR_KEYWORDS = ['pakistan', 'psx', 'kse', 'karachi', 'lahore', 'islamabad', 'rupee', 'rupees', 'pkr']
_ANCHOR_PATTERN = re.compile(r'\b(' + '|'.join(re.escape(k) for k in PAKISTAN_ANCHOR_KEYWORDS) + r')\b')

# Companies whose "name" is itself a bare common word, surname, or globally-used
# brand/ticker with a large, unrelated same-named entity - confirmed via a
# historical Google News backfill test where generic FINANCE_CONTEXT_KEYWORDS
# corroboration wasn't enough, because the unrelated entity has its own genuine
# finance news too (e.g. "Abbott Laboratories Stock (ABT) Moved Up by 3.08%" is
# real finance content, just about the NYSE-listed US company, not PSX's ABOT).
# These require an explicit Pakistan/PSX anchor specifically, not just any
# finance keyword, for every term tier including the company's own name.
REQUIRES_PAKISTAN_ANCHOR = {
    'ABOT',   # 'Abbott' collides with Texas Gov. Greg Abbott, 'Abbott Elementary'
              # (TV show), actor Christopher Abbott, and NYSE-listed Abbott Labs.
    'GLAXO',  # 'GSK' collides with the global GlaxoSmithKline plc itself.
    'SEARL',  # 'Searle' collides with philosopher John Searle and the historical
              # US Searle pharmaceutical company.
    'AGP',    # 'AGP' (ticker) collides with unrelated orgs/firms using the same
              # 3-letter acronym (confirmed: a Ghana NGO report, an unrelated
              # trading firm's CEO hire, a watch model name).
    'POWER',  # 'POWER' (ticker) and 'Power Cement' (name) collide heavily with
              # generic global energy-sector news (NEPRA tariff hikes, Chinese
              # carbon policy, Belgian energy grids) that has nothing to do with
              # this specific cement company.
    'LOTCHEM',  # Lotte Chemical Pakistan's Korean parent ('Lotte Chemical Corp')
                # dominates global search results with its own Malaysia/Korea
                # unit's earnings, asset sales, and stock moves (XKRX, Bursa
                # Malaysia) - confirmed most "Lotte Chemical" hits are about the
                # Korean/Malaysian entities, not the Pakistani subsidiary.
}

# NOTE: demoting EVERY company's bare ticker to the generic/corroboration-
# required tier was tried and reverted - it broke genuinely correct matches for
# tickers that aren't actually generic-collision-prone (PSO, NBP, MCB, HBL all
# lost real news like "PSO hosts flag-hoisting ceremony" and "Cabinet seals
# NBP's new Chief" for lacking an explicit finance/Pakistan keyword). Ticker
# symbols stay trusted by default; only the specific confirmed-generic ones
# above get the stricter treatment.


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

        if _DENY_PATTERN.search(text_to_search):
            # Sports/entertainment content never counts as company news,
            # regardless of which term matched.
            return []

        if _CROSS_ENTITY_PATTERN.search(text_to_search):
            # A known unrelated namesake is explicitly mentioned - this article
            # is about that entity, not our PSX-listed company.
            return []

        has_corroboration = bool(_CONTEXT_PATTERN.search(text_to_search)) or bool(_ANCHOR_PATTERN.search(text_to_search))
        has_pakistan_anchor = bool(_ANCHOR_PATTERN.search(text_to_search))

        for ticker, metadata in COMPANIES.items():
            trusted_terms = [metadata.name, metadata.ticker] + metadata.search_keywords
            generic_terms = metadata.aliases

            if ticker in REQUIRES_PAKISTAN_ANCHOR:
                # Generic finance-context corroboration isn't enough here - the
                # colliding entity has its own genuine finance news too. Require
                # a real Pakistan/PSX anchor for every tier, including the name.
                if has_pakistan_anchor:
                    for term in trusted_terms + generic_terms:
                        pattern = r'\b' + re.escape(term.lower()) + r'\b'
                        if re.search(pattern, text_to_search):
                            matched_tickers.add(ticker)
                            break
                continue

            matched = False
            for term in trusted_terms:
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                if re.search(pattern, text_to_search):
                    matched = True
                    break

            if not matched:
                for term in generic_terms:
                    # Generic aliases/tickers (e.g. "National Bank", "AGP") need
                    # corroboration - a finance keyword or a Pakistan-specific
                    # anchor - since they're the ones that collide with
                    # unrelated same-named entities/events (see module
                    # docstring above).
                    pattern = r'\b' + re.escape(term.lower()) + r'\b'
                    if re.search(pattern, text_to_search) and has_corroboration:
                        matched = True
                        break

            if matched:
                matched_tickers.add(ticker)

        # Create separate article instances for each matched ticker
        matched_articles = []
        for ticker in matched_tickers:
            # We must create a new Pydantic model instance with the assigned ticker
            new_article = article.model_copy()
            new_article.ticker = ticker
            matched_articles.append(new_article)

        return matched_articles
