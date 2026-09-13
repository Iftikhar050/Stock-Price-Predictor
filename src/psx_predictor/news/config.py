from typing import Dict
from .models import CompanyMetadata

# System Configuration
NEWS_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Dictionary of monitored companies
COMPANIES: Dict[str, CompanyMetadata] = {
    "PSO": CompanyMetadata(
        ticker="PSO",
        name="Pakistan State Oil Company Limited",
        aliases=["Pakistan State Oil", "PSO"],
        search_keywords=["PSO stock", "Pakistan State Oil financial", "PSO earnings"]
    ),
    "FFC": CompanyMetadata(
        ticker="FFC",
        name="Fauji Fertilizer Company Limited",
        aliases=["Fauji Fertilizer", "FFC"],
        search_keywords=["Fauji Fertilizer stock", "FFC dividend", "FFC earnings"]
    ),
    "NBP": CompanyMetadata(
        ticker="NBP",
        name="National Bank of Pakistan",
        aliases=["National Bank", "NBP"],
        search_keywords=["National Bank of Pakistan news", "NBP stock", "NBP earnings"]
    ),
    "MEBL": CompanyMetadata(
        ticker="MEBL",
        name="Meezan Bank Limited",
        aliases=["Meezan Bank", "MEBL"],
        search_keywords=["Meezan Bank stock", "MEBL earnings", "Meezan Bank dividend"]
    ),
    "OGDC": CompanyMetadata(
        ticker="OGDC",
        name="Oil & Gas Development Company Limited",
        aliases=["Oil & Gas Development Company", "OGDC", "Oil and Gas Development"],
        search_keywords=["OGDC stock", "Oil and Gas Development Company news"]
    ),
    "LUCK": CompanyMetadata(
        ticker="LUCK",
        name="Lucky Cement Limited",
        aliases=["Lucky Cement", "LUCK"],
        search_keywords=["Lucky Cement stock", "LUCK earnings", "Lucky Cement news"]
    )
}

def _load_active_companies() -> None:
    """
    Extends COMPANIES with every other active ticker from stock_metadata, so
    news collection/matching covers the full active universe (e.g. all of
    KSE-100) instead of only the 6 hand-curated names above. Curated entries
    are left untouched; everyone else gets a simple alias/keyword set built
    from their company name.
    """
    try:
        from sqlalchemy import text
        from src.psx_predictor.db.connection import engine
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT ticker, company_name FROM stock_metadata WHERE is_active = true AND company_name IS NOT NULL"
            )).fetchall()
        for ticker, company_name in rows:
            ticker = ticker.upper()
            if ticker in COMPANIES:
                continue
            COMPANIES[ticker] = CompanyMetadata(
                ticker=ticker,
                name=company_name,
                aliases=[company_name, ticker],
                search_keywords=[f"{company_name} stock", f"{company_name} earnings"],
            )
    except Exception:
        # DB may not be reachable at import time in some contexts (e.g. offline
        # unit tests) - fall back to the curated list above only.
        pass

_load_active_companies()
