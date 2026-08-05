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
