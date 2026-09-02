from datetime import datetime, timezone
from sqlalchemy import String, Float, BigInteger, Date, DateTime, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""
    pass

class StockEODData(Base):
    """
    SQLAlchemy ORM Model for End-of-Day (EOD) Stock Data.
    
    Table: stock_eod_data
    Primary Key: (ticker, date)
    """
    __tablename__ = "stock_eod_data"

    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    
    open: Mapped[float] = mapped_column(Float, nullable=True)
    high: Mapped[float] = mapped_column(Float, nullable=True)
    low: Mapped[float] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float, nullable=True)
    adjusted_close: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Adjusted closing price after corporate actions (rights, bonus, dividend). Source: PSX DPS.")
    volume: Mapped[int] = mapped_column(BigInteger, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        doc="Timestamp of when the record was inserted."
    )

    def __repr__(self) -> str:
        return f"<StockEODData(ticker='{self.ticker}', date='{self.date}', close={self.close})>"

class StockNews(Base):
    """
    SQLAlchemy ORM Model for Raw Financial News Articles.
    """
    __tablename__ = "stock_news"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    headline: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(100))
    author: Mapped[str] = mapped_column(String(100), nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=True)

    # Multi-label topic classification (PDF Groups 19/20/25/26)
    topic_category: Mapped[str] = mapped_column(String(50), nullable=True, index=True,
        doc="Canonical topic: CORPORATE, POLITICAL, GEOPOLITICAL, MACRO_ECONOMIC, SECTOR_SPECIFIC")
    # FinBERT probability triplet for downstream weighting
    finbert_pos: Mapped[float] = mapped_column(Float, nullable=True,
        doc="FinBERT positive probability [0-1].")
    finbert_neg: Mapped[float] = mapped_column(Float, nullable=True,
        doc="FinBERT negative probability [0-1].")
    finbert_neu: Mapped[float] = mapped_column(Float, nullable=True,
        doc="FinBERT neutral probability [0-1].")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc)
    )

class StockNewsSentiment(Base):
    """
    SQLAlchemy ORM Model for Aggregated Daily Sentiment.
    """
    __tablename__ = "stock_news_sentiment"

    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc)
    )

class StockDividend(Base):
    """
    SQLAlchemy ORM Model for Historical Dividend Payouts.
    """
    __tablename__ = "stock_dividends"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    
    announcement_date: Mapped[datetime.date] = mapped_column(Date, nullable=True)
    ex_dividend_date: Mapped[datetime.date] = mapped_column(Date, index=True)
    
    dividend_amount: Mapped[float] = mapped_column(Float, nullable=False)
    dividend_type: Mapped[str] = mapped_column(String(50), nullable=True) # e.g., 'Cash', 'Bonus', 'Right'
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc)
    )

class StockMetadata(Base):
    """
    SQLAlchemy ORM Model for Ticker Universe Metadata.
    """
    __tablename__ = "stock_metadata"

    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    company_name: Mapped[str] = mapped_column(String(200), nullable=True)
    sector: Mapped[str] = mapped_column(String(100), nullable=True)
    market_cap_tier: Mapped[str] = mapped_column(String(50), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    listed_date: Mapped[datetime.date] = mapped_column(Date, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc)
    )

class StockMarketIndex(Base):
    """
    SQLAlchemy ORM Model for Daily Market Index (e.g. KSE100).
    """
    __tablename__ = "stock_market_index"
    
    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    index_name: Mapped[str] = mapped_column(String(50), primary_key=True, default="KSE100")
    
    open: Mapped[float] = mapped_column(Float, nullable=True)
    high: Mapped[float] = mapped_column(Float, nullable=True)
    low: Mapped[float] = mapped_column(Float, nullable=True)
    close: Mapped[float] = mapped_column(Float, nullable=True)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=True)
    
    is_synthetic_index: Mapped[bool] = mapped_column(default=True, doc="Flag to indicate this is a proxy index computed from constituents, not real KSE-100 data.")
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc)
    )

class StockFundamentals(Base):
    """
    SQLAlchemy ORM Model for Quarterly/Annual Fundamentals.
    """
    __tablename__ = "stock_fundamentals"
    
    ticker: Mapped[str] = mapped_column(String(20), primary_key=True)
    report_date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    
    eps: Mapped[float] = mapped_column(Float, nullable=True)
    pe_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    roe: Mapped[float] = mapped_column(Float, nullable=True)
    debt_to_equity: Mapped[float] = mapped_column(Float, nullable=True)
    book_value_per_share: Mapped[float] = mapped_column(Float, nullable=True)
    
    revenue: Mapped[float] = mapped_column(Float, nullable=True)
    net_income: Mapped[float] = mapped_column(Float, nullable=True)
    free_cash_flow: Mapped[float] = mapped_column(Float, nullable=True)
    operating_cash_flow: Mapped[float] = mapped_column(Float, nullable=True)
    total_assets: Mapped[float] = mapped_column(Float, nullable=True)
    total_debt: Mapped[float] = mapped_column(Float, nullable=True)
    
    ebitda: Mapped[float] = mapped_column(Float, nullable=True)
    total_cash: Mapped[float] = mapped_column(Float, nullable=True)
    shares_outstanding: Mapped[float] = mapped_column(Float, nullable=True)
    
    eps_growth_yoy: Mapped[float] = mapped_column(Float, nullable=True)
    peg_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    gross_profit_margin: Mapped[float] = mapped_column(Float, nullable=True)
    net_profit_margin: Mapped[float] = mapped_column(Float, nullable=True)
    
    free_float: Mapped[float] = mapped_column(Float, nullable=True)
    free_float_pct: Mapped[float] = mapped_column(Float, nullable=True)
    market_cap: Mapped[float] = mapped_column(Float, nullable=True)
    
    insider_buy_shares_30d: Mapped[float] = mapped_column(Float, nullable=True)
    insider_sell_shares_30d: Mapped[float] = mapped_column(Float, nullable=True)
    insider_net_flow_30d: Mapped[float] = mapped_column(Float, nullable=True)
    sponsor_holding_pct: Mapped[float] = mapped_column(Float, nullable=True)
    institutional_holding_pct: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Priority 1: Banking Metrics (MEBL) & Energy/Fundamentals (PSO)
    gross_profit: Mapped[float] = mapped_column(Float, nullable=True)
    operating_profit: Mapped[float] = mapped_column(Float, nullable=True)
    roic: Mapped[float] = mapped_column(Float, nullable=True)
    revenue_growth: Mapped[float] = mapped_column(Float, nullable=True)
    profit_growth: Mapped[float] = mapped_column(Float, nullable=True)
    current_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    working_capital: Mapped[float] = mapped_column(Float, nullable=True)
    receivables: Mapped[float] = mapped_column(Float, nullable=True)
    inventory: Mapped[float] = mapped_column(Float, nullable=True)
    asset_growth: Mapped[float] = mapped_column(Float, nullable=True)
    payout_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Banking specific (MEBL)
    net_interest_margin: Mapped[float] = mapped_column(Float, nullable=True)
    casa_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    casa_deposits: Mapped[float] = mapped_column(Float, nullable=True)
    total_advances: Mapped[float] = mapped_column(Float, nullable=True)
    total_deposits: Mapped[float] = mapped_column(Float, nullable=True)
    npl_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    provisioning_coverage: Mapped[float] = mapped_column(Float, nullable=True)
    capital_adequacy_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    adr_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    idr_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    
    # Energy specific (PSO)
    circular_debt_level: Mapped[float] = mapped_column(Float, nullable=True)
    government_receivables: Mapped[float] = mapped_column(Float, nullable=True)
    refinery_margin: Mapped[float] = mapped_column(Float, nullable=True)
    petroleum_sales_volume: Mapped[float] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc)
    )

class MacroIndicators(Base):
    """
    SQLAlchemy ORM Model for Daily/Periodic Macroeconomic Data.
    """
    __tablename__ = "macro_indicators"
    
    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    
    sbp_policy_rate: Mapped[float] = mapped_column(Float, nullable=True)
    is_synthetic_rate: Mapped[bool] = mapped_column(default=False, doc="Flag to indicate this is a hardcoded placeholder SBP policy rate, not real historical data.")
    pkr_usd_rate: Mapped[float] = mapped_column(Float, nullable=True)
    brent_oil_price: Mapped[float] = mapped_column(Float, nullable=True)
    
    sp500_close: Mapped[float] = mapped_column(Float, nullable=True)
    nasdaq_close: Mapped[float] = mapped_column(Float, nullable=True)
    dxy_close: Mapped[float] = mapped_column(Float, nullable=True)
    us10y_yield: Mapped[float] = mapped_column(Float, nullable=True)
    
    gold_price: Mapped[float] = mapped_column(Float, nullable=True)
    copper_price: Mapped[float] = mapped_column(Float, nullable=True)
    coal_price: Mapped[float] = mapped_column(Float, nullable=True)
    cotton_price: Mapped[float] = mapped_column(Float, nullable=True)
    gas_price: Mapped[float] = mapped_column(Float, nullable=True)
    wti_oil_price: Mapped[float] = mapped_column(Float, nullable=True,
        doc="WTI Crude oil price (USD/bbl). Source: CL=F via yfinance. Distinct from brent_oil_price (BZ=F).")
    aluminum_price: Mapped[float] = mapped_column(Float, nullable=True,
        doc="LME Aluminum spot price (USD/MT). Source: ALI=F via yfinance.")
    wheat_price: Mapped[float] = mapped_column(Float, nullable=True)
    soybean_price: Mapped[float] = mapped_column(Float, nullable=True)
    
    nikkei_close: Mapped[float] = mapped_column(Float, nullable=True)
    hang_seng_close: Mapped[float] = mapped_column(Float, nullable=True)
    shanghai_close: Mapped[float] = mapped_column(Float, nullable=True)
    ftse_close: Mapped[float] = mapped_column(Float, nullable=True)
    dax_close: Mapped[float] = mapped_column(Float, nullable=True)
    vix_close: Mapped[float] = mapped_column(Float, nullable=True)
    
    # SBP EasyData Fields
    kibor_3m: Mapped[float] = mapped_column(Float, nullable=True)
    kibor_6m: Mapped[float] = mapped_column(Float, nullable=True)
    kibor_1y: Mapped[float] = mapped_column(Float, nullable=True)
    
    tbill_3m: Mapped[float] = mapped_column(Float, nullable=True)
    tbill_6m: Mapped[float] = mapped_column(Float, nullable=True)
    tbill_1y: Mapped[float] = mapped_column(Float, nullable=True)
    
    pib_3y: Mapped[float] = mapped_column(Float, nullable=True)
    pib_5y: Mapped[float] = mapped_column(Float, nullable=True)
    pib_10y: Mapped[float] = mapped_column(Float, nullable=True)
    
    cpi_headline: Mapped[float] = mapped_column(Float, nullable=True)
    cpi_core: Mapped[float] = mapped_column(Float, nullable=True)
    
    sbp_reserves: Mapped[float] = mapped_column(Float, nullable=True)
    commercial_bank_reserves: Mapped[float] = mapped_column(Float, nullable=True)
    total_fx_reserves: Mapped[float] = mapped_column(Float, nullable=True)
    
    monthly_remittances: Mapped[float] = mapped_column(Float, nullable=True)
    remittances_saudi: Mapped[float] = mapped_column(Float, nullable=True)
    remittances_uae: Mapped[float] = mapped_column(Float, nullable=True)
    remittances_usa: Mapped[float] = mapped_column(Float, nullable=True)
    remittances_uk: Mapped[float] = mapped_column(Float, nullable=True)
    
    m2_money_supply: Mapped[float] = mapped_column(Float, nullable=True)
    currency_in_circulation: Mapped[float] = mapped_column(Float, nullable=True)
    
    current_account_balance: Mapped[float] = mapped_column(Float, nullable=True)
    trade_deficit: Mapped[float] = mapped_column(Float, nullable=True)
    
    advancing_stocks_pct: Mapped[float] = mapped_column(Float, nullable=True)
    declining_stocks_pct: Mapped[float] = mapped_column(Float, nullable=True)
    market_breadth_ratio: Mapped[float] = mapped_column(Float, nullable=True)
    
    fipi_foreign_corporate_net: Mapped[float] = mapped_column(Float, nullable=True)
    fipi_foreign_individual_net: Mapped[float] = mapped_column(Float, nullable=True)
    fipi_overseas_pakistani_net: Mapped[float] = mapped_column(Float, nullable=True)
    lipi_mutual_funds_net: Mapped[float] = mapped_column(Float, nullable=True)
    lipi_banks_net: Mapped[float] = mapped_column(Float, nullable=True)
    lipi_insurance_net: Mapped[float] = mapped_column(Float, nullable=True)
    lipi_companies_net: Mapped[float] = mapped_column(Float, nullable=True)
    lipi_individuals_net: Mapped[float] = mapped_column(Float, nullable=True)
    
    kmi30_index_level: Mapped[float] = mapped_column(Float, nullable=True)
    kmi30_return_pct: Mapped[float] = mapped_column(Float, nullable=True)
    kse30_index_level: Mapped[float] = mapped_column(Float, nullable=True)
    kse30_return_pct: Mapped[float] = mapped_column(Float, nullable=True)
    all_share_index_level: Mapped[float] = mapped_column(Float, nullable=True)
    all_share_return_pct: Mapped[float] = mapped_column(Float, nullable=True)
    banking_sector_index_level: Mapped[float] = mapped_column(Float, nullable=True)
    banking_sector_return_pct: Mapped[float] = mapped_column(Float, nullable=True)
    oil_gas_sector_index_level: Mapped[float] = mapped_column(Float, nullable=True)
    oil_gas_sector_return_pct: Mapped[float] = mapped_column(Float, nullable=True)
    
    imf_real_gdp_growth: Mapped[float] = mapped_column(Float, nullable=True)
    imf_cpi_inflation: Mapped[float] = mapped_column(Float, nullable=True)
    imf_govt_gross_debt_pct_gdp: Mapped[float] = mapped_column(Float, nullable=True)
    imf_current_account_balance_pct_gdp: Mapped[float] = mapped_column(Float, nullable=True)
    imf_govt_revenue_pct_gdp: Mapped[float] = mapped_column(Float, nullable=True)
    imf_govt_expenditure_pct_gdp: Mapped[float] = mapped_column(Float, nullable=True)
    imf_govt_fiscal_balance_pct_gdp: Mapped[float] = mapped_column(Float, nullable=True)
    imf_primary_balance_pct_gdp: Mapped[float] = mapped_column(Float, nullable=True)
    imf_investment_pct_gdp: Mapped[float] = mapped_column(Float, nullable=True)
    imf_national_savings_pct_gdp: Mapped[float] = mapped_column(Float, nullable=True)
    imf_unemployment_rate: Mapped[float] = mapped_column(Float, nullable=True)
    imf_export_volume_growth: Mapped[float] = mapped_column(Float, nullable=True)
    imf_import_volume_growth: Mapped[float] = mapped_column(Float, nullable=True)
    imf_gdp_usd_billions: Mapped[float] = mapped_column(Float, nullable=True)
    imf_sdr_allocation_bal: Mapped[float] = mapped_column(Float, nullable=True)
    imf_sdr_holdings_bal: Mapped[float] = mapped_column(Float, nullable=True)
    imf_total_loans_outstanding: Mapped[float] = mapped_column(Float, nullable=True)
    imf_quota_sdrs: Mapped[float] = mapped_column(Float, nullable=True)
    imf_tranche_disbursements: Mapped[float] = mapped_column(Float, nullable=True)
    imf_net_financial_position: Mapped[float] = mapped_column(Float, nullable=True)


    # Google Trends search interest (0-100 scale, weekly, forward-filled to daily)
    # Source: Google Trends via pytrends. Frequency: Weekly (Google Trends minimum).
    search_trend_pso: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Google Trends weekly search interest for 'PSO Pakistan' in Pakistan (0-100 scale).")
    search_trend_mebl: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Google Trends weekly search interest for 'Meezan Bank' in Pakistan (0-100 scale).")
    search_trend_kse: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Google Trends weekly search interest for 'KSE 100' in Pakistan (0-100 scale).")

    # PSX Market Statistics (PDF Groups #18)
    # Source: PSX DPS daily market statistics page
    market_total_volume: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Total market trading volume (shares). Source: PSX DPS market statistics.")
    market_total_traded_value: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Total market traded value (PKR). Source: PSX DPS market statistics.")
    market_number_of_trades: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Total number of trades across all securities. Source: PSX DPS market statistics.")
    new_highs: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Number of securities hitting 52-week highs. Source: PSX DPS.")
    new_lows: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Number of securities hitting 52-week lows. Source: PSX DPS.")
    advancing_count: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Number of advancing securities. Source: PSX DPS.")
    declining_count: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Number of declining securities. Source: PSX DPS.")
    sector_breadth: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Market breadth ratio: (advancing - declining) / total. Source: Computed from PSX DPS.")

    # ── TIER-1 ADDITIONS: Global Equity Markets (PDF Groups #7) ─────────────────
    # MSCI EM & FM were previously downloaded in macro_scraper.py but NOT mapped to DB columns.
    # Fixed: now properly persisted.
    msci_em_close: Mapped[float] = mapped_column(Float, nullable=True,
        doc="MSCI Emerging Markets ETF (EEM) daily close. Source: yfinance.")
    msci_fm_close: Mapped[float] = mapped_column(Float, nullable=True,
        doc="MSCI Frontier Markets ETF (FM) daily close. Source: yfinance.")
    dow_jones_close: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Dow Jones Industrial Average daily close. Source: yfinance ^DJI.")

    # ── TIER-1 ADDITIONS: FX & Currency Risks (PDF Groups #5) ───────────────────
    eur_pkr_rate: Mapped[float] = mapped_column(Float, nullable=True,
        doc="EUR/PKR exchange rate. Source: EURPKR=X via yfinance.")
    gbp_pkr_rate: Mapped[float] = mapped_column(Float, nullable=True,
        doc="GBP/PKR exchange rate. Source: GBPPKR=X via yfinance.")
    cny_pkr_rate: Mapped[float] = mapped_column(Float, nullable=True,
        doc="CNY/PKR derived cross rate (CNYUSD * USDPKR). Source: yfinance cross computation.")
    pkr_usd_volatility_20d: Mapped[float] = mapped_column(Float, nullable=True,
        doc="20-day rolling std of daily PKR/USD % change. Computed from pkr_usd_rate.")
    dxy_volatility_20d: Mapped[float] = mapped_column(Float, nullable=True,
        doc="20-day rolling std of daily DXY % change. Computed from dxy_close.")

    # ── TIER-1 ADDITIONS: Commodity Prices (PDF Group #4) ──────────────────────
    steel_price: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Steel ETF proxy (SPDR S&P Metals & Mining, SLX). Source: yfinance.")
    iron_ore_price: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Iron ore proxy via Vale S.A. (VALE) stock price. Source: yfinance.")
    palm_oil_price: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Palm Oil continuous contract (POO.L). Source: yfinance London.")
    urea_price: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Fertilizer/Urea proxy via Mosaic Co. (MOS). Source: yfinance.")
    lng_price: Mapped[float] = mapped_column(Float, nullable=True,
        doc="LNG proxy via ProShares Ultra DJ-AIG Natural Gas 2x (BOIL). Source: yfinance.")

    # ── TIER-1 ADDITIONS: Global Economic Indicators (PDF Group #8) ─────────────
    us2y_yield: Mapped[float] = mapped_column(Float, nullable=True,
        doc="US 2-Year Yield proxy via 13-Week T-Bill rate (^IRX). Source: yfinance.")
    us5y_yield: Mapped[float] = mapped_column(Float, nullable=True,
        doc="US 5-Year Treasury Yield (^FVX). Source: yfinance.")
    tips_etf_price: Mapped[float] = mapped_column(Float, nullable=True,
        doc="TIPS ETF price (SCHP) as real yield proxy. Source: yfinance.")
    ecb_rate: Mapped[float] = mapped_column(Float, nullable=True,
        doc="ECB Main Refinancing Rate step-function from official ECB decisions. Forward-filled daily.")
    fed_funds_rate: Mapped[float] = mapped_column(Float, nullable=True,
        doc="US Federal Funds Rate (upper bound) from FOMC decisions. Forward-filled daily.")

    # ── TIER-1 COMPUTED: Spreads & Derived (PDF Groups #3, #8) ──────────────────
    us_yield_curve_2y10y: Mapped[float] = mapped_column(Float, nullable=True,
        doc="US 2s10s yield curve spread (us10y_yield - us2y_yield). Recession indicator.")
    fed_ecb_policy_spread: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Fed Funds Rate minus ECB Main Refinancing Rate. Measures G2 policy divergence.")

    # ── TIER-1 COMPUTED: Geopolitical & Supply Shock Flags (PDF Groups #12) ─────
    global_oil_supply_shock_flag: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="Binary: 1 when 5-day WTI return > 2.5 std of trailing 252-day distribution. No look-ahead.")
    red_sea_disruption_flag: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="Binary: 1 from Dec 19 2023 onwards (Houthi shipping attacks on Red Sea). Structural event.")

    # ── TIER-2 ADDITIONS: SBP EasyData Additional Series ───────────────────────
    private_sector_credit_growth: Mapped[float] = mapped_column(Float, nullable=True,
        doc="YoY % growth in private sector credit outstanding. Source: SBP EasyData. Monthly, lagged 1m.")
    banking_deposits_growth: Mapped[float] = mapped_column(Float, nullable=True,
        doc="YoY % growth in total banking deposits. Source: SBP EasyData. Monthly, lagged 1m.")
    sbp_omo_net_outstanding: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Net outstanding SBP OMO injections (PKR billions). Source: SBP EasyData. Weekly.")
    t_bill_cutoff_3m: Mapped[float] = mapped_column(Float, nullable=True,
        doc="3-Month T-Bill auction cutoff yield (%). Source: SBP EasyData. Weekly.")
    t_bill_cutoff_6m: Mapped[float] = mapped_column(Float, nullable=True,
        doc="6-Month T-Bill auction cutoff yield (%). Source: SBP EasyData. Weekly.")
    forward_usd_pkr_3m: Mapped[float] = mapped_column(Float, nullable=True,
        doc="3-Month USD/PKR forward rate (PKR per USD). Source: SBP EasyData. Monthly.")
    reer_index: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Real Effective Exchange Rate index (base 2010=100). Source: SBP EasyData. Monthly, lagged 1m.")
    external_debt_total_usd_bn: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Total external debt (USD billions). Source: SBP EasyData. Quarterly, lagged 1m.")
    sbp_additional_is_synthetic: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="1 if any SBP additional series used synthetic fallback data for this row.")

    # ── TIER-3 ADDITIONS: Pakistan Real Economy Activity (PDF Group #10) ────────
    cement_dispatches_mt: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Monthly cement dispatches (thousand tonnes). Source: APCMA. Monthly, lagged 1m.")
    auto_sales_total: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Monthly automobile total sales (units). Source: PAMA. Monthly, lagged 1m.")
    electricity_gen_gwh: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Monthly electricity generation (GWh). Source: NEPRA. Monthly, lagged 1m.")
    wheat_procurement_mt: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Monthly wheat procurement (thousand tonnes). Source: MNFSR/PASSCO. Seasonal (Apr-Jun).")
    pakistan_activity_is_synthetic: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="1 if Pakistan real economy activity data used synthetic fallback for this row.")

    # ── TIER-4 ADDITIONS: Political & Geopolitical Flags (PDF Groups #11, #12) ──
    election_flag: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="Binary: 1 within ±90 days of a Pakistan general election date. Source: ECP reference dates.")
    fatf_greylist_flag: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="Binary: 1 during FATF grey-listing of Pakistan. Windows: 2008-2015, 2018-2022, 2025-present.")
    government_stability_score: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="Ordinal: 0=caretaker/coup, 1=coalition/minority, 2=majority. Source: Pakistan constitutional record.")
    political_uncertainty_score: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="Ordinal 0–3 composite: stability + election + tension + FATF score. Higher = more uncertain.")
    india_pakistan_tension_flag: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="Binary: 1 during documented India-Pakistan military/diplomatic escalation windows.")
    middle_east_conflict_flag: Mapped[int] = mapped_column(Integer, nullable=True,
        doc="Binary: 1 during major Middle East conflict escalation periods (Iraq, Gaza, ISIS, Iran-Israel).")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )






class CorporateAnnouncementsPUCARS(Base):
    """
    Stores raw textual corporate announcements, PSX PUCARS notices, press releases,
    and board meeting decisions alongside categorized metadata and sentiment scores.
    """
    __tablename__ = "corporate_announcements_pucars"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    announcement_date: Mapped[datetime.date] = mapped_column(Date, index=True, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    headline_raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    body_raw_text: Mapped[str] = mapped_column(Text, nullable=True)
    document_url: Mapped[str] = mapped_column(String(500), nullable=True)
    source: Mapped[str] = mapped_column(String(100), default="PSX PUCARS")
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


class CorporateEvent(Base):
    """
    Structured corporate event records derived from PUCARS notices.
    One row per notice, post trading-day-cutoff shift.
    Independent of stock_news (structured events, not free-text news).
    Closes PDF Groups 25 (Corporate Announcements) and 36 (News/Event Shocks).
    """
    __tablename__ = "corporate_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    trading_date: Mapped[datetime.date] = mapped_column(Date, index=True, nullable=False,
        doc="Post trading-day-cutoff date. News after 16:00 PKT shifts to T+1.")
    event_type: Mapped[str] = mapped_column(String(50), index=True, nullable=True,
        doc="Matches *_event feature column name (e.g. 'earnings_event', 'dividend_event'). Nullable if unclassified.")
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(String(500), nullable=True)
    source: Mapped[str] = mapped_column(String(100), default="PUCARS")
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=True, default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )


class TopicSentimentDaily(Base):
    """
    Daily aggregated sentiment per topic category.
    Stores political, geopolitical, sector, and macro news sentiment metrics.
    Closes PDF Groups 19, 20, 26, 40.
    """
    __tablename__ = "topic_sentiment_daily"

    date: Mapped[datetime.date] = mapped_column(Date, primary_key=True)
    topic: Mapped[str] = mapped_column(String(50), primary_key=True,
        doc="Topic category: CORPORATE, POLITICAL, GEOPOLITICAL, MACRO_ECONOMIC, SECTOR_SPECIFIC")
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sentiment_std: Mapped[float] = mapped_column(Float, nullable=True,
        doc="Standard deviation of article sentiments for this topic-day. Measures sentiment dispersion.")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
