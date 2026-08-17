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
