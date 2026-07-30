from datetime import datetime, timezone
from sqlalchemy import String, Float, BigInteger, Date, DateTime
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
