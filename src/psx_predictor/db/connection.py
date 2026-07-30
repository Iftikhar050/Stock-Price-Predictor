import os
import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Setup module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def get_database_url() -> str:
    """Constructs the PostgreSQL database URL from environment variables."""
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "psx_predictor")

    if not password:
        logger.warning("DB_PASSWORD is not set in the environment variables.")
        
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"

try:
    db_url = get_database_url()
    
    # Create the SQLAlchemy 2.0 Engine
    engine = create_engine(
        db_url,
        echo=False,  # Set to True to log all SQL queries
        pool_pre_ping=True,  # Verifies connections before usage
        pool_size=10,
        max_overflow=20
    )
    
    # Create a configured "Session" factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create a scoped session for thread-safety during parallel processing/requests
    db_session = scoped_session(SessionLocal)
    
    logger.info("Database engine and session factory created successfully.")
except Exception as e:
    logger.error(f"Failed to initialize database connection: {e}")
    raise

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get a database session and ensure it's closed after use.
    Can be used in context managers or FastAPI dependencies.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
