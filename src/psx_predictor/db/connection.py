import os
import logging
from typing import Generator
from sqlalchemy import URL, create_engine
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

def get_database_url() -> URL:
    """Constructs the PostgreSQL database URL from environment variables.
    URL.create escapes each part, so a password containing @ : / still works."""
    password = os.getenv("DB_PASSWORD", "")
    if not password:
        logger.warning("DB_PASSWORD is not set in the environment variables.")

    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER", "postgres"),
        password=password,
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "psx_predictor"),
    )

try:
    db_url = get_database_url()
    
    # Create the SQLAlchemy 2.0 Engine
    # Hosted Postgres (Supabase) wants SSL and caps connections per client on
    # its pooler, so both are overridable; local defaults are unchanged.
    sslmode = os.getenv("DB_SSLMODE")
    engine = create_engine(
        db_url,
        echo=False,  # Set to True to log all SQL queries
        pool_pre_ping=True,  # Verifies connections before usage
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
        connect_args={"sslmode": sslmode} if sslmode else {},
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
