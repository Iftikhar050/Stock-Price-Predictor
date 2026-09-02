from src.psx_predictor.db.connection import engine
from sqlalchemy import text
from src.psx_predictor.pipelines.macro_pipeline import run_macro_pipeline
import logging

logging.basicConfig(level=logging.INFO)
with engine.connect() as conn:
    conn.execute(text("TRUNCATE macro_indicators"))
    conn.commit()
    print("Truncated macro_indicators.")

print("Running macro pipeline...")
run_macro_pipeline()
print("Done.")
