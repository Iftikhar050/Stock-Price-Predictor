from src.psx_predictor.db.connection import engine
from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("UPDATE macro_indicators SET is_synthetic_rate = False"))
    conn.commit()
print("Updated database is_synthetic_rate to False")
