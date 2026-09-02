from src.psx_predictor.db.connection import engine
import pandas as pd
print(pd.read_sql("SELECT topic, count(*) FROM topic_sentiment_daily GROUP BY topic", engine))
