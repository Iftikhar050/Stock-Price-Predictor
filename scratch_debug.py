import traceback
from src.psx_predictor.db.connection import engine
import pandas as pd
from src.psx_predictor.data.feature_news_sentiment import generate_news_sentiment_features

try:
    generate_news_sentiment_features('PSO', pd.date_range('2020-01-01', '2020-01-10'), engine)
except Exception as e:
    traceback.print_exc()
