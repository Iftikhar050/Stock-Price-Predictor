from src.psx_predictor.news.aggregator import NewsAggregator
import logging

logging.basicConfig(level=logging.INFO)
agg = NewsAggregator(use_finbert=False)
# Remove google news collector to speed up
agg.collectors = [c for c in agg.collectors if "Google" not in c.__class__.__name__]
agg.run_pipeline()
