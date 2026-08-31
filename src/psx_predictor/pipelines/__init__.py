"""
PSX Predictor Domain Data Ingestion Pipelines
"""
from src.psx_predictor.pipelines.news_sentiment_pipeline import run_news_sentiment_pipeline
from src.psx_predictor.pipelines.macro_pipeline import run_macro_pipeline
from src.psx_predictor.pipelines.institutional_flows_pipeline import run_institutional_flows_pipeline
from src.psx_predictor.pipelines.fundamentals_pipeline import run_fundamentals_pipeline
from src.psx_predictor.pipelines.orchestrator import run_full_data_pipeline

__all__ = [
    "run_news_sentiment_pipeline",
    "run_macro_pipeline",
    "run_institutional_flows_pipeline",
    "run_fundamentals_pipeline",
    "run_full_data_pipeline",
]
