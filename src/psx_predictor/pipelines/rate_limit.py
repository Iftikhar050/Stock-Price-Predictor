"""Shared per-ticker throttle for pipeline loops that hit external scrapers with no
built-in rate limiting of their own. Only matters at batch scale (staged ingestion of
~100 tickers per phase); harmless no-op-ish delay for the existing 2-ticker PSO/MEBL runs.
"""
import os
import time

DELAY_SECONDS = float(os.environ.get("PIPELINE_TICKER_DELAY_SEC", "1.5"))


def throttle():
    time.sleep(DELAY_SECONDS)
