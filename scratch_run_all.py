import logging
import os
import sys

from src.psx_predictor.data.fetch_pakistan_activity import fetch_pakistan_activity
from src.psx_predictor.data.fetch_sbp_additional import fetch_sbp_additional
from src.psx_predictor.data.feature_political_events import build_political_features
from src.psx_predictor.data.build_features import build_features

logging.basicConfig(level=logging.INFO)

print("Running extra pipelines...")
fetch_pakistan_activity()
fetch_sbp_additional()
build_political_features()

print("Building features...")
build_features('PSO')
build_features('MEBL')

print("All done!")
