"""
state.py — Shared, process-wide mutable state for the API.

Single source of truth for ml_models and psx_cache so every router (and the
app's lifespan handler in main.py) mutate/read the exact same objects.
"""

# Loaded ML models/scalers/category-maps, populated by main.py's lifespan handler.
ml_models: dict = {}

# Generic response cache. Two conventions coexist here (both pre-date this
# module and are left as-is):
#   - TTL-style entries: cache[key] = (timestamp, value) — see cache.py helpers.
#   - mtime-equality entries (used only by /api/predict): cache[key] = (file_mtime, value).
psx_cache: dict = {}
