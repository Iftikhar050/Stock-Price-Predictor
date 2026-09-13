"""
json_safe.py — Some DB numeric columns store NaN instead of NULL (a pre-existing
data-quality quirk, not something this API layer should paper over silently, but
JSON has no NaN literal so it must become None or every affected endpoint 500s).
"""
import math


def safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value
