"""
cache.py — TTL cache helpers over the shared psx_cache dict (see state.py).

Extracted from the repeated inline
    if key in psx_cache:
        cached_time, data = psx_cache[key]
        if time.time() - cached_time < TTL:
            return data
pattern that used to appear independently in each endpoint.
"""
import time


def get_cached(cache: dict, key: str, ttl: float):
    """Return the cached value for `key` if present and younger than `ttl` seconds, else None."""
    entry = cache.get(key)
    if entry is None:
        return None
    cached_time, value = entry
    if time.time() - cached_time < ttl:
        return value
    return None


def set_cached(cache: dict, key: str, value) -> None:
    cache[key] = (time.time(), value)
