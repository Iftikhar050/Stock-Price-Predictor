"""
http_client.py — Shared retry/backoff HTTP client for scraping dps.psx.com.pk.

No retry/backoff existed anywhere in the codebase before this; every scrape
call was a bare requests.get/post. This is used by the endpoints that still
need to scrape live (company profile, realtime price) after market_performers
was made DB-backed.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0"}

_session = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def get_with_retry(url: str, timeout: float = 10, headers: dict | None = None, **kwargs) -> requests.Response:
    """GET with automatic retry/backoff on connection errors and 429/5xx responses."""
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}
    return _session.get(url, headers=merged_headers, timeout=timeout, **kwargs)


def post_with_retry(url: str, timeout: float = 10, headers: dict | None = None, **kwargs) -> requests.Response:
    """POST with automatic retry/backoff on connection errors and 429/5xx responses."""
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}
    return _session.post(url, headers=merged_headers, timeout=timeout, **kwargs)
