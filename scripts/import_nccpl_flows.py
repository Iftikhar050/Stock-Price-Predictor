"""
NCCPL FIPI Normal / LIPI Normal bulk downloader
------------------------------------------------
Setup (once):
    pip install playwright pandas lxml
    playwright install chrome

Run:
    python nccpl_download.py
    python nccpl_download.py --start 2016-12-09 --end 2026-11-10

How it deals with Cloudflare:
  * It drives a REAL, visible Chrome window with a persistent profile
    (./chrome_profile), so Cloudflare sees an ordinary browser.
  * If a "verify you are human" check shows up, solve it by hand once.
    The cookie is saved in the profile, so later runs usually skip it.
  * Requests are paced with small random delays to stay polite.

The script is resumable: files that already exist are skipped.
"""
import argparse
import random
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

URL = "https://www.nccpl.com.pk/market-information#tab-fipi-lipi-portfolio-0"
TABS = {"FIPI Normal": "fipi_normal", "LIPI Normal": "lipi_normal"}
OUT_DIR = Path("nccpl_data")
PROFILE_DIR = Path("chrome_profile")
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def pause(a=1.0, b=2.5):
    time.sleep(random.uniform(a, b))


def wait_until_site_ready(page, timeout_s=300):
    """Wait for the real page (tabs visible). Lets you solve Cloudflare manually."""
    print("Waiting for the page / Cloudflare check (solve it in the browser if shown)...")
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            if page.get_by_text("FIPI Normal", exact=True).first.is_visible():
                print("Page ready.")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Page never became ready (Cloudflare not passed?).")


def open_tab(page, name):
    page.get_by_text(name, exact=True).first.click()
    pause(1.5, 2.5)


def find_date_input(page):
    """The visible input whose value looks like MM/DD/YYYY (ignores the top search box)."""
    for inp in page.locator("input:visible").all():
        try:
            if DATE_RE.match(inp.input_value().strip()):
                return inp
        except Exception:
            continue
    raise RuntimeError("Could not find the date input.")


def set_date(page, d: date):
    inp = find_date_input(page)
    value = d.strftime("%m/%d/%Y")  # site shows 11/10/2026 = MM/DD/YYYY
    inp.evaluate(
        """(el, v) => {
            const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
            setter.call(el, v);
            el.dispatchEvent(new Event('input',  {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
            el.dispatchEvent(new Event('blur',   {bubbles: true}));
        }""",
        value,
    )
    return inp


def click_in_card(inp, text):
    """Click the button/link with `text` inside the same card as the date input."""
    card = inp.locator(
        f"xpath=ancestor::*[.//*[normalize-space()='Search'] and .//*[contains(normalize-space(),'Export')]][1]"
    )
    card.get_by_text(text, exact=False).first.click()


def download_one(page, tab_label, prefix, d: date):
    target_base = OUT_DIR / prefix / f"{prefix}_{d.isoformat()}"
    if any(target_base.parent.glob(target_base.name + ".*")):
        return "skip"

    inp = set_date(page, d)
    click_in_card(inp, "Search")
    pause(1.5, 3.0)

    # 1) Preferred: the site's own Export button
    try:
        with page.expect_download(timeout=20000) as dl_info:
            click_in_card(inp, "Export")
        dl = dl_info.value
        ext = Path(dl.suggested_filename).suffix or ".xlsx"
        dl.save_as(str(target_base) + ext)
        return "ok"
    except PWTimeout:
        pass

    # 2) Fallback: scrape the visible table (e.g. holidays / empty days will fail here)
    try:
        import pandas as pd
        from io import StringIO

        html = page.locator("table:visible").first.evaluate("el => el.outerHTML")
        df = pd.read_html(StringIO(html))[0]
        if df.empty:
            return "nodata"
        df.to_csv(str(target_base) + ".csv", index=False)
        return "ok-table"
    except Exception:
        return "nodata"


def daterange(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:  # skip Sat/Sun, markets are closed
            yield d
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-12-09")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    for p in TABS.values():
        (OUT_DIR / p).mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            str(PROFILE_DIR),
            channel="chrome",      # real Chrome, not bundled Chromium
            headless=False,        # headless is far more likely to be challenged
            accept_downloads=True,
            viewport={"width": 1400, "height": 900},
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded")
        wait_until_site_ready(page)

        for tab_label, prefix in TABS.items():
            print(f"\n=== {tab_label} ===")
            open_tab(page, tab_label)
            failures = []
            for d in daterange(start, end):
                try:
                    status = download_one(page, tab_label, prefix, d)
                except Exception as e:
                    status = f"error: {e.__class__.__name__}"
                    # recover: reload and reopen the tab
                    try:
                        page.goto(URL, wait_until="domcontentloaded")
                        wait_until_site_ready(page, 120)
                        open_tab(page, tab_label)
                    except Exception:
                        pass
                print(f"{tab_label} {d} -> {status}")
                if status.startswith("error"):
                    failures.append(d.isoformat())
                if status != "skip":
                    pause(1.0, 3.0)
            if failures:
                (OUT_DIR / f"{prefix}_failed_dates.txt").write_text("\n".join(failures))
                print(f"{len(failures)} failures logged; just re-run to retry them.")

        ctx.close()
    print("\nDone.")


if __name__ == "__main__":
    main() 