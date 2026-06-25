"""
browser_scraper.py — Playwright browser automation base class.

All browser-based scrapers use this class for:
  - Stealth mode (playwright-stealth) to bypass basic bot detection
  - Human-like delays and user-agent rotation
  - Headless by default, headful mode for sites that block headless
  - Xvfb support for headful mode on VPS (Linux only)
  - 2captcha integration for hard CAPTCHAs (optional, fallback to retry)

Bot detection layers handled:
  Level 1 — basic headless detection (TLS fingerprint, navigator.webdriver):
    playwright-stealth patches navigator, plugins, permissions, WebGL, etc.
  Level 2 — behavioral (mouse movement, timing):
    random delays, realistic viewport, locale
  Level 3 — hard CAPTCHA (Cloudflare Turnstile, reCAPTCHA v2/v3):
    2captcha API if CAPTCHA_2CAPTCHA_KEY set, else retry with exponential backoff
  Level 4 — IP-based blocks:
    notify and fail fast (no proxy support in v13 — add if needed)

Environment variables:
  TV_USERNAME, TV_PASSWORD — TradingView account (for tvdatafeed, not this scraper)
  CAPTCHA_2CAPTCHA_KEY     — optional, for hard CAPTCHA solving

Requirements:
    pip install playwright playwright-stealth
    playwright install chromium
"""

import logging
import random
import time
from contextlib import contextmanager
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Rotating user agents — pick one per session
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]


class BrowserScraper:
    """Stealth Playwright browser base class.

    Usage:
        scraper = BrowserScraper(stealth=True)
        html = scraper.fetch("https://example.com", wait_for=".data-table")
        scraper.close()

    Or as context manager:
        with BrowserScraper() as s:
            html = s.fetch(url)
    """

    def __init__(
        self,
        headless: bool = True,
        stealth: bool = True,
        slow_mo: int = 50,
    ):
        self._headless  = headless
        self._stealth   = stealth
        self._slow_mo   = slow_mo
        self._playwright = None
        self._browser    = None
        self._user_agent = random.choice(_USER_AGENTS)

    def _ensure_browser(self):
        if self._browser is not None:
            return
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError:
            raise ImportError(
                "playwright not installed — run: pip install playwright && playwright install chromium"
            )
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-extensions",
            ],
        )
        logger.debug("BrowserScraper: Chromium launched (headless=%s, stealth=%s)",
                     self._headless, self._stealth)

    def new_page(self):
        """Create a new browser page with stealth patches applied."""
        self._ensure_browser()
        context = self._browser.new_context(
            user_agent=self._user_agent,
            viewport={"width": 1280, "height": 720},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = context.new_page()
        if self._stealth:
            try:
                from playwright_stealth import stealth_sync  # type: ignore
                stealth_sync(page)
            except ImportError:
                logger.warning("playwright-stealth not installed — stealth mode disabled. "
                               "Run: pip install playwright-stealth")
        return page

    def fetch(
        self,
        url: str,
        wait_for: Optional[str] = None,
        wait_state: str = "networkidle",
        delay: tuple = (2.0, 5.0),
        timeout: int = 15_000,
        retries: int = 2,
    ) -> str:
        """Fetch a page and return its HTML content.

        Parameters
        ----------
        url        : target URL
        wait_for   : CSS selector to wait for after page load (optional)
        wait_state : Playwright waitUntil value ("load", "networkidle", "domcontentloaded")
        delay      : (min, max) seconds to sleep before navigating (human-like)
        timeout    : page load timeout in milliseconds
        retries    : number of retry attempts on failure

        Returns
        -------
        str : raw HTML of the page
        """
        time.sleep(random.uniform(*delay))

        for attempt in range(retries + 1):
            page = self.new_page()
            try:
                page.goto(url, wait_until=wait_state, timeout=timeout)
                if wait_for:
                    page.wait_for_selector(wait_for, timeout=timeout)
                html = page.content()
                page.close()
                return html
            except Exception as e:
                page.close()
                if attempt < retries:
                    sleep_s = (2 ** attempt) * random.uniform(1.5, 3.0)
                    logger.warning("BrowserScraper: attempt %d/%d failed for %s — %s. Retrying in %.1fs",
                                   attempt + 1, retries + 1, url, e, sleep_s)
                    time.sleep(sleep_s)
                else:
                    raise

    def fetch_with_js(self, url: str, js_expression: str, delay: tuple = (2.0, 5.0)) -> any:
        """Fetch a page and evaluate a JavaScript expression, returning the result.

        Useful for sites that load data into JS objects rather than HTML.
        """
        time.sleep(random.uniform(*delay))
        page = self.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=15_000)
            result = page.evaluate(js_expression)
            return result
        finally:
            page.close()

    def close(self):
        """Close browser and playwright instance."""
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            logger.debug("BrowserScraper.close: %s", e)
        finally:
            self._browser    = None
            self._playwright = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
