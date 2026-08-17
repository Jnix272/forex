"""
Scrape ForexLive headlines via Playwright.

Primary: intercept homepage/API JSON responses.
Fallback: parse article links from the rendered DOM (site layout changes often).
"""

from __future__ import annotations

import argparse
import os
import re

import pandas as pd
from playwright.sync_api import sync_playwright


def _normalize_ts(ts: str) -> str:
    if not ts or len(str(ts)) < 10:
        return "1970-01-01T00:00:00Z"
    try:
        return pd.to_datetime(ts, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return str(ts)


def _currency_from_headline(headline: str) -> str:
    text = (headline or "").lower()
    if any(k in text for k in ("eur", "ecb", "euro")):
        return "EUR"
    if any(k in text for k in ("gbp", "boe", "pound", "sterling")):
        return "GBP"
    if any(k in text for k in ("jpy", "boj", "yen")):
        return "JPY"
    if any(k in text for k in ("aud", "rba")):
        return "AUD"
    if any(k in text for k in ("cad", "boc")):
        return "CAD"
    if any(k in text for k in ("nzd", "rbnz")):
        return "NZD"
    if any(k in text for k in ("chf", "snb")):
        return "CHF"
    return "USD"


def _row(headline: str, link: str, ts: str, seen: set, rows: list) -> None:
    if not headline or not link:
        return
    if "/news/" not in link and "/wire/" not in link:
        return
    full_url = "https://www.forexlive.com" + link if link.startswith("/") else link
    if full_url in seen:
        return
    seen.add(full_url)
    rows.append(
        {
            "timestamp_utc": _normalize_ts(ts),
            "event_type": "headline",
            "currency": _currency_from_headline(headline),
            "impact": "medium",
            "headline": headline.strip(),
            "actual": "",
            "forecast": "",
            "source": "forexlive_scraper",
            "url": full_url,
            "sentiment_score": "",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape ForexLive news via API + DOM")
    parser.add_argument("--test", action="store_true", help="Run in test mode (only 5 scrolls)")
    parser.add_argument("--out", type=str, default="data/raw/news/historical_news_forexlive.csv")
    args = parser.parse_args()

    all_data: list[dict] = []
    seen_urls: set[str] = set()

    def extract_articles(data):
        if isinstance(data, dict):
            if "title" in data and ("url" in data or "link" in data or "slug" in data):
                headline = data.get("title") or data.get("headline") or ""
                link = data.get("url") or data.get("link") or data.get("slug") or ""
                ts = (
                    data.get("publishedAt")
                    or data.get("createdAt")
                    or data.get("publishedDate")
                    or data.get("date")
                    or "1970-01-01T00:00:00Z"
                )
                _row(headline, link, ts, seen_urls, all_data)
            for v in data.values():
                extract_articles(v)
        elif isinstance(data, list):
            for item in data:
                extract_articles(item)

    def handle_response(response):
        try:
            ctype = (response.headers or {}).get("content-type", "")
            url = response.url.lower()
            if "json" not in ctype and "api" not in url and "graphql" not in url:
                return
            if any(k in url for k in ("article", "news", "wire", "graphql", "homepage", "category")):
                data = response.json()
                extract_articles(data)
        except Exception:
            pass

    def scrape_dom(page):
        """Fallback: collect /news/ anchors from the live DOM."""
        items = page.eval_on_selector_all(
            "a[href*='/news/'], a[href*='/wire/']",
            """els => els.map(a => ({
                href: a.getAttribute('href') || '',
                title: (a.getAttribute('title') || a.textContent || '').trim(),
                time: (a.closest('article,li,div')?.querySelector('time')?.getAttribute('datetime')) || ''
            }))""",
        )
        for it in items:
            href = it.get("href") or ""
            title = re.sub(r"\s+", " ", it.get("title") or "").strip()
            if len(title) < 12:
                continue
            _row(title, href, it.get("time") or "1970-01-01T00:00:00Z", seen_urls, all_data)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()
        page.on("response", handle_response)

        print("Navigating to ForexLive...", flush=True)
        page.goto("https://www.forexlive.com/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(4000)
        scrape_dom(page)
        print(f"After initial load: {len(all_data)} articles", flush=True)

        max_scrolls = 5 if args.test else 200
        stagnant = 0
        print("Scrolling to load more articles...", flush=True)
        for scrolls in range(max_scrolls):
            prev = len(all_data)
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2500)
            scrape_dom(page)
            print(f"Scroll {scrolls + 1}/{max_scrolls} | {len(all_data)} articles", flush=True)
            if len(all_data) == prev:
                stagnant += 1
                if stagnant >= 3:
                    print("No new articles for 3 scrolls - stopping.", flush=True)
                    break
            else:
                stagnant = 0

        browser.close()

    if not all_data:
        print("WARN: ForexLive returned 0 articles (site/API may have changed).", flush=True)
        return

    df = pd.DataFrame(all_data)
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True)
    df = df.drop_duplicates(subset=["url"]).sort_values("timestamp_utc")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} total unique rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
