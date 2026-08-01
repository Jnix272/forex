import os
import argparse
import pandas as pd
from playwright.sync_api import sync_playwright

def main():
    parser = argparse.ArgumentParser(description="Scrape ForexLive news via API interception")
    parser.add_argument('--test', action='store_true', help="Run in test mode (only 5 scrolls)")
    parser.add_argument('--out', type=str, default='data/raw/news/historical_news_forexlive.csv')
    args = parser.parse_args()
    
    all_data = []
    seen_urls = set()
    
    def extract_articles(data):
        if isinstance(data, dict):
            # Check if this node looks like an article
            if 'title' in data and ('url' in data or 'link' in data):
                headline = data.get('title')
                link = data.get('url') or data.get('link') or ""
                
                # Check if it's actually an article and not a menu item
                if headline and link and "/news/" in link:
                    # Get timestamp
                    ts = data.get('publishedAt') or data.get('createdAt') or data.get('publishedDate') or "1970-01-01T00:00:00Z"
                    
                    if len(ts) >= 10:
                        try:
                            # Try to parse to our standard format
                            dt = pd.to_datetime(ts)
                            ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        except Exception:
                            pass
                            
                    currency = 'USD'
                    text_lower = headline.lower()
                    if 'eur' in text_lower or 'ecb' in text_lower: currency = 'EUR'
                    elif 'gbp' in text_lower or 'boe' in text_lower: currency = 'GBP'
                    elif 'jpy' in text_lower or 'boj' in text_lower: currency = 'JPY'
                    
                    full_url = 'https://www.forexlive.com' + link if link.startswith('/') else link
                    
                    if full_url not in seen_urls:
                        seen_urls.add(full_url)
                        all_data.append({
                            'timestamp_utc': ts,
                            'event_type': 'headline',
                            'currency': currency,
                            'impact': 'medium',
                            'headline': headline,
                            'actual': '',
                            'forecast': '',
                            'source': 'forexlive_scraper',
                            'url': full_url,
                        })
            
            # Recurse
            for k, v in data.items():
                extract_articles(v)
        elif isinstance(data, list):
            for item in data:
                extract_articles(item)
                
    def handle_response(response):
        if response.request.resource_type in ["fetch", "xhr"]:
            url = response.url
            if "/api/homepage/articles" in url or "/api/categories/get-all-news-video" in url or "graphql" in url:
                try:
                    data = response.json()
                    extract_articles(data)
                except Exception:
                    pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()
        
        page.on("response", handle_response)
        
        print("Navigating to ForexLive...")
        page.goto("https://www.forexlive.com/", wait_until="networkidle")
        page.wait_for_timeout(3000) # Initial wait for DOM and APIs
        
        max_scrolls = 5 if args.test else 1000
        scrolls = 0
        
        print("Scrolling to trigger API requests...")
        while scrolls < max_scrolls:
            prev_len = len(all_data)
            
            # Scroll to bottom
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(3000)
            
            new_len = len(all_data)
            print(f"Scroll {scrolls+1}/{max_scrolls} | Extracted {new_len} total articles so far.")
            
            if new_len == prev_len:
                print("No new articles loaded. Might have hit the end or rate limit.")
                # Wait a bit longer and try once more before breaking
                page.wait_for_timeout(5000)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(3000)
                if len(all_data) == prev_len:
                    print("Still no new articles. Stopping.")
                    break
                    
            scrolls += 1
            
        browser.close()
        
    if all_data:
        df = pd.DataFrame(all_data)
        out_path = args.out
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            df = pd.concat([existing, df])
            
        df = df.drop_duplicates(subset=['url']).sort_values('timestamp_utc')
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} total unique rows to {out_path}")

if __name__ == "__main__":
    main()
