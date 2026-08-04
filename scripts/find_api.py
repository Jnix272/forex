import time

from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()

        print("Listening for API requests...")

        def handle_response(response):
            # We are looking for the infinite scroll endpoint
            if response.request.resource_type in ["fetch", "xhr", "document"]:
                url = response.url
                if "/news/" in url or "api" in url or "graphql" in url or "scroll" in url or "page" in url:
                    print(f"[{response.status}] {url}")

        page.on("response", handle_response)

        page.goto("https://www.forexlive.com/", wait_until="networkidle")

        print("Scrolling down to trigger infinite scroll...")
        for i in range(3):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(3)

        print("Done.")
        browser.close()

if __name__ == "__main__":
    main()
