import json

from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        page = context.new_page()

        def handle_response(response):
            if "/api/homepage/articles" in response.url:
                with open("data/raw/news/forexlive_api_dump.json", "w") as f:
                    f.write(json.dumps(response.json(), indent=2))
                print("Dumped!")

        page.on("response", handle_response)
        page.goto("https://www.forexlive.com/", wait_until="networkidle")
        browser.close()


if __name__ == "__main__":
    main()
