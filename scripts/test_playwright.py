from playwright.sync_api import sync_playwright


def test_fetch():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print("Navigating to forexlive...")
        response = page.goto("https://www.forexlive.com/", wait_until="networkidle")
        print("Status:", response.status)

        # Save HTML
        html = page.content()
        with open("data/raw/news/forexlive_test.html", "w", encoding="utf-8") as f:
            f.write(html)

        print("Saved HTML!")
        browser.close()

if __name__ == "__main__":
    test_fetch()
