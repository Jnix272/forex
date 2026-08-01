from bs4 import BeautifulSoup

def parse():
    with open('data/raw/news/forexlive_test.html', 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    
    # Try to find common article containers
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        if '/news/' in href and text and len(text) > 20:
            print(f"URL: {href}")
            print(f"Headline: {text}")
            # Try to find timestamp (e.g. <time> or something nearby)
            parent = a.parent
            for _ in range(3):
                if not parent: break
                time_tag = parent.find('time')
                if time_tag:
                    print(f"Time: {time_tag.get('datetime', time_tag.get_text(strip=True))}")
                    break
                parent = parent.parent
            print("-" * 50)

if __name__ == '__main__':
    parse()
