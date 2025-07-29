# tools/job_scraper.py

import requests
from bs4 import BeautifulSoup

class Job_Platform:
    def __init__(self, url):
        self.url = url
        self.title = ""
        self.description = ""
        self._scrape()

    def _scrape(self):
        try:
            response = requests.get(self.url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Attempt to extract title and content (customize per platform)
            self.title = soup.title.text.strip() if soup.title else "Untitled Page"
            content_blocks = soup.find_all('p')
            self.description = "\n".join(p.get_text(strip=True) for p in content_blocks)

        except Exception as e:
            print(f"[ERROR] Failed to scrape {self.url}: {e}")
            self.description = "Could not fetch content."

    def get_content(self):
        return f"{self.title}\n\n{self.description}"
