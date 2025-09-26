"""
Web content fetching and processing module.
Handles efficient retrieval and preparation of web content for LLM processing.
"""
import aiohttp
from bs4 import BeautifulSoup
from typing import Dict, List, Optional

class WebFetcher:
    def __init__(self):
        self.session = None
        
    async def fetch_url(self, url: str) -> str:
        """
        Fetch and process content from a URL.
        
        Args:
            url: The URL to fetch content from
            
        Returns:
            Processed content optimized for LLM input
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        async with self.session.get(url) as response:
            html = await response.text()
            
        # Process HTML into clean text optimized for LLM
        soup = BeautifulSoup(html, 'html.parser')
        return self._clean_content(soup)
        
    def _clean_content(self, soup: BeautifulSoup) -> str:
        """Clean and optimize HTML content for LLM processing."""
        # Remove scripts, styles, etc.
        for tag in soup(['script', 'style', 'meta', 'link']):
            tag.decompose()
            
        return soup.get_text(separator=' ', strip=True)