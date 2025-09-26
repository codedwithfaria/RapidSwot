"""
Browser automation core module using Playwright.
"""
from playwright.async_api import async_playwright, Browser, Page
from typing import Optional, Dict, Any
import asyncio

class BrowserController:
    def __init__(self):
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        
    async def start(self):
        """Initialize browser instance."""
        playwright = await async_playwright().start()
        self._browser = await playwright.chromium.launch()
        self._page = await self._browser.new_page()
        
    async def stop(self):
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
            
    async def navigate(self, url: str):
        """Navigate to a URL."""
        if not self._page:
            await self.start()
        await self._page.goto(url)
        
    async def extract_content(self, selector: str) -> str:
        """Extract content from elements matching selector."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        elements = await self._page.query_selector_all(selector)
        texts = []
        for element in elements:
            text = await element.text_content()
            if text:
                texts.append(text.strip())
        return "\n".join(texts)
        
    async def click(self, selector: str):
        """Click an element matching selector."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        await self._page.click(selector)
        
    async def type(self, selector: str, text: str):
        """Type text into an element matching selector."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        await self._page.fill(selector, text)
        
    async def screenshot(self, path: str):
        """Take a screenshot of the current page."""
        if not self._page:
            raise RuntimeError("Browser not initialized")
        await self._page.screenshot(path=path)