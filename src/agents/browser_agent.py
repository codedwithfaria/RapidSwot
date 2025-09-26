"""
Browser Automation Agent with Playwright integration.
"""
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.function_tool import FunctionTool
from playwright.async_api import async_playwright

class BrowserAutomationAgent(BaseAgent):
    """Agent specialized in browser automation using Playwright."""
    
    def __init__(self, name: str = "BrowserAutomator"):
        super().__init__(name=name)
        self.browser = None
        self.page = None
        self._setup_tools()
    
    def _setup_tools(self):
        """Initialize browser automation tools."""
        self.tools = [
            FunctionTool(self.navigate_to_page),
            FunctionTool(self.extract_content),
            FunctionTool(self.take_screenshot),
            FunctionTool(self.fill_form)
        ]
    
    async def initialize_browser(self):
        """Initialize the browser instance if not already running."""
        if not self.browser:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch()
            self.page = await self.browser.new_page()
    
    async def navigate_to_page(self, url: str) -> Dict[str, Any]:
        """Navigate to a specified URL."""
        await self.initialize_browser()
        await self.page.goto(url)
        return {"status": "success", "url": url}
    
    async def extract_content(self, selector: str) -> Dict[str, Any]:
        """Extract content from the page using a CSS selector."""
        await self.initialize_browser()
        content = await self.page.text_content(selector)
        return {"content": content, "selector": selector}
    
    async def take_screenshot(self, selector: Optional[str] = None) -> Dict[str, Any]:
        """Take a screenshot of the entire page or a specific element."""
        await self.initialize_browser()
        if selector:
            element = await self.page.wait_for_selector(selector)
            screenshot = await element.screenshot()
        else:
            screenshot = await self.page.screenshot()
        return {"screenshot": screenshot}
    
    async def fill_form(self, selector: str, value: str) -> Dict[str, Any]:
        """Fill a form field with the specified value."""
        await self.initialize_browser()
        await self.page.fill(selector, value)
        return {"status": "success", "selector": selector}
    
    async def cleanup(self):
        """Clean up browser resources."""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.page = None
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Implementation of the agent's run logic."""
        try:
            # Initialize browser on first run
            await self.initialize_browser()
            
            # Process the input using available tools
            for tool in self.tools:
                if tool.matches(ctx):
                    result = await tool.run_async(ctx)
                    yield Event(
                        author=self.name,
                        content=result
                    )
        except Exception as e:
            yield Event(
                author=self.name,
                content=f"Error: {str(e)}",
                actions=EventActions(escalate=True)
            )
        finally:
            # Ensure cleanup happens
            await self.cleanup()