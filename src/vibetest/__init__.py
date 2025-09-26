"""
Vibetest module for automated QA testing.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import json

from ..agent import Agent, BrowserController
from ..agent.llm import LLMBackend

@dataclass
class VibetestConfig:
    url: str
    num_agents: int = 3
    headless: bool = True
    max_depth: int = 5
    screenshot_dir: Optional[str] = None
    report_file: Optional[str] = None
    
    def __post_init__(self):
        if self.screenshot_dir:
            Path(self.screenshot_dir).mkdir(parents=True, exist_ok=True)

class VibetestAgent:
    def __init__(self, config: VibetestConfig):
        self.config = config
        self.browser = BrowserController(headless=config.headless)
        self.visited_urls = set()
        self.issues = []
        
    async def start(self):
        """Initialize the testing agent."""
        await self.browser.start()
        
    async def stop(self):
        """Clean up resources."""
        await self.browser.stop()
        
    async def test_url(self, url: str) -> Dict[str, Any]:
        """Test a specific URL for issues."""
        if url in self.visited_urls:
            return {}
            
        self.visited_urls.add(url)
        
        try:
            await self.browser.navigate(url)
            
            # Take screenshot if configured
            if self.config.screenshot_dir:
                screenshot_path = Path(self.config.screenshot_dir) / f"{len(self.visited_urls)}.png"
                await self.browser.screenshot(str(screenshot_path))
            
            # Check for issues
            issues = await self._check_page_issues()
            if issues:
                self.issues.extend(issues)
                
            # Find new URLs to test
            new_urls = await self._extract_urls()
            return {
                "url": url,
                "issues": issues,
                "new_urls": new_urls
            }
            
        except Exception as e:
            self.issues.append({
                "type": "error",
                "url": url,
                "message": str(e)
            })
            return {"url": url, "error": str(e)}
            
    async def _check_page_issues(self) -> List[Dict[str, Any]]:
        """Check for various page issues."""
        issues = []
        
        # Check for broken links
        links = await self.browser._page.query_selector_all("a[href]")
        for link in links:
            href = await link.get_attribute("href")
            if href and not await self._is_valid_link(href):
                issues.append({
                    "type": "broken_link",
                    "url": href,
                    "element": await link.evaluate("el => el.outerHTML")
                })
                
        # Check for accessibility issues
        try:
            a11y_issues = await self.browser._page.accessibility.snapshot()
            if a11y_issues:
                issues.extend([
                    {"type": "accessibility", "issue": issue}
                    for issue in a11y_issues
                ])
        except Exception:
            pass
            
        return issues
        
    async def _is_valid_link(self, url: str) -> bool:
        """Check if a link is valid."""
        try:
            response = await self.browser._page.request.head(url)
            return response.status < 400
        except:
            return False
            
    async def _extract_urls(self) -> List[str]:
        """Extract URLs from the current page for crawling."""
        urls = []
        links = await self.browser._page.query_selector_all("a[href]")
        for link in links:
            href = await link.get_attribute("href")
            if href and href.startswith(self.config.url):
                urls.append(href)
        return urls

class Vibetest:
    def __init__(self, config: VibetestConfig):
        self.config = config
        self.agents: List[VibetestAgent] = []
        
    async def run(self) -> Dict[str, Any]:
        """Run the test suite with multiple agents."""
        # Initialize agents
        self.agents = [
            VibetestAgent(self.config)
            for _ in range(self.config.num_agents)
        ]
        
        try:
            # Start all agents
            await asyncio.gather(
                *(agent.start() for agent in self.agents)
            )
            
            # Initial URL queue
            url_queue = asyncio.Queue()
            url_queue.put_nowait(self.config.url)
            
            # Run tests
            results = await asyncio.gather(
                *(self._agent_worker(i, url_queue) for i in range(len(self.agents)))
            )
            
            # Combine results
            all_issues = []
            for agent in self.agents:
                all_issues.extend(agent.issues)
                
            report = {
                "url": self.config.url,
                "total_urls_tested": sum(len(agent.visited_urls) for agent in self.agents),
                "issues": all_issues
            }
            
            # Save report if configured
            if self.config.report_file:
                with open(self.config.report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                    
            return report
            
        finally:
            # Cleanup
            await asyncio.gather(
                *(agent.stop() for agent in self.agents)
            )
            
    async def _agent_worker(self, agent_id: int, url_queue: asyncio.Queue):
        """Worker process for each agent."""
        agent = self.agents[agent_id]
        visited_count = 0
        
        while visited_count < self.config.max_depth:
            try:
                url = await url_queue.get()
            except asyncio.QueueEmpty:
                break
                
            result = await agent.test_url(url)
            visited_count += 1
            
            # Add new URLs to queue
            for new_url in result.get("new_urls", []):
                if new_url not in agent.visited_urls:
                    url_queue.put_nowait(new_url)