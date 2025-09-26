"""
MCP server implementation for Vibetest.
"""
from typing import Dict, Any
import asyncio
import json
import os
from pathlib import Path

from ..agent.mcp import MCPToolResponse
from . import Vibetest, VibetestConfig

class VibetestMCPService:
    def __init__(self):
        self.active_tests = {}
        
    async def handle_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResponse:
        """Handle MCP tool calls."""
        try:
            if not hasattr(self, f"tool_{tool_name}"):
                raise ValueError(f"Unknown tool: {tool_name}")
                
            handler = getattr(self, f"tool_{tool_name}")
            result = await handler(**arguments)
            return MCPToolResponse(content=[{"text": json.dumps(result)}])
            
        except Exception as e:
            return MCPToolResponse(content=[], error=str(e))
            
    async def tool_vibetest_start(
        self,
        url: str,
        num_agents: int = 3,
        headless: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start a new Vibetest run."""
        # Set up output directory
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            screenshot_dir = str(Path(output_dir) / "screenshots")
            report_file = str(Path(output_dir) / "report.json")
        else:
            screenshot_dir = None
            report_file = None
            
        config = VibetestConfig(
            url=url,
            num_agents=num_agents,
            headless=headless,
            screenshot_dir=screenshot_dir,
            report_file=report_file
        )
        
        test_id = len(self.active_tests)
        self.active_tests[test_id] = Vibetest(config)
        
        # Start test in background
        asyncio.create_task(self._run_test(test_id))
        
        return {
            "test_id": test_id,
            "status": "started",
            "config": {
                "url": url,
                "num_agents": num_agents,
                "headless": headless,
                "output_dir": output_dir
            }
        }
        
    async def tool_vibetest_status(self, test_id: int) -> Dict[str, Any]:
        """Get status of a running test."""
        if test_id not in self.active_tests:
            return {"error": "Test ID not found"}
            
        test = self.active_tests[test_id]
        return {
            "test_id": test_id,
            "urls_tested": sum(len(agent.visited_urls) for agent in test.agents),
            "issues_found": sum(len(agent.issues) for agent in test.agents)
        }
        
    async def tool_vibetest_stop(self, test_id: int) -> Dict[str, Any]:
        """Stop a running test."""
        if test_id not in self.active_tests:
            return {"error": "Test ID not found"}
            
        test = self.active_tests[test_id]
        await asyncio.gather(*(agent.stop() for agent in test.agents))
        del self.active_tests[test_id]
        
        return {
            "test_id": test_id,
            "status": "stopped"
        }
        
    async def _run_test(self, test_id: int):
        """Run a test in the background."""
        test = self.active_tests[test_id]
        await test.run()