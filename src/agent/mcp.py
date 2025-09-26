"""
Browser automation service exposed via MCP.
"""
from typing import Dict, Any, Optional
import asyncio
import json
from dataclasses import dataclass
from .browser import BrowserController

@dataclass
class MCPToolResponse:
    content: list
    error: Optional[str] = None

class BrowserMCPService:
    def __init__(self):
        self.browser = BrowserController()
        self.sessions: Dict[str, BrowserController] = {}
        
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
            
    async def tool_browser_navigate(self, url: str) -> Dict[str, Any]:
        """Navigate to a URL."""
        await self.browser.navigate(url)
        return {"status": "success", "url": url}
        
    async def tool_browser_click(self, selector: str, index: int = 0) -> Dict[str, Any]:
        """Click an element by selector and index."""
        await self.browser.click(f"{selector}:nth({index})")
        return {"status": "success", "action": "click", "selector": selector, "index": index}
        
    async def tool_browser_type(self, selector: str, text: str) -> Dict[str, Any]:
        """Type text into an element."""
        await self.browser.type(selector, text)
        return {"status": "success", "action": "type", "selector": selector}
        
    async def tool_browser_extract_content(self, selector: str) -> Dict[str, Any]:
        """Extract content from the page."""
        content = await self.browser.extract_content(selector)
        return {"content": content}
        
    async def tool_browser_get_state(self, include_screenshot: bool = False) -> Dict[str, Any]:
        """Get current page state."""
        # Implementation to get page state
        pass
        
    async def tool_browser_list_sessions(self) -> Dict[str, Any]:
        """List all active browser sessions."""
        return {
            "sessions": [
                {"id": session_id, "url": await session.get_current_url()}
                for session_id, session in self.sessions.items()
            ]
        }
        
    async def tool_browser_close_session(self, session_id: str) -> Dict[str, Any]:
        """Close a specific browser session."""
        if session_id in self.sessions:
            await self.sessions[session_id].stop()
            del self.sessions[session_id]
            return {"status": "success", "message": f"Session {session_id} closed"}
        return {"status": "error", "message": "Session not found"}