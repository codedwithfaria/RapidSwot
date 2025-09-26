"""
Main entry point for the RapidSwot MCP server.
"""
import asyncio
import json
import os
import sys
from typing import Any, Dict, Optional

from .agent.mcp import BrowserMCPService
from .agent.browser import BrowserController

class MCPServer:
    def __init__(self):
        self.service = BrowserMCPService()
        
    async def handle_stdin(self):
        """Handle stdin MCP protocol communication."""
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                    
                request = json.loads(line)
                response = await self.handle_request(request)
                
                print(json.dumps(response))
                sys.stdout.flush()
                
            except Exception as e:
                print(json.dumps({
                    "error": str(e),
                    "request_id": request.get("request_id") if "request" in locals() else None
                }))
                sys.stdout.flush()
                
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request and return response."""
        request_type = request.get("type")
        
        if request_type == "initialize":
            return {
                "type": "initialize_response",
                "request_id": request["request_id"],
                "protocols": ["mcp1"],
                "tools": self.get_tool_specifications()
            }
            
        elif request_type == "tool_call":
            tool_name = request["tool"]["name"]
            arguments = request["tool"].get("arguments", {})
            
            response = await self.service.handle_tool(tool_name, arguments)
            
            return {
                "type": "tool_call_response",
                "request_id": request["request_id"],
                "content": response.content,
                "error": response.error
            }
            
    def get_tool_specifications(self) -> Dict[str, Any]:
        """Return MCP tool specifications."""
        return {
            "browser_navigate": {
                "description": "Navigate to a URL",
                "parameters": {
                    "url": {"type": "string", "description": "URL to navigate to"}
                }
            },
            "browser_click": {
                "description": "Click on an element",
                "parameters": {
                    "selector": {"type": "string", "description": "Element selector"},
                    "index": {"type": "integer", "description": "Element index", "default": 0}
                }
            },
            "browser_type": {
                "description": "Type text into an element",
                "parameters": {
                    "selector": {"type": "string", "description": "Element selector"},
                    "text": {"type": "string", "description": "Text to type"}
                }
            }
            # Add more tool specifications as needed
        }

def main():
    """Start the MCP server."""
    server = MCPServer()
    
    # Configure environment
    os.environ["BROWSER_USE_HEADLESS"] = os.environ.get("BROWSER_USE_HEADLESS", "true")
    
    # Run server
    asyncio.run(server.handle_stdin())

if __name__ == "__main__":
    main()