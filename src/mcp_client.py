"""
MCP client implementation for programmatic usage.
"""
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, Tuple, AsyncGenerator
import json
import subprocess

@dataclass
class MCPClientConfig:
    command: str = "uvx"
    args: list = None
    env: Dict[str, str] = None

class MCPClient:
    def __init__(self, config: MCPClientConfig):
        self.config = config
        self.process = None
        self.request_id = 0
        
    async def start(self):
        """Start the MCP server process."""
        args = [self.config.command]
        if self.config.args:
            args.extend(self.config.args)
            
        self.process = await asyncio.create_subprocess_exec(
            *args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.config.env
        )
        
        # Initialize connection
        await self._send_request({
            "type": "initialize",
            "request_id": self._next_request_id()
        })
        
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the MCP server and get response."""
        if not self.process:
            raise RuntimeError("MCP server not started")
            
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()
        
        response_line = await self.process.stdout.readline()
        return json.loads(response_line)
        
    def _next_request_id(self) -> str:
        """Generate next request ID."""
        self.request_id += 1
        return str(self.request_id)
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        response = await self._send_request({
            "type": "tool_call",
            "request_id": self._next_request_id(),
            "tool": {
                "name": tool_name,
                "arguments": arguments
            }
        })
        
        if response.get("error"):
            raise Exception(response["error"])
            
        return response.get("content", [{}])[0]
        
    async def close(self):
        """Clean up resources."""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.process = None

async def example_usage():
    """Example of using the MCP client."""
    config = MCPClientConfig(
        command="rapidswot",
        args=["--mcp"],
        env={"OPENAI_API_KEY": "your-key-here"}
    )
    
    client = MCPClient(config)
    await client.start()
    
    try:
        # Navigate to a website
        await client.call_tool(
            "browser_navigate",
            {"url": "https://example.com"}
        )
        
        # Extract content
        result = await client.call_tool(
            "browser_extract_content",
            {"selector": "h1"}
        )
        print(result)
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(example_usage())