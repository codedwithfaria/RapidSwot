"""
Main entry point for the RapidSwot AI browser agent.
"""
import asyncio
import uvicorn
from fastapi import FastAPI
from typing import Dict, Any

from agent import BrowserAgent
from fetch import WebFetcher
from filesystem import FileSystem
from git import GitTools
from memory import MemorySystem
from sequential import SequentialThinking
from time import TimeManager

app = FastAPI(title="RapidSwot AI Browser Agent")
agent = BrowserAgent()

@app.post("/execute")
async def execute_task(task: Dict[str, Any]):
    """Execute a browser automation task."""
    return await agent.execute_task(task["description"])

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

def main():
    """Start the RapidSwot server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()