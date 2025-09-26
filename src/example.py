"""
Example usage of the multi-agent system.
"""
import asyncio
import uuid
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.coordinator import AgentCoordinator
from agents.browser_agent import BrowserAutomationAgent
from agents.workflow_agent import WorkflowAgent

async def run_example():
    """Run an example multi-agent workflow."""
    
    # Initialize services
    session_service = InMemorySessionService()
    coordinator = AgentCoordinator()
    
    # Create a session
    session = await session_service.create_session(
        app_name="multi_agent_demo",
        user_id=f"user_{uuid.uuid4()}",
        state={}
    )
    
    # Set up the runner
    runner = Runner(
        app_name="multi_agent_demo",
        agent=coordinator.coordinator,
        session_service=session_service
    )
    
    # Example task: Research and summarize information about a topic
    task_prompt = """
    Research the topic of "artificial intelligence in healthcare" and create a summary.
    Steps:
    1. Search for relevant information
    2. Extract key points from multiple sources
    3. Synthesize the information into a coherent summary
    """
    
    content = types.Content(
        role='user',
        parts=[types.Part(text=task_prompt)]
    )
    
    print("Starting multi-agent task...")
    events_async = runner.run_async(
        session_id=session.id,
        user_id=session.user_id,
        new_message=content
    )
    
    async for event in events_async:
        if event.is_final_response():
            print(f"\nFinal Response: {event.content}")
        else:
            print(f"Event from {event.author}: {event.content}")

if __name__ == "__main__":
    asyncio.run(run_example())