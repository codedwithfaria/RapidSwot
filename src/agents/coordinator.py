"""
Coordinator Agent for managing multi-agent workflows.
"""
import os
from typing import AsyncGenerator, Optional, List
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

class AgentCoordinator:
    """Main coordinator for managing the multi-agent system."""
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self._setup_agents()
    
    def _setup_agents(self):
        """Initialize the agent hierarchy and specialized agents."""
        # Initialize specialized agents
        self.browser_agent = self._create_browser_agent()
        self.research_agent = self._create_research_agent()
        self.synthesis_agent = self._create_synthesis_agent()
        
        # Create the root coordinator agent
        self.coordinator = LlmAgent(
            name="MainCoordinator",
            model=self.model,
            description="I coordinate complex tasks across multiple specialized agents.",
            instruction="""
            You are the main coordinator for a multi-agent system. Your role is to:
            1. Break down complex tasks into subtasks
            2. Delegate tasks to appropriate specialized agents
            3. Manage sequential and parallel workflows
            4. Synthesize results from multiple agents
            """,
            sub_agents=[
                self.browser_agent,
                self.research_agent,
                self.synthesis_agent
            ]
        )

    def _create_browser_agent(self) -> LlmAgent:
        """Create an agent specialized in browser automation."""
        return LlmAgent(
            name="BrowserAgent",
            model=self.model,
            description="I handle web browsing and interaction tasks.",
            instruction="""
            You are specialized in browser automation tasks. You can:
            1. Navigate to web pages
            2. Extract information
            3. Fill forms and interact with web elements
            4. Capture screenshots and page content
            """
        )
    
    def _create_research_agent(self) -> LlmAgent:
        """Create an agent specialized in research and information gathering."""
        return LlmAgent(
            name="ResearchAgent",
            model=self.model,
            description="I gather and analyze information from multiple sources.",
            instruction="""
            You are specialized in research tasks. You can:
            1. Search for information across multiple sources
            2. Verify and cross-reference facts
            3. Extract key insights
            4. Generate research summaries
            """
        )
    
    def _create_synthesis_agent(self) -> LlmAgent:
        """Create an agent specialized in synthesizing information."""
        return LlmAgent(
            name="SynthesisAgent",
            model=self.model,
            description="I combine and synthesize information from multiple sources.",
            instruction="""
            You are specialized in synthesizing information. You can:
            1. Combine data from multiple sources
            2. Generate comprehensive reports
            3. Identify patterns and insights
            4. Create structured summaries
            """
        )

    def create_sequential_workflow(self, agents: List[BaseAgent], name: str = "SequentialWorkflow") -> SequentialAgent:
        """Create a sequential workflow with the specified agents."""
        return SequentialAgent(name=name, sub_agents=agents)
    
    def create_parallel_workflow(self, agents: List[BaseAgent], name: str = "ParallelWorkflow") -> ParallelAgent:
        """Create a parallel workflow with the specified agents."""
        return ParallelAgent(name=name, sub_agents=agents)