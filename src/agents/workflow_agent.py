"""
Workflow Management Agent for orchestrating complex task sequences.
"""
from typing import AsyncGenerator, List, Dict, Any
from google.adk.agents import BaseAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext

class WorkflowAgent(BaseAgent):
    """Agent specialized in managing and executing complex workflows."""
    
    def __init__(self, name: str = "WorkflowManager"):
        super().__init__(name=name)
        self.active_workflows: Dict[str, BaseAgent] = {}
    
    def create_sequential_workflow(
        self,
        workflow_id: str,
        agents: List[BaseAgent],
        state: Dict[str, Any] = None
    ) -> SequentialAgent:
        """Create a new sequential workflow."""
        workflow = SequentialAgent(
            name=f"Sequential_{workflow_id}",
            sub_agents=agents
        )
        self.active_workflows[workflow_id] = workflow
        return workflow
    
    def create_parallel_workflow(
        self,
        workflow_id: str,
        agents: List[BaseAgent],
        state: Dict[str, Any] = None
    ) -> ParallelAgent:
        """Create a new parallel workflow."""
        workflow = ParallelAgent(
            name=f"Parallel_{workflow_id}",
            sub_agents=agents
        )
        self.active_workflows[workflow_id] = workflow
        return workflow
    
    def create_loop_workflow(
        self,
        workflow_id: str,
        agents: List[BaseAgent],
        max_iterations: int = None,
        state: Dict[str, Any] = None
    ) -> LoopAgent:
        """Create a new loop workflow."""
        workflow = LoopAgent(
            name=f"Loop_{workflow_id}",
            sub_agents=agents,
            max_iterations=max_iterations
        )
        self.active_workflows[workflow_id] = workflow
        return workflow
    
    async def execute_workflow(
        self,
        workflow_id: str,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Execute a specific workflow."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            yield Event(
                author=self.name,
                content=f"Error: Workflow {workflow_id} not found",
                actions=EventActions(escalate=True)
            )
            return
        
        try:
            async for event in workflow.run_async(ctx):
                yield event
        except Exception as e:
            yield Event(
                author=self.name,
                content=f"Error executing workflow {workflow_id}: {str(e)}",
                actions=EventActions(escalate=True)
            )
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Implementation of the agent's run logic."""
        # Process workflow management commands from the context
        command = ctx.session.state.get("workflow_command")
        if command:
            if command["type"] == "create":
                if command["workflow_type"] == "sequential":
                    self.create_sequential_workflow(
                        command["workflow_id"],
                        command["agents"],
                        command.get("state")
                    )
                elif command["workflow_type"] == "parallel":
                    self.create_parallel_workflow(
                        command["workflow_id"],
                        command["agents"],
                        command.get("state")
                    )
                elif command["workflow_type"] == "loop":
                    self.create_loop_workflow(
                        command["workflow_id"],
                        command["agents"],
                        command.get("max_iterations"),
                        command.get("state")
                    )
                yield Event(
                    author=self.name,
                    content=f"Created {command['workflow_type']} workflow: {command['workflow_id']}"
                )
            
            elif command["type"] == "execute":
                async for event in self.execute_workflow(command["workflow_id"], ctx):
                    yield event
                    
            elif command["type"] == "list":
                workflow_list = list(self.active_workflows.keys())
                yield Event(
                    author=self.name,
                    content=f"Active workflows: {workflow_list}"
                )