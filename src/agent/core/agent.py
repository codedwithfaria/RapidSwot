"""Core agent implementation for bridging high-level intents to actions."""

import asyncio
import json
import logging
import re
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)
from dataclasses import dataclass, field

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskIntent:
    """Represents a high-level user intention."""
    description: str
    context: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ActionPlan:
    """Structured plan for executing a task intent."""
    steps: List[Dict[str, Any]]
    resources: Dict[str, Any]
    estimated_duration: float
    success_metrics: Dict[str, Any]

class IntentProcessor:
    """Processes high-level intents into structured action plans."""

    def __init__(self, llm_agent: LlmAgent):
        self.llm = llm_agent

    async def process_intent(self, intent: TaskIntent) -> ActionPlan:
        """Convert a high-level intent into an action plan."""
        # Use LLM to analyze and structure the intent
        response = await self.llm.generate_response(
            f"""Analyze this task intent and create a structured plan:
            Description: {intent.description}
            Context: {intent.context}
            Constraints: {intent.constraints}
            
            Generate a detailed plan with:
            1. Sequential steps with clear success criteria
            2. Required resources and dependencies
            3. Estimated duration and complexity
            4. Measurable success metrics"""
        )
        
        # Parse LLM response into structured plan
        plan_data = self._parse_llm_response(response)

        return ActionPlan(
            steps=plan_data["steps"],
            resources=plan_data["resources"],
            estimated_duration=plan_data["duration"],
            success_metrics=plan_data["metrics"]
        )

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse an LLM response into a structured action plan."""

        def _default_plan() -> Dict[str, Any]:
            return {
                "steps": [],
                "resources": {},
                "duration": 0.0,
                "metrics": {},
            }

        if not response or not response.strip():
            logger.warning("Empty response received from LLM; using default plan structure")
            return _default_plan()

        json_block = self._extract_json_block(response)

        if json_block:
            try:
                parsed = json.loads(json_block)
                return self._normalize_plan(parsed)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON block from LLM response: %s", exc)

        logger.info("Falling back to default plan parsing for response: %s", response)
        return _default_plan()

    @staticmethod
    def _extract_json_block(response: str) -> Optional[str]:
        """Extract the first JSON block from an LLM response."""

        fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)

        curly_match = re.search(r"(\{.*\})", response, re.DOTALL)
        if curly_match:
            return curly_match.group(1)

        return None

    @staticmethod
    def _normalize_plan(raw_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure the parsed plan contains the expected structure and defaults."""

        def _normalize_steps(steps: Union[List[Any], Dict[str, Any], None]) -> List[Dict[str, Any]]:
            if steps is None:
                return []
            if isinstance(steps, dict):
                steps_iter: Iterable[Any] = steps.values()
            else:
                steps_iter = steps  # type: ignore[assignment]

            normalized_steps: List[Dict[str, Any]] = []
            for idx, step in enumerate(steps_iter):
                if isinstance(step, dict):
                    normalized_steps.append(step)
                elif isinstance(step, str):
                    normalized_steps.append({"description": step, "tool": "", "params": {}})
                else:
                    logger.debug("Skipping unsupported step format at index %s: %r", idx, step)
            return normalized_steps

        return {
            "steps": _normalize_steps(
                raw_plan.get("steps")
                or raw_plan.get("Steps")
                or raw_plan.get("plan")
            ),
            "resources": raw_plan.get("resources") or raw_plan.get("Resources") or {},
            "duration": raw_plan.get("estimated_duration")
            or raw_plan.get("duration")
            or 0.0,
            "metrics": raw_plan.get("success_metrics")
            or raw_plan.get("metrics")
            or {},
        }

ToolFn = Callable[[Dict[str, Any], Dict[str, Any]], Union[Any, Awaitable[Any]]]


class ExecutionEngine:
    """Executes action plans using available tools and resources."""

    def __init__(self):
        self.available_tools: Dict[str, ToolFn] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}

    def register_tool(self, name: str, tool: ToolFn) -> None:
        """Register a new tool for task execution."""
        self.available_tools[name] = tool

    async def execute_plan(
        self,
        plan: ActionPlan,
        context: Dict[str, Any]
    ) -> AsyncGenerator[Event, None]:
        """Execute an action plan step by step."""
        for step in plan.steps:
            try:
                tool = self.available_tools.get(step["tool"])
                if not tool:
                    raise ValueError(f"Tool not found: {step['tool']}")

                invocation = tool(step.get("params", {}), context)
                result = await invocation if asyncio.iscoroutine(invocation) else invocation

                yield Event(
                    author="ExecutionEngine",
                    content={"step": step, "result": result},
                    actions=EventActions(escalate=False)
                )
                
            except Exception as e:
                logger.error(f"Step execution error: {e}")
                yield Event(
                    author="ExecutionEngine",
                    content={"error": str(e), "step": step},
                    actions=EventActions(escalate=True)
                )
                return

class RapidSwotAgent(BaseAgent):
    """Main agent class that bridges intentions to actions."""
    
    def __init__(
        self,
        name: str,
        llm_agent: LlmAgent,
        execution_engine: Optional[ExecutionEngine] = None
    ):
        super().__init__(name=name)
        self.intent_processor = IntentProcessor(llm_agent)
        self.execution_engine = execution_engine or ExecutionEngine()
        self.active_tasks: Dict[str, TaskIntent] = {}
    
    async def process_intention(
        self,
        description: str,
        context: Dict[str, Any] = None,
        constraints: Dict[str, Any] = None
    ) -> str:
        """Process a new task intention."""
        task_id = f"task_{len(self.active_tasks)}"
        
        intent = TaskIntent(
            description=description,
            context=context or {},
            constraints=constraints or {}
        )
        
        self.active_tasks[task_id] = intent
        return task_id
    
    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Implementation of the agent's run logic."""
        try:
            # Get task details from context
            task_id = ctx.session.state.get("task_id")
            if not task_id or task_id not in self.active_tasks:
                yield Event(
                    author=self.name,
                    content="No active task found",
                    actions=EventActions(escalate=True)
                )
                return
            
            intent = self.active_tasks[task_id]
            
            # Process intent into action plan
            plan = await self.intent_processor.process_intent(intent)
            
            # Execute the plan
            async for event in self.execution_engine.execute_plan(
                plan,
                ctx.session.state
            ):
                yield event
            
            # Clean up completed task
            del self.active_tasks[task_id]
            
            yield Event(
                author=self.name,
                content="Task completed successfully",
                actions=EventActions(escalate=False)
            )
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            yield Event(
                author=self.name,
                content=f"Error: {str(e)}",
                actions=EventActions(escalate=True)
            )