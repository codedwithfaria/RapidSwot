"""Core agent implementation for bridging high-level intents to actions."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext


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

        if not response or not response.strip():
            logger.warning("Empty response received from LLM; using default plan structure")
            return self._default_plan()

        json_block = self._extract_json_block(response)

        if json_block:
            try:
                parsed = json.loads(json_block)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON block from LLM response: %s", exc)
            else:
                normalized = self._normalize_plan(parsed)
                if normalized != self._default_plan():
                    return normalized
                logger.info("Parsed plan did not contain meaningful content; using default")

        logger.info("Falling back to default plan parsing for response: %s", response)
        return self._default_plan()

    @staticmethod
    def _default_plan() -> Dict[str, Any]:
        """Return a canonical empty plan structure."""

        return {
            "steps": [],
            "resources": {},
            "duration": 0.0,
            "metrics": {},
        }

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

        steps_key, steps_value = IntentProcessor._first_matching_key(
            raw_plan,
            ("steps", "Steps", "plan", "Plan"),
        )
        _, resources_value = IntentProcessor._first_matching_key(
            raw_plan,
            ("resources", "Resources"),
        )
        _, duration_value = IntentProcessor._first_matching_key(
            raw_plan,
            ("estimated_duration", "duration", "Duration"),
        )
        _, metrics_value = IntentProcessor._first_matching_key(
            raw_plan,
            ("success_metrics", "metrics", "Metrics"),
        )

        if steps_key is None:
            logger.debug("Plan data missing steps; returning empty list")

        return {
            "steps": _normalize_steps(steps_value),
            "resources": resources_value or {},
            "duration": float(duration_value) if isinstance(duration_value, (int, float)) else 0.0,
            "metrics": metrics_value or {},
        }

    @staticmethod
    def _first_matching_key(
        data: Dict[str, Any],
        keys: Tuple[str, ...],
    ) -> Tuple[Optional[str], Any]:
        """Return the first matching key and its value from ``data``."""

        for key in keys:
            if key in data:
                return key, data[key]
        return None, None

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
                if not isinstance(step, dict):
                    raise TypeError(f"Plan step must be a mapping, received: {type(step)!r}")

                tool_name = step.get("tool")
                if not tool_name:
                    raise KeyError("Plan step missing required 'tool' field")

                tool = self.available_tools.get(tool_name)
                if not tool:
                    raise ValueError(f"Tool not found: {tool_name}")

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