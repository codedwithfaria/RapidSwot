"""Core agent implementation for bridging high-level intents to actions."""

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
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

@runtime_checkable
class PlanStepLike(Protocol):
    """Runtime protocol representing planner step objects."""

    action: str
    params: Dict[str, Any]


PlanStep = Union[Dict[str, Any], PlanStepLike]


class PromptTechnique(Enum):
    """Prompt engineering techniques supported by the agent."""

    ZERO_SHOT = "Zero-shot Prompting"
    FEW_SHOT = "Few-shot Prompting"
    CHAIN_OF_THOUGHT = "Chain-of-Thought Prompting"
    META = "Meta Prompting"
    SELF_CONSISTENCY = "Self-Consistency"
    GENERATE_KNOWLEDGE = "Generate Knowledge Prompting"
    PROMPT_CHAINING = "Prompt Chaining"
    TREE_OF_THOUGHTS = "Tree of Thoughts"
    RETRIEVAL_AUGMENTED = "Retrieval Augmented Generation"
    AUTOMATIC_REASONING_TOOL_USE = "Automatic Reasoning and Tool-use"
    AUTOMATIC_PROMPT_ENGINEER = "Automatic Prompt Engineer"
    ACTIVE_PROMPT = "Active-Prompt"
    DIRECTIONAL_STIMULUS = "Directional Stimulus Prompting"
    PROGRAM_AIDED = "Program-Aided Language Models"
    REACT = "ReAct"
    REFLEXION = "Reflexion"
    MULTIMODAL_COT = "Multimodal CoT"
    GRAPH_PROMPTING = "Graph Prompting"

    @property
    def description(self) -> str:
        """Human-readable description for documentation and prompting."""

        return _TECHNIQUE_DESCRIPTIONS[self]


_TECHNIQUE_DESCRIPTIONS: Dict[PromptTechnique, str] = {
    PromptTechnique.ZERO_SHOT: "Solve tasks directly from instructions without examples to test generalization.",
    PromptTechnique.FEW_SHOT: (
        "Show concise input/output exemplars so the model can mirror the demonstrated pattern and raise answer quality."
    ),
    PromptTechnique.CHAIN_OF_THOUGHT: (
        "Ask the model to think step-by-step, decomposing complex problems into intermediate reasoning before final answers."
    ),
    PromptTechnique.META: "Use instructions that describe how to construct or critique prompts themselves.",
    PromptTechnique.SELF_CONSISTENCY: "Sample multiple reasoning paths and pick the most consistent answer.",
    PromptTechnique.GENERATE_KNOWLEDGE: "Ask the model to recall or synthesize supporting facts before solving.",
    PromptTechnique.PROMPT_CHAINING: "Break tasks into sequential prompts whose outputs feed later steps.",
    PromptTechnique.TREE_OF_THOUGHTS: "Explore alternative reasoning branches and evaluate the best path.",
    PromptTechnique.RETRIEVAL_AUGMENTED: "Retrieve external knowledge to ground responses in authoritative data.",
    PromptTechnique.AUTOMATIC_REASONING_TOOL_USE: "Let the model autonomously call tools or APIs when needed.",
    PromptTechnique.AUTOMATIC_PROMPT_ENGINEER: "Iteratively refine prompts with model feedback to improve results.",
    PromptTechnique.ACTIVE_PROMPT: "Prioritize difficult examples when crafting few-shot demonstrations.",
    PromptTechnique.DIRECTIONAL_STIMULUS: "Nudge generation with steering phrases that emphasize desired traits.",
    PromptTechnique.PROGRAM_AIDED: "Pair natural language with code execution for reliable calculations.",
    PromptTechnique.REACT: (
        "Blend explicit reasoning with grounded tool use so the model can plan, act, and verify results during execution."
    ),
    PromptTechnique.REFLEXION: "Reflect on earlier attempts and critique them to self-correct.",
    PromptTechnique.MULTIMODAL_COT: "Extend chain-of-thought to combine textual and visual reasoning.",
    PromptTechnique.GRAPH_PROMPTING: "Represent relational knowledge explicitly to support structured reasoning.",
}


@dataclass
class PromptEngineeringGuide:
    """Helper for formatting prompt engineering instructions."""

    intro: str = (
        "Effective prompt design matters—apply these structured techniques to plan with clarity, specificity, and safe reasoning:"
    )

    def format_instructions(self, techniques: Iterable[PromptTechnique]) -> str:
        """Return a formatted instruction block for the given techniques."""

        items = list(dict.fromkeys(techniques))  # Preserve order while removing duplicates
        if not items:
            return ""

        bullet_lines = [f"- {tech.value}: {tech.description}" for tech in items]
        return "\n".join([self.intro, *bullet_lines])


@dataclass
class ActionPlan:
    """Structured plan for executing a task intent."""
    steps: List[PlanStep]
    resources: Dict[str, Any]
    estimated_duration: float
    success_metrics: Dict[str, Any]

class PlanParser:
    """Parse heterogeneous LLM responses into canonical plan structures."""

    BULLET_PATTERN = re.compile(r"^\s*(?:\d+[\.)]|[-*])\s+(?P<body>.+)$")
    KEY_VALUE_PATTERN = re.compile(r"^\s*([A-Za-z_\s]+):\s*(.+?)\s*$")
    JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
    CURLY_BLOCK_PATTERN = re.compile(r"(\{.*\})", re.DOTALL)
    DEFAULT_PLAN = {
        "steps": [],
        "resources": {},
        "duration": 0.0,
        "metrics": {},
    }

    def parse(self, response: str) -> Dict[str, Any]:
        """Parse an LLM response into a plan dictionary."""

        if not response or not response.strip():
            logger.warning("Empty response received from LLM; using default plan structure")
            return dict(self.DEFAULT_PLAN)

        json_block = self._extract_json_block(response)
        if json_block:
            parsed = self._try_parse_json(json_block)
            if parsed:
                normalized = self._normalize_plan(parsed)
                if normalized != self.DEFAULT_PLAN:
                    return normalized

        structured = self._parse_structured_text(response)
        if structured:
            return structured

        logger.info("Falling back to default plan parsing for response: %s", response)
        return dict(self.DEFAULT_PLAN)

    def _extract_json_block(self, response: str) -> Optional[str]:
        match = self.JSON_BLOCK_PATTERN.search(response)
        if match:
            return match.group(1)

        curly = self.CURLY_BLOCK_PATTERN.search(response)
        if curly:
            return curly.group(1)
        return None

    @staticmethod
    def _try_parse_json(candidate: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON block from LLM response: %s", exc)
            return None
        if not isinstance(parsed, dict):
            logger.debug("Parsed JSON is not an object: %r", parsed)
            return None
        return parsed

    def _parse_structured_text(self, response: str) -> Optional[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        resources: Optional[Any] = None
        metrics: Optional[Any] = None
        duration: Optional[float] = None

        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue

            bullet_match = self.BULLET_PATTERN.match(line)
            if bullet_match:
                step = self._parse_step_line(bullet_match.group("body"))
                if step:
                    steps.append(step)
                continue

            key_value = self.KEY_VALUE_PATTERN.match(line)
            if not key_value:
                continue

            key = key_value.group(1).strip().lower()
            value = key_value.group(2).strip()

            if key in {"resources", "resource requirements"}:
                resources = self._parse_value(value)
            elif key in {"success metrics", "metrics"}:
                metrics = self._parse_value(value)
            elif key in {"estimated duration", "duration", "time estimate"}:
                duration = self._parse_duration(value)

        if not steps and resources is None and metrics is None and duration is None:
            return None

        return {
            "steps": steps,
            "resources": resources if isinstance(resources, dict) else self._wrap_value(resources, "items"),
            "duration": duration if duration is not None else 0.0,
            "metrics": metrics if isinstance(metrics, dict) else self._wrap_value(metrics, "details"),
        }

    def _parse_step_line(self, body: str) -> Optional[Dict[str, Any]]:
        tool_name = ""
        params: Dict[str, Any] = {}
        description = body

        meta_match = re.search(r"\(([^()]*)\)\s*$", body)
        if meta_match:
            description = body[: meta_match.start()].strip()
            meta = meta_match.group(1)
            tool_name = self._extract_tool(meta)
            params = self._extract_params(meta)

        if not description:
            description = body

        return {
            "description": description.strip(),
            "tool": tool_name,
            "params": params,
        }

    @staticmethod
    def _extract_tool(meta: str) -> str:
        tool_match = re.search(r"tool\s*=\s*([A-Za-z0-9_\-\.]+)", meta)
        if tool_match:
            return tool_match.group(1)
        return ""

    def _extract_params(self, meta: str) -> Dict[str, Any]:
        params_match = re.search(r"params\s*=\s*(\{.*\})", meta)
        if not params_match:
            return {}
        params_text = params_match.group(1)
        try:
            parsed = json.loads(params_text)
        except json.JSONDecodeError:
            logger.debug("Failed to parse params JSON from meta: %s", meta)
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    @staticmethod
    def _parse_value(raw: str) -> Any:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            if len(parts) > 1:
                return parts
            return raw
        return value

    @staticmethod
    def _wrap_value(value: Any, key: str) -> Dict[str, Any]:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return value
        return {key: value}

    @staticmethod
    def _parse_duration(raw: str) -> Optional[float]:
        number_match = re.search(r"(\d+(?:\.\d+)?)", raw)
        if not number_match:
            return None
        try:
            return float(number_match.group(1))
        except ValueError:
            return None

    def _normalize_plan(self, raw_plan: Dict[str, Any]) -> Dict[str, Any]:
        def _normalize_steps(steps: Union[List[Any], Dict[str, Any], None]) -> List[Dict[str, Any]]:
            if steps is None:
                return []
            if isinstance(steps, dict):
                values = steps.values()
            else:
                values = steps  # type: ignore[assignment]

            normalized: List[Dict[str, Any]] = []
            for idx, step in enumerate(values):
                if isinstance(step, dict):
                    payload = dict(step)
                    payload.setdefault("tool", step.get("tool", step.get("action", "")))
                    payload.setdefault("params", step.get("params", {}))
                    normalized.append(payload)
                elif isinstance(step, str):
                    normalized.append({"description": step, "tool": "", "params": {}})
                else:
                    logger.debug("Skipping unsupported step format at index %s: %r", idx, step)
            return normalized

        key_sets = {
            "steps": ("steps", "Steps", "plan", "Plan"),
            "resources": ("resources", "Resources"),
            "duration": ("estimated_duration", "duration", "Duration"),
            "metrics": ("success_metrics", "metrics", "Metrics"),
        }

        resolved: Dict[str, Any] = {}
        for canonical_key, options in key_sets.items():
            match, value = self._first_matching_key(raw_plan, options)
            if match is None:
                value = None
            resolved[canonical_key] = value

        return {
            "steps": _normalize_steps(resolved["steps"]),
            "resources": resolved["resources"] or {},
            "duration": float(resolved["duration"]) if isinstance(resolved["duration"], (int, float)) else 0.0,
            "metrics": resolved["metrics"] or {},
        }

    @staticmethod
    def _first_matching_key(data: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[str], Any]:
        for key in keys:
            if key in data:
                return key, data[key]
        return None, None


class IntentProcessor:
    """Processes high-level intents into structured action plans."""

    def __init__(
        self,
        llm_agent: LlmAgent,
        prompt_guide: Optional[PromptEngineeringGuide] = None,
        plan_parser: Optional[PlanParser] = None,
    ):
        self.llm = llm_agent
        self.prompt_guide = prompt_guide or PromptEngineeringGuide()
        self.plan_parser = plan_parser or PlanParser()

    async def process_intent(
        self,
        intent: TaskIntent,
        techniques: Optional[Iterable[PromptTechnique]] = None,
    ) -> ActionPlan:
        """Convert a high-level intent into an action plan."""
        prompt_sections = [
            "Analyze this task intent and create a structured plan:",
            f"Description: {intent.description}",
            f"Context: {intent.context}",
            f"Constraints: {intent.constraints}",
            "",
            "Generate a detailed plan with:",
            "1. Sequential steps with clear success criteria",
            "2. Required resources and dependencies",
            "3. Estimated duration and complexity",
            "4. Measurable success metrics",
        ]

        if techniques:
            prompt_sections.append("")
            prompt_sections.append(self.prompt_guide.format_instructions(techniques))

        prompt = "\n".join(prompt_sections)
        response = await self.llm.generate_response(prompt)
        plan_data = self.plan_parser.parse(response)

        return ActionPlan(
            steps=plan_data["steps"],
            resources=plan_data["resources"],
            estimated_duration=plan_data["duration"],
            success_metrics=plan_data["metrics"],
        )

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
                tool_name, params, step_payload = self._coerce_step(step)

                tool = self.available_tools.get(tool_name)
                if not tool:
                    raise ValueError(f"Tool not found: {tool_name}")

                invocation = tool(params, context)
                result = await invocation if asyncio.iscoroutine(invocation) else invocation

                yield Event(
                    author="ExecutionEngine",
                    content={"step": step_payload, "result": result},
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

    def _coerce_step(
        self,
        step: PlanStep,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Normalize heterogeneous step payloads into a canonical representation."""

        if isinstance(step, dict):
            tool_name = step.get("tool") or step.get("action")
            if not tool_name:
                raise KeyError("Plan step missing required 'tool' or 'action' field")

            params = step.get("params", {})
            if not isinstance(params, dict):
                raise TypeError("Plan step 'params' must be a mapping")

            payload = dict(step)
            payload.setdefault("tool", tool_name)
            return tool_name, params, payload

        if isinstance(step, PlanStepLike):
            tool_name = step.action
            params = getattr(step, "params", {}) or {}
            if not isinstance(params, dict):
                raise TypeError("Plan step 'params' must be a mapping")

            if is_dataclass(step):
                payload = asdict(step)
            else:
                payload = {
                    "action": tool_name,
                    "params": params,
                }
                for attr in ("description", "requirements", "validation", "retry_policy", "artifacts"):
                    if hasattr(step, attr):
                        payload[attr] = getattr(step, attr)

            payload.setdefault("tool", tool_name)
            return tool_name, params, payload

        raise TypeError(f"Plan step must be a mapping or PlanStepLike instance, got {type(step)!r}")

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