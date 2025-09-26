"""
Advanced task planning and execution module with autonomous capabilities.
Supports multi-modal processing, parallel execution, and continuous operation.
"""
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of tasks that can be planned and executed."""
    AUTOMATION = "automation"
    ANALYSIS = "analysis"
    DEVELOPMENT = "development"
    CONTENT_CREATION = "content_creation"
    DATA_PROCESSING = "data_processing"
    INTEGRATION = "integration"

@dataclass
class TaskContext:
    """Rich context for task execution."""
    task_id: str
    type: TaskType
    description: str
    input_data: Dict[str, Any]
    settings: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Step:
    """Enhanced step with rich metadata and validation."""
    action: str
    params: Dict[str, Any]
    description: str
    requirements: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_attempts": 3,
        "delay": 1.0
    })
    artifacts: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Detailed result of step execution."""
    step: Step
    success: bool
    output: Any
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class SequentialPlanner:
    """Advanced planner with autonomous execution capabilities."""
    
    def __init__(self):
        self.steps: List[Step] = []
        self.active_contexts: Dict[str, TaskContext] = {}
        self.execution_history: Dict[str, List[ExecutionResult]] = {}
        self._tools: Dict[str, callable] = {}
        self._llm = None  # Will be set during initialization
    
    def register_tool(self, name: str, tool: callable):
        """Register a tool for task execution."""
        self._tools[name] = tool
    
    async def plan_task(
        self,
        task: str,
        task_type: TaskType,
        llm: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Step]:
        """
        Create a sophisticated task plan using LLM.
        
        Args:
            task: Task description
            task_type: Type of task to plan
            llm: LLM instance for planning
            context: Additional context for planning
        
        Returns:
            List of planned steps with rich metadata
        """
        self._llm = llm
        task_id = f"task_{len(self.active_contexts)}"
        
        # Create task context
        task_context = TaskContext(
            task_id=task_id,
            type=task_type,
            description=task,
            input_data=context or {},
        )
        self.active_contexts[task_id] = task_context
        
        # Generate detailed plan using LLM
        plan = await llm.generate(f"""
        Create a detailed execution plan for this task:
        Type: {task_type.value}
        Description: {task}
        Context: {context}
        
        Available tools: {list(self._tools.keys())}
        
        For each step, provide:
        1. action: specific tool/action to use
        2. params: detailed parameters needed
        3. description: purpose and expected outcome
        4. requirements: data/resources needed
        5. validation: success criteria
        6. retry_policy: error handling strategy
        
        Consider:
        - Dependencies between steps
        - Resource requirements
        - Error handling
        - Success validation
        - Performance optimization
        """)
        
        # Parse LLM response into enhanced Step objects
        self.steps = await self._parse_plan(plan, task_context)
        return self.steps
    
    async def _parse_plan(
        self,
        plan: str,
        context: TaskContext
    ) -> List[Step]:
        """Parse LLM plan into structured steps."""
        # Use LLM to parse the plan into structured format
        structured_plan = await self._llm.generate(f"""
        Parse this plan into a structured format:
        {plan}
        
        For each step, extract:
        - Action name (must match available tools)
        - Parameters with types
        - Description and purpose
        - Required resources
        - Validation criteria
        - Retry settings
        """)
        
        # Convert structured format to Step objects
        # This is a simplified implementation
        return [
            Step(
                action="placeholder",
                params={},
                description="Placeholder step",
                requirements={},
                validation={},
                retry_policy={"max_attempts": 3, "delay": 1.0}
            )
        ]
    
    async def execute_plan(
        self,
        steps: List[Step],
        agent: Any,
        context: Optional[TaskContext] = None
    ) -> AsyncGenerator[ExecutionResult, None]:
        """
        Execute a plan with advanced features.
        
        Args:
            steps: Steps to execute
            agent: Agent instance for execution
            context: Task context for execution
        
        Yields:
            Execution results for each step
        """
        # Validate and prepare execution
        if not steps:
            raise ValueError("No steps to execute")
        
        # Map actions to tools/methods
        action_map = {
            "navigate": agent.browser.navigate,
            "click": agent.browser.click,
            "type": agent.browser.type,
            "extract": agent.browser.extract_content,
            **self._tools  # Include registered tools
        }
        
        # Track dependencies and parallel execution opportunities
        completed_steps: Set[str] = set()
        pending_steps = steps.copy()
        
        while pending_steps:
            # Find steps that can be executed (all dependencies met)
            executable_steps = [
                step for step in pending_steps
                if self._dependencies_met(step, completed_steps)
            ]
            
            if not executable_steps:
                raise ValueError("Circular dependency detected")
            
            # Execute steps in parallel if possible
            execution_tasks = [
                self._execute_step(
                    step,
                    action_map,
                    context
                )
                for step in executable_steps
            ]
            
            results = await asyncio.gather(
                *execution_tasks,
                return_exceptions=True
            )
            
            # Process results
            for step, result in zip(executable_steps, results):
                if isinstance(result, Exception):
                    # Handle execution error
                    error_result = ExecutionResult(
                        step=step,
                        success=False,
                        output=None,
                        error=str(result)
                    )
                    yield error_result
                    
                    # Store in history
                    if context:
                        history = self.execution_history.setdefault(
                            context.task_id,
                            []
                        )
                        history.append(error_result)
                    
                    # Check retry policy
                    if not await self._handle_error(step, result, context):
                        return
                else:
                    # Success case
                    yield result
                    
                    # Update tracking
                    completed_steps.add(id(step))
                    pending_steps.remove(step)
                    
                    # Store in history
                    if context:
                        history = self.execution_history.setdefault(
                            context.task_id,
                            []
                        )
                        history.append(result)
    
    async def _execute_step(
        self,
        step: Step,
        action_map: Dict[str, callable],
        context: Optional[TaskContext]
    ) -> ExecutionResult:
        """Execute a single step with full lifecycle management."""
        start_time = datetime.utcnow()
        
        try:
            # Validate requirements
            if not self._validate_requirements(step, context):
                raise ValueError("Step requirements not met")
            
            # Get the action
            action = action_map.get(step.action)
            if not action:
                raise ValueError(f"Unknown action: {step.action}")
            
            # Execute with retry policy
            attempts = 0
            while attempts < step.retry_policy["max_attempts"]:
                try:
                    # Execute action
                    output = await action(**step.params)
                    
                    # Validate output
                    if self._validate_output(output, step.validation):
                        return ExecutionResult(
                            step=step,
                            success=True,
                            output=output,
                            metrics={
                                "duration": (datetime.utcnow() - start_time).total_seconds(),
                                "attempts": attempts + 1
                            }
                        )
                    
                except Exception as e:
                    logger.error(f"Step execution error: {e}")
                    if attempts + 1 >= step.retry_policy["max_attempts"]:
                        raise
                
                attempts += 1
                await asyncio.sleep(step.retry_policy["delay"])
            
            raise ValueError("Max retry attempts reached")
            
        except Exception as e:
            return ExecutionResult(
                step=step,
                success=False,
                output=None,
                error=str(e),
                metrics={
                    "duration": (datetime.utcnow() - start_time).total_seconds(),
                    "attempts": attempts + 1
                }
            )
    
    def _dependencies_met(self, step: Step, completed: Set[str]) -> bool:
        """Check if all step dependencies are met."""
        return all(
            req_id in completed
            for req_id in step.requirements.get("dependencies", [])
        )
    
    def _validate_requirements(
        self,
        step: Step,
        context: Optional[TaskContext]
    ) -> bool:
        """Validate step requirements are met."""
        if not context:
            return True
        
        for req_name, req_value in step.requirements.items():
            if req_name == "dependencies":
                continue
            if req_name not in context.state:
                return False
            if context.state[req_name] != req_value:
                return False
        
        return True
    
    def _validate_output(self, output: Any, validation: Dict[str, Any]) -> bool:
        """Validate step output against criteria."""
        for criterion, expected in validation.items():
            if criterion not in output:
                return False
            if output[criterion] != expected:
                return False
        return True
    
    async def _handle_error(
        self,
        step: Step,
        error: Exception,
        context: Optional[TaskContext]
    ) -> bool:
        """Handle step execution error."""
        # Log error
        logger.error(f"Error executing step {step.action}: {error}")
        
        # Update context state if available
        if context:
            context.state["last_error"] = str(error)
            context.state["error_step"] = step.action
        
        # Check if should retry based on policy
        return False  # Stop execution on error