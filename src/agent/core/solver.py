"""
Problem-solving framework for RapidSwot agent system.
Implements GAIA-like evaluation and solution generation capabilities.
"""
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import asyncio
import logging
from enum import Enum

from .agent import TaskIntent, ActionPlan
from .domains import DomainRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProblemType(Enum):
    """Types of problems the agent can solve."""
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    OPTIMIZATION = "optimization"
    CREATION = "creation"
    INTEGRATION = "integration"

@dataclass
class Problem:
    """Representation of a problem to solve."""
    type: ProblemType
    description: str
    domain: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Solution:
    """Representation of a problem solution."""
    problem: Problem
    plan: ActionPlan
    estimated_success_rate: float
    resource_requirements: Dict[str, Any]
    validation_steps: List[Dict[str, Any]]

class ProblemSolver:
    """Core problem-solving engine."""
    
    def __init__(
        self,
        domain_registry: DomainRegistry,
        validation_threshold: float = 0.8
    ):
        self.domain_registry = domain_registry
        self.validation_threshold = validation_threshold
    
    async def analyze_problem(
        self,
        problem: Problem
    ) -> Dict[str, Any]:
        """Analyze a problem and determine solution approach."""
        # Get domain-specific adapter
        adapter = self.domain_registry.get_adapter(problem.domain)
        if not adapter:
            raise ValueError(f"No adapter found for domain: {problem.domain}")
        
        # Analyze available tools and workflows
        tools = adapter.tools
        workflows = adapter.workflows
        
        # Match problem type to solution patterns
        patterns = self._match_solution_patterns(
            problem.type,
            tools,
            workflows
        )
        
        return {
            "patterns": patterns,
            "tools": list(tools.keys()),
            "workflows": list(workflows.keys())
        }
    
    def _match_solution_patterns(
        self,
        problem_type: ProblemType,
        tools: Dict[str, callable],
        workflows: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Match problem type to potential solution patterns."""
        patterns = []
        
        if problem_type == ProblemType.AUTOMATION:
            # Look for automation workflows
            for name, workflow in workflows.items():
                if any(
                    step["tool"] in tools 
                    for step in workflow
                ):
                    patterns.append({
                        "type": "workflow",
                        "name": name,
                        "steps": len(workflow)
                    })
        
        elif problem_type == ProblemType.OPTIMIZATION:
            # Look for optimization tools
            for name, tool in tools.items():
                if "optimize" in name.lower():
                    patterns.append({
                        "type": "tool",
                        "name": name
                    })
        
        return patterns
    
    async def generate_solution(
        self,
        problem: Problem
    ) -> Solution:
        """Generate a solution for the given problem."""
        # Analyze the problem
        analysis = await self.analyze_problem(problem)
        
        # Create action plan
        plan = await self._create_action_plan(
            problem,
            analysis["patterns"]
        )
        
        # Estimate success rate
        success_rate = await self._estimate_success_rate(
            problem,
            plan
        )
        
        # Define validation steps
        validation = self._define_validation_steps(
            problem,
            plan
        )
        
        return Solution(
            problem=problem,
            plan=plan,
            estimated_success_rate=success_rate,
            resource_requirements=self._calculate_resources(plan),
            validation_steps=validation
        )
    
    async def _create_action_plan(
        self,
        problem: Problem,
        patterns: List[Dict[str, Any]]
    ) -> ActionPlan:
        """Create an action plan based on solution patterns."""
        steps = []
        resources = {}
        
        for pattern in patterns:
            if pattern["type"] == "workflow":
                # Get workflow steps
                adapter = self.domain_registry.get_adapter(problem.domain)
                workflow = adapter.workflows[pattern["name"]]
                steps.extend(workflow)
                
                # Add required resources
                resources.update(
                    self._get_workflow_resources(workflow)
                )
            
            elif pattern["type"] == "tool":
                # Add tool step
                steps.append({
                    "tool": pattern["name"],
                    "params": {}
                })
        
        return ActionPlan(
            steps=steps,
            resources=resources,
            estimated_duration=self._estimate_duration(steps),
            success_metrics=problem.success_criteria
        )
    
    def _get_workflow_resources(
        self,
        workflow: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate resources needed for a workflow."""
        resources = {}
        for step in workflow:
            # Add resource requirements based on tool
            tool_name = step["tool"]
            resources[tool_name] = {
                "type": "tool",
                "permissions": ["execute"]
            }
        return resources
    
    def _estimate_duration(
        self,
        steps: List[Dict[str, Any]]
    ) -> float:
        """Estimate duration for executing steps."""
        # Simple duration estimation
        return len(steps) * 5.0  # 5 seconds per step
    
    async def _estimate_success_rate(
        self,
        problem: Problem,
        plan: ActionPlan
    ) -> float:
        """Estimate likelihood of solution success."""
        # Basic success rate estimation
        base_rate = 0.7  # Base 70% success rate
        
        # Adjust based on plan completeness
        if len(plan.steps) > 0:
            base_rate += 0.1
        
        # Adjust based on resource availability
        if all(
            resource in plan.resources
            for resource in problem.constraints.get("required_resources", [])
        ):
            base_rate += 0.1
        
        return min(base_rate, 1.0)
    
    def _define_validation_steps(
        self,
        problem: Problem,
        plan: ActionPlan
    ) -> List[Dict[str, Any]]:
        """Define steps to validate the solution."""
        validation_steps = []
        
        # Add validation for each success criterion
        for criterion, value in problem.success_criteria.items():
            validation_steps.append({
                "type": "validation",
                "criterion": criterion,
                "expected_value": value,
                "validation_method": "comparison"
            })
        
        return validation_steps
    
    def _calculate_resources(
        self,
        plan: ActionPlan
    ) -> Dict[str, Any]:
        """Calculate required resources for the plan."""
        resources = {}
        
        # Aggregate resources from all steps
        for step in plan.steps:
            tool_name = step["tool"]
            if tool_name not in resources:
                resources[tool_name] = {
                    "type": "tool",
                    "count": 1,
                    "permissions": ["execute"]
                }
            else:
                resources[tool_name]["count"] += 1
        
        return resources

class SolutionExecutor:
    """Executes and validates solutions."""
    
    def __init__(
        self,
        domain_registry: DomainRegistry,
        validation_threshold: float = 0.8
    ):
        self.domain_registry = domain_registry
        self.validation_threshold = validation_threshold
    
    async def execute_solution(
        self,
        solution: Solution
    ) -> Dict[str, Any]:
        """Execute a solution and validate results."""
        results = []
        success = True
        
        try:
            # Get domain adapter
            adapter = self.domain_registry.get_adapter(
                solution.problem.domain
            )
            
            # Execute each step
            for step in solution.plan.steps:
                result = await self._execute_step(step, adapter)
                results.append(result)
                
                # Validate step result
                if not await self._validate_step(
                    step,
                    result,
                    solution.problem.success_criteria
                ):
                    success = False
                    break
            
            return {
                "success": success,
                "results": results,
                "validation": await self._validate_solution(
                    solution,
                    results
                )
            }
            
        except Exception as e:
            logger.error(f"Solution execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": results
            }
    
    async def _execute_step(
        self,
        step: Dict[str, Any],
        adapter: Any
    ) -> Dict[str, Any]:
        """Execute a single solution step."""
        tool = adapter.tools.get(step["tool"])
        if not tool:
            raise ValueError(f"Tool not found: {step['tool']}")
        
        return await tool(step["params"], {})
    
    async def _validate_step(
        self,
        step: Dict[str, Any],
        result: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> bool:
        """Validate the result of a solution step."""
        # Basic validation - check for error status
        if result.get("status") == "error":
            return False
        
        # Check against success criteria
        for criterion, expected in criteria.items():
            if criterion in result:
                if result[criterion] != expected:
                    return False
        
        return True
    
    async def _validate_solution(
        self,
        solution: Solution,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate the overall solution results."""
        validation_results = []
        
        for step in solution.validation_steps:
            validation_results.append(
                await self._run_validation(step, results)
            )
        
        # Calculate overall validation score
        total_validations = len(validation_results)
        passed_validations = sum(
            1 for result in validation_results
            if result["passed"]
        )
        
        validation_score = (
            passed_validations / total_validations
            if total_validations > 0
            else 0.0
        )
        
        return {
            "score": validation_score,
            "passed": validation_score >= self.validation_threshold,
            "details": validation_results
        }
    
    async def _run_validation(
        self,
        validation_step: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run a single validation step."""
        criterion = validation_step["criterion"]
        expected = validation_step["expected_value"]
        
        # Look for criterion in results
        actual = None
        for result in results:
            if criterion in result:
                actual = result[criterion]
                break
        
        passed = (actual == expected) if actual is not None else False
        
        return {
            "criterion": criterion,
            "expected": expected,
            "actual": actual,
            "passed": passed
        }