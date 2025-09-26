"""
Task planning and execution module.
"""
from typing import List, Dict, Any
from dataclasses import dataclass
import asyncio

@dataclass
class Step:
    action: str
    params: Dict[str, Any]
    description: str

class SequentialPlanner:
    def __init__(self):
        self.steps: List[Step] = []
        
    async def plan_task(self, task: str, llm: Any) -> List[Step]:
        """
        Break down a task into sequential steps using LLM.
        
        Args:
            task: Task description
            llm: LLM instance for planning
            
        Returns:
            List of planned steps
        """
        # Use LLM to break down task into steps
        plan = await llm.generate(f"""
        Break down this task into sequential steps:
        {task}
        
        Return steps in this format:
        1. action: what to do
           params: any parameters needed
           description: why this step is needed
        """)
        
        # Parse LLM response into Step objects
        # Implementation details...
        
        return self.steps
        
    async def execute_step(self, step: Step, agent: Any):
        """
        Execute a single planned step.
        
        Args:
            step: Step to execute
            agent: Agent instance for execution
        """
        # Map step actions to agent methods
        action_map = {
            "navigate": agent.browser.navigate,
            "click": agent.browser.click,
            "type": agent.browser.type,
            "extract": agent.browser.extract_content
        }
        
        if step.action in action_map:
            await action_map[step.action](**step.params)
        else:
            raise ValueError(f"Unknown action: {step.action}")