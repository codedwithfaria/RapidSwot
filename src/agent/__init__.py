"""
Main agent implementation combining all components.
"""
from typing import Optional, Dict, Any
from .browser import BrowserController
from .llm import BaseLLM, LLMBackend
from ..memory import MemorySystem
from ..sequential import SequentialPlanner

class Agent:
    def __init__(
        self,
        task: str,
        llm: Optional[BaseLLM] = None,
        memory: Optional[MemorySystem] = None,
        planner: Optional[SequentialPlanner] = None
    ):
        self.task = task
        self.llm = llm or LLMBackend.gemini()
        self.memory = memory or MemorySystem()
        self.planner = planner or SequentialPlanner()
        self.browser = BrowserController()
        self.context_window = 2_000_000  # Maximum token context window
        
    async def run(self) -> Dict[str, Any]:
        """
        Execute the assigned task.
        
        Returns:
            Task results and execution metadata
        """
        try:
            # Plan task steps
            steps = await self.planner.plan_task(self.task, self.llm)
            
            # Initialize browser
            await self.browser.start()
            
            # Execute steps
            results = []
            for step in steps:
                result = await self.planner.execute_step(step, self)
                results.append(result)
                
                # Store in memory
                self.memory.store_knowledge(
                    subject=step.action,
                    predicate="result",
                    object_=result
                )
                
            return {
                "task": self.task,
                "steps": len(steps),
                "results": results
            }
            
        finally:
            # Cleanup
            await self.browser.stop()