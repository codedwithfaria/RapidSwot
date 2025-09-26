"""
Advanced browser agent with chaining and persistent sessions.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import asyncio

from .browser import BrowserController
from .llm import BaseLLM, LLMBackend
from ..memory import MemorySystem
from ..sequential import SequentialPlanner

@dataclass
class AgentConfig:
    use_vision: bool = True
    vision_detail_level: str = 'auto'
    max_actions_per_step: int = 10
    max_failures: int = 3
    final_response_after_failure: bool = True
    use_thinking: bool = True
    flash_mode: bool = False
    llm_timeout: int = 90
    step_timeout: int = 120
    directly_open_url: bool = True
    calculate_cost: bool = False
    display_files_in_done_text: bool = True
    keep_alive: bool = False
    max_history_items: Optional[int] = None

@dataclass
class Agent:
    task: str
    llm: Optional[BaseLLM] = None
    memory: Optional[MemorySystem] = None
    planner: Optional[SequentialPlanner] = None
    config: AgentConfig = field(default_factory=AgentConfig)
    browser_session: Optional[BrowserController] = None
    
    def __post_init__(self):
        self.llm = self.llm or LLMBackend.gemini()
        self.memory = self.memory or MemorySystem()
        self.planner = self.planner or SequentialPlanner()
        self.browser = self.browser_session or BrowserController()
        self.tasks: List[str] = [self.task]
        
    async def run(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the current task with optional step limit.
        
        Args:
            max_steps: Maximum number of steps to execute
            
        Returns:
            Task execution results
        """
        try:
            if not self.browser_session and not self.config.keep_alive:
                await self.browser.start()
                
            steps = await self.planner.plan_task(self.task, self.llm)
            if max_steps:
                steps = steps[:max_steps]
                
            results = []
            for step in steps:
                if self.config.use_thinking:
                    self.memory.store_knowledge(
                        "thinking",
                        "current",
                        f"Planning to {step.description}"
                    )
                    
                result = await self.planner.execute_step(step, self)
                results.append(result)
                
                if not self.config.flash_mode:
                    self.memory.store_knowledge(
                        step.action,
                        "result",
                        result
                    )
                    
            return {
                "task": self.task,
                "steps_executed": len(results),
                "results": results,
                "has_more": bool(self.tasks[1:])
            }
            
        finally:
            if not self.config.keep_alive and not self.browser_session:
                await self.browser.stop()
                
    def add_new_task(self, task: str):
        """Add a follow-up task to the queue."""
        self.tasks.append(task)
        
    async def run_all(self) -> List[Dict[str, Any]]:
        """Execute all queued tasks in sequence."""
        results = []
        while self.tasks:
            self.task = self.tasks.pop(0)
            result = await self.run()
            results.append(result)
        return results
        
    def run_sync(self) -> Dict[str, Any]:
        """Synchronous version of run() for easier usage."""
        return asyncio.run(self.run())