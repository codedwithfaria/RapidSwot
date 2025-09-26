"""
Autonomous agent capabilities for AG-UI protocol.
"""
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from dataclasses import dataclass, field
import asyncio
import json
import logging
from datetime import datetime
from . import AGUIProtocol, AGUIEvent, EventType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskContext:
    """Context for autonomous task execution."""
    task_id: str
    session_id: str
    user_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def update_state(self, updates: Dict[str, Any]):
        """Update task state."""
        self.state.update(updates)
        
    def add_artifact(self, key: str, artifact: Any):
        """Add a task artifact."""
        self.artifacts[key] = artifact

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Autonomous task definition."""
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }

class AutonomousAgent:
    """Agent capable of autonomous task execution."""
    
    def __init__(self, protocol: AGUIProtocol, name: str = "AutonomousAgent"):
        self.protocol = protocol
        self.name = name
        self.active_tasks: Dict[str, Task] = {}
        self._task_queue = asyncio.Queue()
        self._background_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start the agent's background task processing."""
        self._background_tasks.append(
            asyncio.create_task(self._process_task_queue())
        )
    
    async def stop(self):
        """Stop the agent and cleanup resources."""
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
    
    async def submit_task(
        self,
        name: str,
        description: str,
        context: Dict[str, Any]
    ) -> str:
        """Submit a new task for execution."""
        task_id = f"{name}_{len(self.active_tasks)}"
        task = Task(name=name, description=description)
        self.active_tasks[task_id] = task
        
        # Create task context
        task_context = TaskContext(
            task_id=task_id,
            session_id=context.get("session_id", "default"),
            user_id=context.get("user_id", "default"),
            metadata=context
        )
        
        # Queue task for execution
        await self._task_queue.put((task, task_context))
        
        await self.protocol.emit(AGUIEvent(
            type=EventType.TOOL_START,
            data={
                "task_id": task_id,
                "task": task.to_dict()
            }
        ))
        
        return task_id
    
    async def _process_task_queue(self):
        """Process tasks from the queue."""
        while True:
            try:
                task, context = await self._task_queue.get()
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow().isoformat()
                
                try:
                    # Execute task steps
                    result = await self._execute_task_steps(task, context)
                    
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.utcnow().isoformat()
                    
                    await self.protocol.emit(AGUIEvent(
                        type=EventType.TASK_COMPLETE,
                        data={
                            "task_id": context.task_id,
                            "task": task.to_dict(),
                            "result": result
                        }
                    ))
                    
                except Exception as e:
                    logger.error(f"Task execution error: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.utcnow().isoformat()
                    
                    await self.protocol.emit(AGUIEvent(
                        type=EventType.ERROR,
                        data={
                            "task_id": context.task_id,
                            "message": str(e),
                            "task": task.to_dict()
                        }
                    ))
                
                finally:
                    self._task_queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task queue processing error: {e}")
                await asyncio.sleep(1)  # Prevent tight error loops
    
    async def _execute_task_steps(
        self,
        task: Task,
        context: TaskContext
    ) -> Any:
        """Execute task steps with progress updates."""
        # Simulate task steps - override this in subclasses
        total_steps = 5
        for step in range(total_steps):
            # Update progress
            task.progress = (step + 1) / total_steps
            
            await self.protocol.emit(AGUIEvent(
                type=EventType.TOOL_START,
                data={
                    "task_id": context.task_id,
                    "step": step + 1,
                    "total_steps": total_steps,
                    "progress": task.progress,
                    "task": task.to_dict()
                }
            ))
            
            # Simulate work
            await asyncio.sleep(1)
            
            # Update context state for demonstration
            context.update_state({
                f"step_{step+1}_completed": True,
                "current_progress": task.progress
            })
        
        return {"status": "success", "message": "Task completed successfully"}

class MultiModalAgent(AutonomousAgent):
    """Agent with multi-modal processing capabilities."""
    
    async def process_text(self, text: str, context: Dict[str, Any]) -> str:
        """Process text input."""
        task_id = await self.submit_task(
            name="text_processing",
            description=f"Processing text: {text[:50]}...",
            context=context
        )
        return task_id
    
    async def process_image(self, image_data: bytes, context: Dict[str, Any]) -> str:
        """Process image input."""
        task_id = await self.submit_task(
            name="image_processing",
            description="Processing image data",
            context=context
        )
        return task_id
    
    async def process_code(self, code: str, context: Dict[str, Any]) -> str:
        """Process code input."""
        task_id = await self.submit_task(
            name="code_processing",
            description=f"Processing code: {code[:50]}...",
            context=context
        )
        return task_id

class PersistentAgent(AutonomousAgent):
    """Agent capable of persistent task execution."""
    
    def __init__(self, protocol: AGUIProtocol, storage_path: str = None):
        super().__init__(protocol)
        self.storage_path = storage_path
        
    async def save_state(self):
        """Save agent state for persistence."""
        if self.storage_path:
            state = {
                "active_tasks": {
                    task_id: task.to_dict()
                    for task_id, task in self.active_tasks.items()
                }
            }
            # Save state to storage
            
    async def load_state(self):
        """Load persisted agent state."""
        if self.storage_path:
            # Load state from storage
            pass
            
    async def resume_tasks(self):
        """Resume previously persisted tasks."""
        await self.load_state()
        # Resume incomplete tasks